//! Gate 1: does choosing a model actually send the request to *that model's*
//! endpoint, with *that model's* key?
//!
//! Every earlier round verified the identity decision — which catalog key a
//! string resolves to — and then reasoned that the endpoint must follow.
//! Reasoning is what produced the original bug report: a freshly configured
//! model whose requests went to 智谱's non-coding endpoint. This test replaces
//! the reasoning with two mock servers and a request count.
//!
//! The catalog is the accident's own shape: two entries sharing the upstream
//! slug `glm-4.6`, on different endpoints, with keys that are not
//! interchangeable (crossing them is a 401 in production).
//!
//! The chain under test is entirely production code:
//!     resolve_requested_model → selection.entry()
//!         → resolve_credentials → sampling_config_for_model → SamplerActor
//! Nothing is hand-assembled between the identity and the socket.

use std::net::SocketAddr;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use agent_client_protocol as acp;
use axum::Router;
use axum::extract::State;
use axum::http::HeaderMap;
use axum::response::sse::{Event, Sse};
use axum::routing::post;
use futures_util::stream;
use indexmap::IndexMap;
use tokio::sync::{mpsc, oneshot};

use xai_grok_sampler::{RequestId, RetryPolicy, SamplerActor, SamplingEvent};
use xai_grok_sampling_types::{ContentPart, ConversationItem, ConversationRequest, UserItem};
use xai_grok_shell::agent::config::{
    ModelEntry, ModelInfo, resolve_credentials, sampling_config_for_model,
};
use xai_grok_shell::agent::models::resolve_requested_model;

/// A mock endpoint that counts what reaches it and remembers the credentials
/// it was offered. Both halves matter: a request landing on the wrong host is
/// one failure mode, the right host receiving the other entry's key is
/// another, and only counting requests would miss the second.
#[derive(Clone, Default)]
struct Probe {
    hits: Arc<AtomicU32>,
    auth_headers: Arc<Mutex<Vec<String>>>,
}

impl Probe {
    fn hits(&self) -> u32 {
        self.hits.load(Ordering::SeqCst)
    }

    fn seen_auth(&self) -> Vec<String> {
        self.auth_headers.lock().unwrap().clone()
    }
}

struct MockEndpoint {
    addr: SocketAddr,
    probe: Probe,
    shutdown_tx: oneshot::Sender<()>,
}

impl MockEndpoint {
    async fn spawn() -> Self {
        let probe = Probe::default();
        let app = Router::new()
            .route("/v1/chat/completions", post(handler))
            .with_state(probe.clone());
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let (shutdown_tx, shutdown_rx) = oneshot::channel::<()>();
        tokio::spawn(async move {
            let _ = axum::serve(listener, app)
                .with_graceful_shutdown(async move {
                    let _ = shutdown_rx.await;
                })
                .await;
        });
        tokio::time::sleep(Duration::from_millis(20)).await;
        Self {
            addr,
            probe,
            shutdown_tx,
        }
    }

    fn base_url(&self) -> String {
        format!("http://{}/v1", self.addr)
    }

    fn shutdown(self) {
        let _ = self.shutdown_tx.send(());
    }
}

async fn handler(
    State(probe): State<Probe>,
    headers: HeaderMap,
) -> Sse<impl futures_util::Stream<Item = Result<Event, std::convert::Infallible>>> {
    probe.hits.fetch_add(1, Ordering::SeqCst);
    if let Some(auth) = headers.get("authorization").and_then(|v| v.to_str().ok()) {
        probe.auth_headers.lock().unwrap().push(auth.to_owned());
    }
    let chunk = serde_json::json!({
        "id": "chatcmpl-routing-test",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": "glm-4.6",
        "choices": [{
            "index": 0,
            "delta": { "role": "assistant", "content": "ok" },
            "finish_reason": "stop"
        }]
    });
    Sse::new(stream::iter(vec![
        Ok(Event::default().data(chunk.to_string())),
        Ok(Event::default().data("[DONE]")),
    ]))
}

const OPEN_KEY: &str = "key-for-open-platform";
const CODING_KEY: &str = "key-for-coding-plan";

fn entry(base_url: &str, api_key: &str) -> ModelEntry {
    let mut info = ModelInfo::fallback("glm-4.6");
    info.base_url = base_url.to_owned();
    info.max_completion_tokens = Some(64);
    ModelEntry {
        info,
        api_key: Some(api_key.to_owned()),
        env_key: None,
        api_base_url: None,
    }
}

/// Catalog built in the given order — order is a parameter because a
/// position-dependent resolver (the `.rev()` last-wins scan this work removed)
/// only misbehaves when the order changes.
fn catalog(
    order: [(&str, &str, &str); 2],
) -> (
    IndexMap<String, ModelEntry>,
    IndexMap<acp::ModelId, acp::ModelInfo>,
) {
    let mut models = IndexMap::new();
    for (key, base_url, api_key) in order {
        models.insert(key.to_owned(), entry(base_url, api_key));
    }
    let available = models
        .keys()
        .map(|k| {
            let id = acp::ModelId::new(k.clone());
            (id.clone(), acp::ModelInfo::new(id, k.clone()))
        })
        .collect();
    (models, available)
}

async fn drain_until_terminal(rx: &mut mpsc::UnboundedReceiver<SamplingEvent>) {
    let deadline = tokio::time::Instant::now() + Duration::from_secs(10);
    loop {
        let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
        if remaining.is_zero() {
            panic!("sampler produced no terminal event within the timeout");
        }
        match tokio::time::timeout(remaining, rx.recv()).await {
            Ok(Some(SamplingEvent::Completed { .. })) | Ok(None) => return,
            Ok(Some(SamplingEvent::Failed { error, .. })) => {
                // A transport/protocol failure is fine for this test — the
                // assertion is about *where* the bytes went, not whether the
                // mock spoke perfect SSE. What must never happen is the
                // request going to the other endpoint.
                eprintln!("sampler reported failure (not fatal to this test): {error:?}");
                return;
            }
            Ok(Some(_)) => continue,
            Err(_) => panic!("sampler produced no terminal event within the timeout"),
        }
    }
}

/// Resolve `requested` through the production chain and actually issue one
/// request with the resulting config.
async fn send_one(order: [(&str, &str, &str); 2], requested: &str) {
    let (models, available) = catalog(order);
    let selection = resolve_requested_model(&models, &available, requested)
        .unwrap_or_else(|e| panic!("{requested} must resolve exactly: {e:?}"));
    let creds = resolve_credentials(selection.entry(), None);
    let mut cfg = sampling_config_for_model(selection.entry(), creds, None, None, None, None);
    // One attempt only: a retry would inflate the hit count and blur the
    // "exactly once" assertion.
    cfg.max_retries = Some(0);
    cfg.idle_timeout_secs = Some(5);

    let (event_tx, mut event_rx) = mpsc::unbounded_channel();
    let handle = SamplerActor::spawn(cfg, RetryPolicy::default(), event_tx);
    handle.submit(
        RequestId::from("gate1"),
        ConversationRequest {
            items: vec![ConversationItem::User(UserItem {
                content: vec![ContentPart::Text {
                    text: Arc::<str>::from("ping"),
                }],
                synthetic_reason: None,
                ..Default::default()
            })],
            ..Default::default()
        },
    );
    drain_until_terminal(&mut event_rx).await;
}

/// Selecting the Coding Plan entry must reach the Coding Plan endpoint, once,
/// with the Coding Plan key — and the open-platform endpoint must see nothing
/// at all.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn selecting_coding_plan_reaches_only_the_coding_endpoint() {
    let open = MockEndpoint::spawn().await;
    let coding = MockEndpoint::spawn().await;

    send_one(
        [
            ("glm-open", &open.base_url(), OPEN_KEY),
            ("glm-coding", &coding.base_url(), CODING_KEY),
        ],
        "glm-coding",
    )
    .await;

    assert_eq!(coding.probe.hits(), 1, "chosen endpoint must receive exactly one request");
    assert_eq!(open.probe.hits(), 0, "the other endpoint must receive nothing");
    let auth = coding.probe.seen_auth();
    assert!(
        auth.iter().any(|h| h.contains(CODING_KEY)),
        "chosen endpoint did not receive its own key: {auth:?}"
    );
    assert!(
        !auth.iter().any(|h| h.contains(OPEN_KEY)),
        "the other entry's key crossed over: {auth:?}"
    );

    open.shutdown();
    coding.shutdown();
}

/// The mirror image. Testing one direction only is how a position-dependent
/// resolver survives a test suite.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn selecting_open_platform_reaches_only_the_open_endpoint() {
    let open = MockEndpoint::spawn().await;
    let coding = MockEndpoint::spawn().await;

    send_one(
        [
            ("glm-open", &open.base_url(), OPEN_KEY),
            ("glm-coding", &coding.base_url(), CODING_KEY),
        ],
        "glm-open",
    )
    .await;

    assert_eq!(open.probe.hits(), 1);
    assert_eq!(coding.probe.hits(), 0);
    let auth = open.probe.seen_auth();
    assert!(auth.iter().any(|h| h.contains(OPEN_KEY)), "{auth:?}");
    assert!(!auth.iter().any(|h| h.contains(CODING_KEY)), "{auth:?}");

    open.shutdown();
    coding.shutdown();
}

/// Same request, catalog declared in the opposite order. A resolver that picks
/// by position would send this one to the other host.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn catalog_order_does_not_change_where_the_request_lands() {
    let open = MockEndpoint::spawn().await;
    let coding = MockEndpoint::spawn().await;

    send_one(
        [
            ("glm-coding", &coding.base_url(), CODING_KEY),
            ("glm-open", &open.base_url(), OPEN_KEY),
        ],
        "glm-coding",
    )
    .await;

    assert_eq!(coding.probe.hits(), 1);
    assert_eq!(open.probe.hits(), 0);
    // Key too, not just the host: an order-dependent bug that swaps
    // credentials while keeping endpoints straight would pass a hit-count
    // check. The pair travels together or the test fails.
    let auth = coding.probe.seen_auth();
    assert!(auth.iter().any(|h| h.contains(CODING_KEY)), "{auth:?}");
    assert!(!auth.iter().any(|h| h.contains(OPEN_KEY)), "{auth:?}");

    open.shutdown();
    coding.shutdown();
}
