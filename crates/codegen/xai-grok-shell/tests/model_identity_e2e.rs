//! v0.18.6 引擎身份链集成测试：从磁盘上一份真实的旧会话记录，走到一次真实
//! 的 HTTP 请求。
//!
//! 刻意不叫 E2E。它没有经过 MvpAgent::load_session → x.ai/modelBlock →
//! Tauri → React 这条真实链路，也没有验证"原会话经过真实切换消息后原地写回
//! 身份"——下面场景 4 用的是新临时目录 + init_session_with_catalog，验证的是
//! 写回后的**恢复语义**，不是写回动作本身。真正的端到端仍缺 GUI 那一段。
//!
//! 这条链此前每一段都有测试，但从没有一次是端到端的：旧格式 summary.json
//! 由**生产写入器**落盘、由生产解析器读回、歧义由生产逻辑判定、用户选定后
//! 由生产 sampling 链发出请求、身份由生产写入器回写。中间任何一环用假数据
//! 替代，都可能掩盖真实的形状错配——`endpoint_label` 那次事故正是这样：
//! 每段都对，接起来是错的。
//!
//! 覆盖（引擎层）：
//!   1. 恢复一个只有重复 slug 的旧会话 → 报歧义，候选带正确端点标签
//!   2. 歧义期间不产生任何上游请求（两个 mock 都是 0）
//!   3. 用户选定精确 key → 请求落到该 key 自己的端点，另一个仍是 0
//!   4. 身份回写后重新加载 → 精确恢复，不再歧义
//!   5. 模型已从配置中移除 → NotFound（不可用类阻塞的数据面）
//!
//! 未覆盖（如实记录）：GUI 点击层。按钮禁用、"稍后再说"不解除、切换会话不
//! 残留这些行为目前只有类型检查与解析单测，没有渲染层断言。

use std::net::SocketAddr;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;
use std::time::Duration;

use agent_client_protocol as acp;
use axum::response::sse::{Event, Sse};
use axum::routing::post;
use axum::extract::State;
use axum::Router;
use futures_util::stream;
use indexmap::IndexMap;
use tokio::sync::{mpsc, oneshot};

use xai_grok_sampler::{RequestId, RetryPolicy, SamplerActor, SamplingEvent};
use xai_grok_sampling_types::{ContentPart, ConversationItem, ConversationRequest, UserItem};
use xai_grok_shell::agent::config::{
    resolve_credentials, sampling_config_for_model, ModelEntry, ModelInfo,
};
use xai_grok_shell::agent::models::{
    resolve_persisted_model, resolve_requested_model, PersistedModelResolution,
};
use xai_grok_shell::session::info::Info;
use xai_grok_shell::session::storage::jsonl::JsonlStorageAdapter;
use xai_grok_shell::session::storage::StorageAdapter;

// ── mock 端点 ────────────────────────────────────────────────────────────
#[derive(Clone, Default)]
struct Hits(Arc<AtomicU32>);

struct Mock {
    addr: SocketAddr,
    hits: Hits,
    shutdown: oneshot::Sender<()>,
}

async fn handler(
    State(h): State<Hits>,
) -> Sse<impl futures_util::Stream<Item = Result<Event, std::convert::Infallible>>> {
    h.0.fetch_add(1, Ordering::SeqCst);
    let chunk = serde_json::json!({
        "id": "e2e", "object": "chat.completion.chunk", "created": 0, "model": "glm-4.6",
        "choices": [{ "index": 0, "delta": { "role": "assistant", "content": "ok" },
                      "finish_reason": "stop" }]
    });
    Sse::new(stream::iter(vec![
        Ok(Event::default().data(chunk.to_string())),
        Ok(Event::default().data("[DONE]")),
    ]))
}

impl Mock {
    async fn spawn() -> Self {
        let hits = Hits::default();
        let app = Router::new()
            .route("/v1/chat/completions", post(handler))
            .with_state(hits.clone());
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let (shutdown, rx) = oneshot::channel::<()>();
        tokio::spawn(async move {
            let _ = axum::serve(listener, app)
                .with_graceful_shutdown(async move {
                    let _ = rx.await;
                })
                .await;
        });
        tokio::time::sleep(Duration::from_millis(20)).await;
        Self { addr, hits, shutdown }
    }
    fn base_url(&self) -> String {
        format!("http://{}/v1", self.addr)
    }
    fn count(&self) -> u32 {
        self.hits.0.load(Ordering::SeqCst)
    }
    fn stop(self) {
        let _ = self.shutdown.send(());
    }
}

// ── 目录与会话 ───────────────────────────────────────────────────────────
const SHARED_SLUG: &str = "glm-4.6";

fn entry(base_url: &str, key: &str) -> ModelEntry {
    let mut info = ModelInfo::fallback(SHARED_SLUG);
    info.base_url = base_url.to_owned();
    info.name = Some(format!("GLM via {key}"));
    info.max_completion_tokens = Some(64);
    ModelEntry {
        info,
        api_key: Some(format!("key-for-{key}")),
        env_key: None,
        api_base_url: None,
    }
}

fn catalog(
    open: &str,
    coding: &str,
) -> (
    IndexMap<String, ModelEntry>,
    IndexMap<acp::ModelId, acp::ModelInfo>,
) {
    let mut models = IndexMap::new();
    models.insert("glm-open".to_owned(), entry(open, "glm-open"));
    models.insert("glm-coding".to_owned(), entry(coding, "glm-coding"));
    let available = models
        .keys()
        .map(|k| {
            let id = acp::ModelId::new(k.clone());
            (id.clone(), acp::ModelInfo::new(id, k.clone()))
        })
        .collect();
    (models, available)
}

/// 用**生产写入器**落一份只有 slug 的旧格式会话记录。
async fn write_legacy_session(dir: &std::path::Path) -> (JsonlStorageAdapter, Info) {
    let adapter = JsonlStorageAdapter::with_explicit_session_dir(dir.to_path_buf());
    let info = Info {
        id: acp::SessionId::new("e2e-legacy"),
        cwd: dir.to_string_lossy().into_owned(),
    };
    // 两参入口 = 不写 catalog_model_id，正是 v0.18.6 之前的记录形状。
    adapter
        .init_session(&info, acp::ModelId::new(SHARED_SLUG))
        .await
        .unwrap();
    (adapter, info)
}

async fn send_one(entry: &ModelEntry) {
    let creds = resolve_credentials(entry, None);
    let mut cfg = sampling_config_for_model(entry, creds, None, None, None, None);
    cfg.max_retries = Some(0);
    cfg.idle_timeout_secs = Some(5);
    let (tx, mut rx) = mpsc::unbounded_channel();
    let handle = SamplerActor::spawn(cfg, RetryPolicy::default(), tx);
    handle.submit(
        RequestId::from("e2e"),
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
    let deadline = tokio::time::Instant::now() + Duration::from_secs(10);
    loop {
        let left = deadline.saturating_duration_since(tokio::time::Instant::now());
        assert!(!left.is_zero(), "sampler 未在超时内给出终态");
        match tokio::time::timeout(left, rx.recv()).await {
            Ok(Some(SamplingEvent::Completed { .. }))
            | Ok(Some(SamplingEvent::Failed { .. }))
            | Ok(None) => return,
            Ok(Some(_)) => continue,
            Err(_) => panic!("sampler 未在超时内给出终态"),
        }
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn legacy_session_is_held_then_resolved_to_the_right_endpoint() {
    let open = Mock::spawn().await;
    let coding = Mock::spawn().await;
    let (models, available) = catalog(&open.base_url(), &coding.base_url());

    let tmp = tempfile::tempdir().unwrap();
    let (adapter, info) = write_legacy_session(tmp.path()).await;

    // ── 场景 1：恢复旧记录 → 歧义，候选带正确端点标签 ────────────────
    let summary = adapter
        .init_session(&info, acp::ModelId::new(SHARED_SLUG))
        .await
        .unwrap();
    assert_eq!(summary.catalog_model_id, None, "前提：这是旧格式记录");

    let resolution = resolve_persisted_model(
        &models,
        &available,
        summary.catalog_model_id.as_ref().map(|m| m.0.as_ref()),
        summary.current_model_id.0.as_ref(),
    );
    let candidates = match resolution {
        PersistedModelResolution::Ambiguous { candidates, .. } => candidates,
        other => panic!("旧记录必须报歧义，实际 {other:?}"),
    };
    assert_eq!(candidates.len(), 2);
    let open_host = open.addr.ip().to_string();
    assert!(
        candidates.iter().all(|c| c.endpoint_label.contains(&open_host)),
        "候选端点标签必须是真实 host，空白或错值等于选择器没用：{candidates:?}"
    );
    assert!(candidates.iter().any(|c| c.id == "glm-open"));
    assert!(candidates.iter().any(|c| c.id == "glm-coding"));

    // ── 场景 2：歧义期间一个上游请求都不该发出 ──────────────────────
    assert_eq!(open.count(), 0);
    assert_eq!(coding.count(), 0);

    // ── 场景 3：用户选定精确 key → 只打它自己的端点 ─────────────────
    let selection = resolve_requested_model(&models, &available, "glm-coding").unwrap();
    send_one(selection.entry()).await;
    assert_eq!(coding.count(), 1, "选定的端点应恰好收到一次请求");
    assert_eq!(open.count(), 0, "另一个端点必须一次都没有");

    // ── 场景 4：带身份的记录重新加载 → 精确恢复，不再歧义 ─────────────
    // 注意范围：这里用新目录 + init_session_with_catalog 造出"已写回身份"的
    // 记录，验证的是它的恢复语义。真实切换消息如何原地更新既有 summary，
    // 由 summary_write 那层的测试覆盖，不在本文件的证据范围内。
    let tmp2 = tempfile::tempdir().unwrap();
    let adapter2 = JsonlStorageAdapter::with_explicit_session_dir(tmp2.path().to_path_buf());
    let info2 = Info {
        id: acp::SessionId::new("e2e-resolved"),
        cwd: tmp2.path().to_string_lossy().into_owned(),
    };
    adapter2
        .init_session_with_catalog(
            &info2,
            acp::ModelId::new(SHARED_SLUG),
            Some(selection.catalog_key().clone()),
        )
        .await
        .unwrap();
    let reloaded = adapter2
        .init_session(&info2, acp::ModelId::new(SHARED_SLUG))
        .await
        .unwrap();
    assert_eq!(
        resolve_persisted_model(
            &models,
            &available,
            reloaded.catalog_model_id.as_ref().map(|m| m.0.as_ref()),
            reloaded.current_model_id.0.as_ref(),
        ),
        PersistedModelResolution::Exact(acp::ModelId::new("glm-coding")),
        "写回身份后必须精确恢复——用户不该被问第二次"
    );

    open.stop();
    coding.stop();
}

/// 场景 5：模型已从配置中移除 —— 不可用类阻塞的数据面。
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_session_whose_model_left_the_catalog_resolves_to_not_found() {
    let empty_models: IndexMap<String, ModelEntry> = IndexMap::new();
    let empty_available: IndexMap<acp::ModelId, acp::ModelInfo> = IndexMap::new();
    assert_eq!(
        resolve_persisted_model(
            &empty_models,
            &empty_available,
            Some("glm-coding"),
            SHARED_SLUG,
        ),
        PersistedModelResolution::NotFound
    );
}
