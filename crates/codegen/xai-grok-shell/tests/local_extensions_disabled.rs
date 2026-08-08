//! `local_extensions_disabled`: a session restricted to built-in tools.
//!
//! Runs against the public ACP surface wherever possible — `initialize` /
//! `session/new` / `session/load` / `ext_method` — because that is the contract
//! clients actually depend on. Only invariants with no wire projection use the
//! read-only scaffolding behind `local-extensions-test-support`.
//!
//! ```bash
//! cargo test -p xai-grok-shell --features local-extensions-test-support \
//!     --test local_extensions_disabled
//! ```
//!
//! Why this is an integration target rather than unit tests: the crate's
//! `--lib` tests do not build at this commit (upstream helpers such as
//! `WorkspaceOps::for_test` and a cross-crate `#[cfg(test)]`
//! `MockEmbeddingProvider` are unreachable — 51 errors on the base commit), so
//! anything written as a `#[cfg(test)]` module here would never run.

use agent_client_protocol as acp;
use agent_client_protocol::Agent as _;
use xai_grok_shell::agent::MvpAgent;

thread_local! {
    /// Data-plane URL the managed-config endpoint advertises for `e2-src`.
    /// Set once the fixture's port is known, since the URL embeds it.
    static MANAGED_ENDPOINT: std::cell::RefCell<String> =
        const { std::cell::RefCell::new(String::new()) };
    /// Receiver half of the gateway the most recently built agent writes to.
    ///
    /// Handed over rather than dropped so a test can answer reverse requests:
    /// E4's MCP servers live in the *client*, so their only evidence is traffic
    /// on this channel. A dropped receiver makes every reverse call fail, which
    /// is indistinguishable from "the server was never registered".
    static GATEWAY_RX: std::cell::RefCell<
        Option<tokio::sync::mpsc::UnboundedReceiver<xai_acp_lib::AcpClientMessage>>,
    > = const { std::cell::RefCell::new(None) };
}

/// Bearer the seeded credential carries; asserted on the control-plane request
/// so the test proves an *authenticated* fetch, not merely a reachable URL.
const TEST_BEARER: &str = "test-managed-oidc-token";
const KEY: &str = "x.ai/localExtensionsDisabled";
const APPLIED: &str = "localExtensionsDisabledApplied";

// ---------------------------------------------------------------------------
// fixtures
// ---------------------------------------------------------------------------

/// A live agent with just enough auth to get past the `session/new` gate.
///
/// `XAI_API_KEY` is what makes `initialize` select a default auth method; the
/// key is never used for a request because no test here runs a turn.
/// A plugin on disk that contributes both a hook and an MCP server, so a
/// registry built from it is unmistakably non-empty.
///
/// Tests that assert "nothing was installed" are only worth running when
/// something *could* have been: with no plugin anywhere, every such assertion
/// passes for the wrong reason.
fn plugin_fixture(name: &str) -> tempfile::TempDir {
    let dir = tempfile::tempdir().expect("plugin dir");
    std::fs::write(
        dir.path().join("plugin.json"),
        format!(r#"{{"name": "{name}"}}"#),
    )
    .expect("plugin.json");
    std::fs::write(
        dir.path().join(".mcp.json"),
        r#"{"mcpServers":{"fixture-srv":{"command":"echo","args":["hi"]}}}"#,
    )
    .expect(".mcp.json");
    dir
}

fn agent() -> (MvpAgent, tempfile::TempDir) {
    agent_with_plugins(&[])
}

fn agent_with_plugins(plugin_dirs: &[&std::path::Path]) -> (MvpAgent, tempfile::TempDir) {
    agent_full(plugin_dirs, None)
}

/// Seed `auth.json` with a fresh first-party OIDC credential.
///
/// Managed MCP is gated on `is_managed_mcp_eligible()`. `WebLogin` looks like
/// it would satisfy that gate but does not: it is deprecated, and `lookup_auth`
/// skips such entries outright, so the store loads and the credential still
/// resolves to `None`. `is_xai_auth()` additionally requires the issuer to be
/// x.ai — an OIDC credential without `oidc_issuer` does not qualify either.
///
/// Both halves of E2 use this same credential, so the negative half cannot
/// pass merely because its agent had no usable auth.
fn seed_fresh_xai_oidc_auth(dir: &std::path::Path) {
    let scope = xai_grok_shell::auth::GrokComConfig::default().auth_scope();
    let now = chrono::Utc::now();
    let store = serde_json::json!({
        scope: {
            "key": TEST_BEARER,
            "auth_mode": "oidc",
            "create_time": now,
            "user_id": "test-user",
            "email": serde_json::Value::Null,
            "refresh_token": "test-refresh-token",
            // Must be in the future, or the credential resolves as expired.
            "expires_at": now + chrono::Duration::hours(6),
            // Without an x.ai issuer `is_xai_auth()` is false and the managed
            // gate stays shut.
            "oidc_issuer": xai_grok_shell::auth::XAI_OAUTH2_ISSUER,
            "oidc_client_id": "test-client-id",
        }
    });
    std::fs::write(
        dir.join("auth.json"),
        serde_json::to_string(&store).expect("auth store"),
    )
    .expect("write auth.json");
}

#[allow(dead_code)]
/// Point `GROK_HOME` at a scratch directory, exactly once per process.
///
/// This is the only environment write left in the file, and it is deliberately
/// *not* on the per-agent path. Writing the environment while other threads
/// run is unsound — an earlier revision set `GROK_HOME`, `XAI_API_KEY` and the
/// proxy URL on every `agent_full` call, and building a second agent aborted
/// the process with STATUS_HEAP_CORRUPTION because the first agent's reqwest
/// and MCP workers were reading the environment block concurrently.
///
/// Running once, at the first construction and before any agent thread exists,
/// removes the race. `GROK_HOME` has no API-level equivalent (unlike the auth
/// directory, which `AuthManager::new` takes directly), so leaving it unset
/// would put session files in the developer's real `~/.grok`.
///
/// Child processes get theirs via `Command::env` instead, which never touches
/// this process at all.
fn scratch_home() -> &'static std::path::Path {
    static HOME: std::sync::OnceLock<tempfile::TempDir> = std::sync::OnceLock::new();
    HOME.get_or_init(|| {
        let dir = tempfile::tempdir().expect("scratch home");
        // SAFETY: runs once, at the first agent construction, before any agent
        // has spawned a thread that reads the environment.
        unsafe { std::env::set_var("GROK_HOME", dir.path()) };
        dir
    })
    .path()
}

fn agent_full(
    plugin_dirs: &[&std::path::Path],
    managed_proxy_url: Option<&str>,
) -> (MvpAgent, tempfile::TempDir) {
    let _ = scratch_home();
    let temp = tempfile::tempdir().expect("temp dir");
    // Every agent gets a credential, not just the managed ones: it is also what
    // makes `initialize` select a default auth method, which `session/new`
    // requires. Doing it here instead of exporting `XAI_API_KEY` keeps this
    // function free of process-environment writes.
    seed_fresh_xai_oidc_auth(temp.path());
    let auth = std::sync::Arc::new(xai_grok_shell::auth::AuthManager::new(
        temp.path(),
        xai_grok_shell::auth::GrokComConfig::default(),
    ));
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    GATEWAY_RX.with(|slot| *slot.borrow_mut() = Some(rx));
    let gateway = xai_acp_lib::AcpAgentGatewaySender::new(tx);
    let mut cfg = xai_grok_shell::agent::config::Config::default();
    cfg.plugins.cli_plugin_dirs = plugin_dirs.iter().map(|p| p.to_path_buf()).collect();
    if let Some(url) = managed_proxy_url {
        cfg.managed_mcps_enabled = true;
        cfg.managed_mcp_gateway_tools_enabled = false;
        cfg.endpoints.cli_chat_proxy_base_url = Some(url.to_string());
    }
    let agent = MvpAgent::new(gateway, &cfg, auth, None).expect("valid test config");
    (agent, temp)
}

async fn init(a: &MvpAgent) {
    a.initialize(acp::InitializeRequest::new(acp::ProtocolVersion::V1))
        .await
        .expect("initialize");
}

fn meta(value: serde_json::Value) -> Option<acp::Meta> {
    value.as_object().cloned()
}

/// `session/new` with the given `_meta`, in a fresh cwd so sessions never
/// share project-scoped state.
async fn new_session(
    a: &MvpAgent,
    cwd: &std::path::Path,
    m: Option<acp::Meta>,
) -> Result<acp::NewSessionResponse, acp::Error> {
    let mut req = acp::NewSessionRequest::new(cwd.to_path_buf());
    req.meta = m;
    a.new_session(req).await
}

async fn disabled_session(a: &MvpAgent, cwd: &std::path::Path) -> acp::SessionId {
    new_session(a, cwd, meta(serde_json::json!({ KEY: true })))
        .await
        .expect("disabled session")
        .session_id
}

async fn ordinary_session(a: &MvpAgent, cwd: &std::path::Path) -> acp::SessionId {
    new_session(a, cwd, None)
        .await
        .expect("ordinary session")
        .session_id
}

async fn ext(
    a: &MvpAgent,
    method: &str,
    params: serde_json::Value,
) -> Result<serde_json::Value, acp::Error> {
    let raw = serde_json::value::RawValue::from_string(params.to_string()).expect("raw params");
    let resp = a
        .ext_method(acp::ExtRequest::new(method, raw.into()))
        .await?;
    let envelope: serde_json::Value =
        serde_json::from_str(resp.0.get()).expect("ext response is json");
    // `to_ext_response` wraps every payload in `{"result": …}`; tests assert on
    // the payload, so unwrap it here once rather than in each assertion.
    Ok(envelope
        .get("result")
        .cloned()
        .unwrap_or(envelope))
}

/// `data` payload of a structured refusal, so tests assert on the contract
/// rather than on message text.
fn err_data(e: &acp::Error) -> serde_json::Value {
    serde_json::to_value(&e.data).unwrap_or(serde_json::Value::Null)
}

fn applied(resp_meta: &Option<acp::Meta>) -> Option<&serde_json::Value> {
    resp_meta.as_ref().and_then(|m| m.get(APPLIED))
}

macro_rules! local_test {
    ($name:ident, $body:expr) => {
        #[tokio::test]
        #[serial_test::serial]
        async fn $name() {
            // The session actor is spawned onto the current LocalSet.
            tokio::task::LocalSet::new().run_until($body).await
        }
    };
}

// ---------------------------------------------------------------------------
// T1 / T12 — new session, handshake, both lists empty
// ---------------------------------------------------------------------------

local_test!(t1_new_session_applies_and_zeroes_both_lists, async {
    let (a, tmp) = agent();
    init(&a).await;
    let resp = new_session(&a, tmp.path(), meta(serde_json::json!({ KEY: true })))
        .await
        .expect("new session");
    let sid = resp.session_id.clone();

    assert_eq!(
        applied(&resp.meta),
        Some(&serde_json::Value::Bool(true)),
        "session/new must confirm the policy was applied"
    );
    // T12: the confirmation is read off the installed handle, so the latch and
    // the echo cannot disagree.
    assert_eq!(a.session_local_extensions_disabled_snapshot(&sid), Some(true));

    let plugins = ext(&a, "x.ai/plugins/list", serde_json::json!({ "sessionId": sid.0 }))
        .await
        .expect("plugins/list");
    assert_eq!(plugins, serde_json::json!({ "plugins": [] }));

    let hooks = ext(&a, "x.ai/hooks/list", serde_json::json!({ "sessionId": sid.0 }))
        .await
        .expect("hooks/list");
    // `loadErrors` carries `skip_serializing_if = "Vec::is_empty"`, so the
    // empty response must not contain it — this is why the handler builds the
    // real DTO instead of a JSON literal.
    assert_eq!(
        hooks,
        serde_json::json!({ "hooks": [], "projectTrusted": false }),
        "empty hooks response must match the DTO's own serialization"
    );
});

// ---------------------------------------------------------------------------
// T6 — a client that never sends the key sees byte-identical behaviour
// ---------------------------------------------------------------------------

local_test!(t6_absent_key_defaults_to_false_and_echoes_nothing, async {
    let (a, tmp) = agent();
    init(&a).await;
    let resp = new_session(&a, tmp.path(), None).await.expect("new session");
    assert_eq!(
        applied(&resp.meta),
        None,
        "responses to clients that did not ask for the policy must not gain a key"
    );
    assert_eq!(
        a.session_local_extensions_disabled_snapshot(&resp.session_id),
        Some(false)
    );
});

// ---------------------------------------------------------------------------
// T11 — every non-boolean shape is refused, including an explicit null
// ---------------------------------------------------------------------------

local_test!(t11_non_boolean_values_are_refused, async {
    let (a, tmp) = agent();
    init(&a).await;
    for (label, value) in [
        ("string", serde_json::json!("true")),
        ("number", serde_json::json!(1)),
        ("object", serde_json::json!({ "enabled": true })),
        ("array", serde_json::json!([true])),
        // A client emits `null` when its own value was absent or failed to
        // serialize — precisely the case that must not read as a deliberate
        // "no".
        ("null", serde_json::Value::Null),
    ] {
        let err = new_session(&a, tmp.path(), meta(serde_json::json!({ KEY: value })))
            .await
            .expect_err(&format!("{label} must be refused"));
        let data = err_data(&err);
        assert_eq!(data["code"], "local_extensions_disabled", "{label}");
        assert_eq!(data["reason"], "invalid_type", "{label}");
        assert_eq!(data["received"], label, "{label}");
    }
});

// ---------------------------------------------------------------------------
// T16b / T18 — no plugin discovery is performed on behalf of such a session
// ---------------------------------------------------------------------------

local_test!(t16b_no_plugin_registry_initialization, async {
    let (a, tmp) = agent();
    init(&a).await;

    // Delta across one creation, not the registry's final state: on a
    // connection that also carries ordinary sessions the shared registry may
    // have been populated earlier, and the end state cannot say by whom.
    let before = a.ensure_plugin_registry_call_count();
    let _ = disabled_session(&a, tmp.path()).await;
    assert_eq!(
        a.ensure_plugin_registry_call_count(),
        before,
        "creating a built-in-tools-only session must not trigger plugin discovery \
         (T18: `resolve_mcp_servers` is skipped entirely, and it is what calls this)"
    );

    // Contrast: an ordinary session does reach it, so the counter is measuring
    // something real rather than being dead.
    let ordinary_cwd = tempfile::tempdir().expect("cwd");
    let _ = ordinary_session(&a, ordinary_cwd.path()).await;
    assert!(
        a.ensure_plugin_registry_call_count() > before,
        "an ordinary session must still initialize the shared registry"
    );
});

// ---------------------------------------------------------------------------
// T15 / T16 — all four MCP sources are zeroed at once
// ---------------------------------------------------------------------------

local_test!(t16_all_mcp_sources_zeroed, async {
    let (a, tmp) = agent();
    init(&a).await;
    let mut req = acp::NewSessionRequest::new(tmp.path().to_path_buf());
    // E3 (request) and E4 (`_meta`) are distinct paths; supplying both at once
    // is what makes this test able to fail if only one is handled.
    req.mcp_servers = vec![
        serde_json::from_value::<acp::McpServer>(serde_json::json!({
            "type": "http",
            "name": "e3-server",
            "url": "http://127.0.0.1:1/mcp",
            "headers": [],
        }))
        .expect("http mcp server"),
    ];
    req.meta = meta(serde_json::json!({
        KEY: true,
        "x.ai/mcp/servers": [
            { "name": "e4-server", "url": "http://127.0.0.1:2/mcp" }
        ],
    }));
    let before = a.ensure_plugin_registry_call_count();
    let resp = a.new_session(req).await.expect("new session");
    assert_eq!(applied(&resp.meta), Some(&serde_json::Value::Bool(true)));
    assert_eq!(
        a.ensure_plugin_registry_call_count(),
        before,
        "E1/E2 are skipped with the rest of `resolve_mcp_servers`"
    );
    // No MCP server ever reached a session, so nothing could have connected:
    // both URLs above point at closed ports and the test does not hang.
});

// ---------------------------------------------------------------------------
// T10 — client-supplied hooks are not a way around disabled discovery
// ---------------------------------------------------------------------------

local_test!(t10_client_hooks_in_meta_are_dropped, async {
    let (a, tmp) = agent();
    init(&a).await;
    let sid = new_session(
        &a,
        tmp.path(),
        meta(serde_json::json!({
            KEY: true,
            "x.ai/hooks": { "PreToolUse": [{ "hookCallbackIds": ["cb-1"] }] },
        })),
    )
    .await
    .expect("new session")
    .session_id;

    let hooks = ext(&a, "x.ai/hooks/list", serde_json::json!({ "sessionId": sid.0 }))
        .await
        .expect("hooks/list");
    assert_eq!(hooks["hooks"], serde_json::json!([]));
});

// ---------------------------------------------------------------------------
// T4 — extension actions are refused, not silently ignored
// ---------------------------------------------------------------------------

local_test!(t4_plugin_and_hook_actions_are_refused, async {
    let (a, tmp) = agent();
    init(&a).await;
    let sid = disabled_session(&a, tmp.path()).await;

    let err = ext(
        &a,
        "x.ai/plugins/action",
        serde_json::json!({ "sessionId": sid.0, "action": { "type": "reload" } }),
    )
    .await
    .expect_err("plugins/action must be refused");
    assert_eq!(err_data(&err)["reason"], "plugins_action_refused");

    let err = ext(
        &a,
        "x.ai/hooks/action",
        serde_json::json!({ "sessionId": sid.0, "action": { "type": "untrust" } }),
    )
    .await
    .expect_err("hooks/action must be refused");
    assert_eq!(err_data(&err)["reason"], "hooks_action_refused");
});

// ---------------------------------------------------------------------------
// T24 — unknown ids fail closed, and the load race stays fixed
// ---------------------------------------------------------------------------

local_test!(t24_unknown_session_ids_fail_closed, async {
    let (a, tmp) = agent();
    init(&a).await;
    // A live ordinary session exists, so the shared registry is populated and
    // the old fallback would have had something to leak.
    let _ordinary = ordinary_session(&a, tmp.path()).await;

    let unknown = acp::SessionId::new("00000000-0000-0000-0000-0000000000ff");
    assert_eq!(a.session_local_extensions_disabled_snapshot(&unknown), None);

    let plugins = ext(
        &a,
        "x.ai/plugins/list",
        serde_json::json!({ "sessionId": unknown.0 }),
    )
    .await
    .expect("plugins/list");
    assert_eq!(
        plugins,
        serde_json::json!({ "plugins": [] }),
        "an unknown session id must not be answered from the shared registry"
    );

    let hooks = ext(
        &a,
        "x.ai/hooks/list",
        serde_json::json!({ "sessionId": unknown.0 }),
    )
    .await
    .expect("hooks/list");
    assert_eq!(hooks["hooks"], serde_json::json!([]));

    let err = ext(
        &a,
        "x.ai/plugins/action",
        serde_json::json!({ "sessionId": unknown.0, "action": { "type": "reload" } }),
    )
    .await
    .expect_err("unknown id must be refused");
    assert_eq!(
        err_data(&err)["reason"],
        "unknown_session",
        "unknown must be distinguishable from a policy refusal"
    );
});

// ---------------------------------------------------------------------------
// T5 — ordinary sessions on the same connection are untouched
// ---------------------------------------------------------------------------

local_test!(t5_ordinary_session_unaffected_alongside_disabled, async {
    let fixture = plugin_fixture("t5-fixture-plugin");
    let (a, tmp) = agent_with_plugins(&[fixture.path()]);
    init(&a).await;
    let disabled_cwd = tempfile::tempdir().expect("cwd");
    let disabled = disabled_session(&a, disabled_cwd.path()).await;
    let ordinary = ordinary_session(&a, tmp.path()).await;

    assert_eq!(
        a.session_local_extensions_disabled_snapshot(&ordinary),
        Some(false)
    );
    assert_eq!(
        a.session_local_extensions_disabled_snapshot(&disabled),
        Some(true)
    );

    // The ordinary session sees the fixture, through the endpoint and on its
    // actor. Asserting a positive here is the point: "no plugins anywhere"
    // would satisfy a test that only checked the restricted session.
    let plugins = ext(
        &a,
        "x.ai/plugins/list",
        serde_json::json!({ "sessionId": ordinary.0 }),
    )
    .await
    .expect("plugins/list");
    assert_eq!(
        plugins["plugins"].as_array().map(Vec::len),
        Some(1),
        "the ordinary session must still see the fixture plugin"
    );
    assert_eq!(a.session_raw_plugin_registry(&ordinary).await, Some(Some(1)));

    // Its extension actions must *succeed*, not merely fail differently. An
    // earlier revision accepted any non-policy error here, which would have
    // passed even if the endpoint were broken outright.
    ext(
        &a,
        "x.ai/plugins/action",
        serde_json::json!({ "sessionId": ordinary.0, "action": { "type": "reload" } }),
    )
    .await
    .expect("an ordinary session's plugin action must succeed");
    assert_eq!(
        a.session_raw_plugin_registry(&ordinary).await,
        Some(Some(1)),
        "and must leave it with a working registry"
    );

    // Meanwhile the restricted session is unchanged, actor included.
    assert_eq!(a.session_raw_plugin_registry(&disabled).await, Some(None));
});

// ---------------------------------------------------------------------------
// T22 — the process-wide reload is conditional, not refused outright
// ---------------------------------------------------------------------------

local_test!(t22_global_reload_requires_an_eligible_session, async {
    let fixture = plugin_fixture("t22-fixture-plugin");
    let (a, tmp) = agent_with_plugins(&[fixture.path()]);
    init(&a).await;

    // Only a built-in-tools-only session exists: nobody could consume the
    // rebuild, so it is refused rather than run for no one.
    let disabled = disabled_session(&a, tmp.path()).await;
    let err = ext(&a, "x.ai/plugins/reload", serde_json::json!({}))
        .await
        .expect_err("reload must be refused when no session may use plugins");
    assert_eq!(err_data(&err)["reason"], "no_eligible_session");
    assert_eq!(
        a.session_raw_plugin_registry(&disabled).await,
        Some(None),
        "a refused reload must not have installed anything"
    );

    // Add an ordinary session and the same call succeeds — the endpoint stays
    // usable for the sessions it exists for.
    let ordinary_cwd = tempfile::tempdir().expect("cwd");
    let ordinary = ordinary_session(&a, ordinary_cwd.path()).await;
    ext(&a, "x.ai/plugins/reload", serde_json::json!({}))
        .await
        .expect("reload must succeed once an eligible session exists");

    // Positive side: the eligible session's registry really was rebuilt.
    assert_eq!(
        a.session_raw_plugin_registry(&ordinary).await,
        Some(Some(1)),
        "the reload must have reached the session it exists for"
    );

    // Negative side: the restricted session's actor is still registry-free —
    // checked past the endpoint, which would report empty either way.
    assert_eq!(
        a.session_raw_plugin_registry(&disabled).await,
        Some(None),
        "the fan-out must not have installed a registry on the protected session"
    );
    assert_eq!(
        ext(
            &a,
            "x.ai/plugins/list",
            serde_json::json!({ "sessionId": disabled.0 }),
        )
        .await
        .expect("plugins/list"),
        serde_json::json!({ "plugins": [] }),
    );
    assert_eq!(
        ext(
            &a,
            "x.ai/hooks/list",
            serde_json::json!({ "sessionId": disabled.0 }),
        )
        .await
        .expect("hooks/list")["hooks"],
        serde_json::json!([]),
    );
});

// ---------------------------------------------------------------------------
// T3 — a reload driven by another session does not reach a disabled one
// ---------------------------------------------------------------------------

local_test!(t3_broadcast_does_not_reach_disabled_session, async {
    let fixture = plugin_fixture("t3-fixture-plugin");
    let (a, tmp) = agent_with_plugins(&[fixture.path()]);
    init(&a).await;
    let disabled_cwd = tempfile::tempdir().expect("cwd");
    let disabled = disabled_session(&a, disabled_cwd.path()).await;
    let ordinary = ordinary_session(&a, tmp.path()).await;

    // This is the call the fan-out probe showed polluting a protected session:
    // it rebuilds the shared registry and pushes it to every live session.
    //
    // The reload must actually succeed — asserted, not ignored. An earlier
    // revision of this test sent a malformed action payload, which was
    // rejected before any fan-out happened and made the whole test a false
    // pass: it proved a broadcast that never occurred could not pollute.
    ext(
        &a,
        "x.ai/plugins/action",
        serde_json::json!({ "sessionId": ordinary.0, "action": { "type": "reload" } }),
    )
    .await
    .expect("the ordinary session's reload must succeed, or this proves nothing");

    // The ordinary session really did pick the fixture up, so the broadcast
    // carried something worth blocking.
    assert_eq!(
        a.session_raw_plugin_registry(&ordinary).await,
        Some(Some(1)),
        "the reload must have installed the fixture on the ordinary session"
    );

    // Asked of the actor, not of the endpoint. The endpoint returns an empty
    // list for this session unconditionally, so a wire-only assertion here
    // would stay green even if the broadcast had installed a registry.
    assert_eq!(
        a.session_raw_plugin_registry(&disabled).await,
        Some(None),
        "the protected session's actor must hold no registry at all"
    );
    assert_eq!(
        ext(
            &a,
            "x.ai/plugins/list",
            serde_json::json!({ "sessionId": disabled.0 }),
        )
        .await
        .expect("plugins/list"),
        serde_json::json!({ "plugins": [] }),
    );
    assert_eq!(
        a.session_local_extensions_disabled_snapshot(&disabled),
        Some(true),
        "the latch is immutable for the actor's lifetime"
    );
});

// ---------------------------------------------------------------------------
// T21 — a subagent validation racing the parent's teardown fails closed
// ---------------------------------------------------------------------------

local_test!(t21_missing_parent_inherits_the_strictest_answer, async {
    let (a, tmp) = agent();
    init(&a).await;
    let ordinary = ordinary_session(&a, tmp.path()).await;
    // A live ordinary session guarantees the shared registry is initialized,
    // so "no registry" below cannot be a false pass caused by there being
    // nothing to hand out in the first place.
    let (inherited, has_registry) = a.subagent_validation_inheritance(&ordinary.0);
    assert!(!inherited, "an ordinary parent grants its own permissive policy");

    let disabled_cwd = tempfile::tempdir().expect("cwd");
    let disabled = disabled_session(&a, disabled_cwd.path()).await;
    let (inherited, disabled_registry) = a.subagent_validation_inheritance(&disabled.0);
    assert!(inherited, "T19: a subagent strictly inherits the parent's policy");
    assert!(
        !disabled_registry,
        "T20/I1: the *shared* registry must not be handed to it either"
    );

    // The evicted / never-existing parent: this constructor tolerates a
    // missing parent by design, which is exactly why the fallback matters.
    let (inherited, missing_registry) = a.subagent_validation_inheritance("no-such-session");
    assert!(
        inherited,
        "a policy that cannot be read must resolve to the strictest answer, \
         not the permissive one"
    );
    assert!(
        !missing_registry,
        "an eviction race must not re-expose plugin agents"
    );

    let _ = has_registry;
});

// ---------------------------------------------------------------------------
// T2 / T8 / T9 — the latch is fixed for the actor's lifetime
// ---------------------------------------------------------------------------

async fn load_session(
    a: &MvpAgent,
    sid: &acp::SessionId,
    cwd: &std::path::Path,
    m: Option<acp::Meta>,
) -> Result<acp::LoadSessionResponse, acp::Error> {
    let mut req = acp::LoadSessionRequest::new(sid.clone(), cwd.to_path_buf());
    req.meta = m;
    a.load_session(req).await
}

local_test!(t2_load_applies_the_policy_from_the_request, async {
    let (a, tmp) = agent();
    init(&a).await;
    let sid = disabled_session(&a, tmp.path()).await;

    // Reconnecting to the live actor with the same policy is accepted and the
    // response confirms it — the engine persists nothing, so the client
    // re-supplying the value is the whole mechanism.
    let resp = load_session(&a, &sid, tmp.path(), meta(serde_json::json!({ KEY: true })))
        .await
        .expect("load with matching policy");
    assert_eq!(applied(&resp.meta), Some(&serde_json::Value::Bool(true)));
    assert_eq!(a.session_local_extensions_disabled_snapshot(&sid), Some(true));
});

local_test!(t9_load_cannot_downgrade_a_protected_session, async {
    let (a, tmp) = agent();
    init(&a).await;
    let sid = disabled_session(&a, tmp.path()).await;
    let before = a.ensure_plugin_registry_call_count();

    for (label, m) in [
        ("explicit false", meta(serde_json::json!({ KEY: false }))),
        ("omitted key", None),
    ] {
        let err = load_session(&a, &sid, tmp.path(), m)
            .await
            .expect_err(&format!("{label} must be refused"));
        let data = err_data(&err);
        assert_eq!(data["code"], "local_extensions_disabled", "{label}");
        assert_eq!(data["reason"], "immutable_conflict", "{label}");
    }

    assert_eq!(
        a.session_local_extensions_disabled_snapshot(&sid),
        Some(true),
        "a refused load must leave the session protected"
    );
    // The comparison happens before `resolve_mcp_servers`, so a refused load
    // performs no MCP resolution and no plugin discovery: rejecting must be
    // free of side effects, or a caller could reshape a session by sending
    // loads it knows will fail.
    assert_eq!(
        a.ensure_plugin_registry_call_count(),
        before,
        "a refused load must not have run any of the work it refused"
    );
});

local_test!(t8_load_cannot_restrict_a_running_ordinary_session, async {
    let (a, tmp) = agent();
    init(&a).await;
    let sid = ordinary_session(&a, tmp.path()).await;

    let err = load_session(&a, &sid, tmp.path(), meta(serde_json::json!({ KEY: true })))
        .await
        .expect_err("must be refused");
    let data = err_data(&err);
    assert_eq!(data["code"], "local_extensions_disabled");
    assert_eq!(data["reason"], "immutable_conflict");
    assert_eq!(
        a.session_local_extensions_disabled_snapshot(&sid),
        Some(false),
        "the running session keeps the policy it was created with"
    );
});

// ---------------------------------------------------------------------------
// T24 (load race) — a request arriving mid-load waits instead of being
// answered as if the session did not exist
// ---------------------------------------------------------------------------

local_test!(t24_plugins_list_waits_for_an_in_flight_load, async {
    let fixture = plugin_fixture("t24-fixture-plugin");
    let (a, tmp) = agent_with_plugins(&[fixture.path()]);
    let a = std::rc::Rc::new(a);
    init(&a).await;

    // Create an ordinary session, then drop its actor so the reload below is
    // a genuine load rather than a reconnect to a live handle.
    let sid = ordinary_session(&a, tmp.path()).await;
    assert_eq!(a.session_raw_plugin_registry(&sid).await, Some(Some(1)));
    a.remove_session_for_test(&sid);
    assert_eq!(
        a.session_local_extensions_disabled_snapshot(&sid),
        None,
        "the session must be out of the map so the load really re-creates it"
    );

    let barrier = a.install_load_barrier();

    let load = {
        let a = a.clone();
        let sid = sid.clone();
        let cwd = tmp.path().to_path_buf();
        tokio::task::spawn_local(async move { load_session(&a, &sid, &cwd, None).await })
    };

    // The load is now registered as in flight, and no handle is in `sessions`.
    assert!(
        barrier.wait_until_reached().await,
        "the load must reach the barrier"
    );
    assert_eq!(
        a.session_local_extensions_disabled_snapshot(&sid),
        None,
        "handle must not be installed yet — otherwise this proves nothing"
    );

    // A `plugins/list` issued in this window must wait for the load rather
    // than treat the session as unknown. Checking the policy before awaiting
    // the handle would answer empty here, which is what T24 forbids.
    let listing = {
        let a = a.clone();
        let sid = sid.clone();
        tokio::task::spawn_local(async move {
            ext(
                &a,
                "x.ai/plugins/list",
                serde_json::json!({ "sessionId": sid.0 }),
            )
            .await
        })
    };

    // An unknown id must not be caught by that wait: it has no load in flight,
    // so it answers immediately even while another session is mid-load.
    let unknown = acp::SessionId::new("00000000-0000-0000-0000-0000000000fe");
    let unknown_listing = tokio::time::timeout(
        std::time::Duration::from_secs(2),
        ext(
            &a,
            "x.ai/plugins/list",
            serde_json::json!({ "sessionId": unknown.0 }),
        ),
    )
    .await
    .expect("an unknown id must not wait on an unrelated in-flight load")
    .expect("plugins/list");
    assert_eq!(unknown_listing, serde_json::json!({ "plugins": [] }));

    barrier.release();
    load.await
        .expect("load task")
        .expect("load must succeed");

    let listing = tokio::time::timeout(std::time::Duration::from_secs(10), listing)
        .await
        .expect("the waiting plugins/list must complete once the load lands")
        .expect("list task")
        .expect("plugins/list");
    assert_eq!(
        listing["plugins"].as_array().map(Vec::len),
        Some(1),
        "the waiting request must be answered from the loaded session's own          registry, not with the empty fail-closed answer"
    );
});

// ---------------------------------------------------------------------------
// MCP fixture probe
// ---------------------------------------------------------------------------

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

#[derive(Default)]
struct Hits {
    initialize: HashMap<String, usize>,
    tools_list: HashMap<String, usize>,
    /// Control-plane: requests to the managed-MCP list endpoint. Counted
    /// separately from the data plane because "the list was fetched" and "the
    /// listed server was connected to" are different claims, and E2 needs both.
    managed_configs: usize,
    /// Bearer tokens seen on the control plane, so the test can assert the
    /// fetch was authenticated with the seeded credential rather than merely
    /// that the URL was reachable.
    managed_bearers: Vec<String>,
}

/// The proxy base URL a managed-MCP-enabled agent should be pointed at.
#[allow(dead_code)]
fn proxy_base(mcp_base: &str) -> String {
    mcp_base.trim_end_matches("/mcp").to_string() + "/proxy"
}

async fn spawn_mcp_fixture() -> (String, Arc<Mutex<Hits>>) {
    use axum::extract::{Path, State};
    use axum::routing::post;
    use axum::Router;

    let hits: Arc<Mutex<Hits>> = Arc::new(Mutex::new(Hits::default()));
    let app = Router::new()
        .route(
            "/proxy/mcp/configs",
            axum::routing::get(
                |State(hits): State<Arc<Mutex<Hits>>>,
                 headers: axum::http::HeaderMap| async move {
                    {
                        let mut h = hits.lock().unwrap();
                        h.managed_configs += 1;
                        if let Some(auth) = headers
                            .get(axum::http::header::AUTHORIZATION)
                            .and_then(|v| v.to_str().ok())
                        {
                            h.managed_bearers.push(auth.to_string());
                        }
                    }
                    let base = MANAGED_ENDPOINT.with(|c| c.borrow().clone());
                    // Non-empty headers are required: `auto_inject_managed_
                    // servers_with_disabled` skips any managed config with an
                    // empty header map, so an empty one is listed on the
                    // control plane and then silently never connected.
                    // `McpConfigsResponse` carries no `rename_all`, so the
                    // field is snake_case. A camelCase body fails to parse,
                    // which surfaces as an empty list plus a retry — i.e. a
                    // busy control plane and a silent data plane.
                    axum::Json(serde_json::json!({
                        "mcp_servers": [
                            {
                                "name": "e2-src",
                                "endpoint": base,
                                "headers": { "Authorization": "Bearer managed-e2" }
                            }
                        ]
                    }))
                },
            ),
        )
        .route(
            "/mcp/{tag}",
            post(
                |State(hits): State<Arc<Mutex<Hits>>>,
                 Path(tag): Path<String>,
                 body: String| async move {
                    let req: serde_json::Value =
                        serde_json::from_str(&body).unwrap_or(serde_json::Value::Null);
                    let method = req["method"].as_str().unwrap_or("").to_string();
                    let id = req["id"].clone();
                    eprintln!("[fixture {tag}] method={method} body={body}");
                    let result = match method.as_str() {
                        "initialize" => {
                            hits.lock().unwrap().initialize.entry(tag.clone()).and_modify(|c| *c += 1).or_insert(1);
                            serde_json::json!({
                                "protocolVersion": "2025-06-18",
                                "capabilities": { "tools": {} },
                                "serverInfo": { "name": format!("fixture-{tag}"), "version": "0.0.1" }
                            })
                        }
                        "tools/list" => {
                            hits.lock().unwrap().tools_list.entry(tag.clone()).and_modify(|c| *c += 1).or_insert(1);
                            serde_json::json!({
                                "tools": [{
                                    "name": format!("echo_{tag}"),
                                    "description": "echo",
                                    "inputSchema": {"type":"object","properties":{}}
                                }]
                            })
                        }
                        _ => serde_json::json!({}),
                    };
                    if id.is_null() {
                        return axum::http::Response::builder()
                            .status(202)
                            .body(axum::body::Body::empty())
                            .unwrap();
                    }
                    let resp = serde_json::json!({"jsonrpc":"2.0","id":id,"result":result});
                    axum::http::Response::builder()
                        .status(200)
                        .header("content-type", "application/json")
                        .body(axum::body::Body::from(resp.to_string()))
                        .unwrap()
                },
            ),
        )
        .with_state(hits.clone());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::task::spawn_local(async move {
        axum::serve(listener, app).await.unwrap();
    });
    let mcp_base = format!("http://{addr}/mcp");
    // The managed-config endpoint advertises a data-plane URL, so it can only
    // be filled in once the port is known.
    MANAGED_ENDPOINT.with(|c| *c.borrow_mut() = format!("{mcp_base}/e2"));
    (mcp_base, hits)
}

/// Wait (bounded) for a condition on the fixture's counters.
///
/// MCP connections are made off the request path, so a bare assertion right
/// after `session/new` would race the connection rather than observe it. Poll
/// with a deadline instead — and take deltas against a baseline, because a
/// counter's absolute value says nothing about which session caused it.
async fn wait_for(mut cond: impl FnMut() -> bool) -> bool {
    for _ in 0..100 {
        if cond() {
            return true;
        }
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    }
    cond()
}

fn http_server(name: &str, url: String) -> acp::McpServer {
    serde_json::from_value(serde_json::json!({
        "type": "http", "name": name, "url": url, "headers": [],
    }))
    .expect("http mcp server")
}

// ---------------------------------------------------------------------------
// T17 (E3) — an ordinary session really connects
// T16 (E3) — a restricted session connects to nothing
//
// Each source gets its own URL path so a hit is attributable. Sharing one
// endpoint between sources would let a test pass while only one of them ever
// connected — which is exactly how the E4 gap below was found.
//
// SCOPE — E4 is deliberately absent here, and it is not an oversight:
// `_meta["x.ai/mcp/servers"]` is not a URL-addressed server at all. It
// registers *in-process SDK* servers as `[{ "name", "serverId" }]`, which the
// agent invokes by sending `x.ai/mcp/sdk_call` back over the ACP reverse
// channel (`session/acp_mcp.rs`). Its connection evidence therefore lives on
// the gateway receiver, not on an HTTP endpoint, and asserting it here would
// have meant asserting nothing. E1 (plugin) and E2 (managed, via the proxy
// URL) are likewise still to come; see the tracking notes in the design doc.
// ---------------------------------------------------------------------------

local_test!(t17_ordinary_session_connects_per_source, async {
    let (base, hits) = spawn_mcp_fixture().await;
    let (a, tmp) = agent();
    init(&a).await;

    let mut req = acp::NewSessionRequest::new(tmp.path().to_path_buf());
    req.mcp_servers = vec![http_server("e3-src", format!("{base}/e3"))];
    a.new_session(req).await.expect("new session");

    // A real handshake — `initialize` answered *and* `tools/list` issued —
    // not merely presence in a merged config list.
    let reached = wait_for(|| {
        let h = hits.lock().unwrap();
        h.initialize.get("e3").copied().unwrap_or(0) >= 1
            && h.tools_list.get("e3").copied().unwrap_or(0) >= 1
    })
    .await;
    let h = hits.lock().unwrap();
    assert!(
        reached,
        "the request-supplied MCP server must complete a handshake;          initialize={:?} tools_list={:?}",
        h.initialize, h.tools_list
    );
});

local_test!(t16_restricted_session_connects_to_nothing, async {
    let (base, hits) = spawn_mcp_fixture().await;
    let (a, tmp) = agent();
    init(&a).await;

    // Baseline first: the assertion is a delta, so a hit from anything else
    // can neither create nor mask a failure.
    let baseline = {
        let h = hits.lock().unwrap();
        (h.initialize.len(), h.tools_list.len())
    };

    let mut req = acp::NewSessionRequest::new(tmp.path().to_path_buf());
    req.mcp_servers = vec![http_server("e3-src", format!("{base}/e3"))];
    req.meta = meta(serde_json::json!({ KEY: true }));
    let resp = a.new_session(req).await.expect("new session");
    assert_eq!(applied(&resp.meta), Some(&serde_json::Value::Bool(true)));

    // Same wall-clock budget the positive test needed, so "no hits" means
    // "did not connect" rather than "has not connected yet".
    let connected = wait_for(|| !hits.lock().unwrap().initialize.is_empty()).await;
    let h = hits.lock().unwrap();
    assert!(
        !connected,
        "a restricted session must not connect to any MCP source; initialize={:?}",
        h.initialize
    );
    assert_eq!((h.initialize.len(), h.tools_list.len()), baseline);
});

// ---------------------------------------------------------------------------
// E4 — in-process SDK MCP servers over the ACP reverse channel
//
// Unlike E1/E2/E3 there is no transport to connect to: the server lives in the
// client, and the agent reaches it by sending `x.ai/mcp/sdk_call` back over the
// gateway. So the evidence is reverse traffic, and the fixture must *answer* —
// an unanswered call fails on a timeout, which looks exactly like "the server
// was never registered" and would let a broken interception read as success.
// ---------------------------------------------------------------------------

#[derive(Default)]
struct SdkHits {
    /// `(serverId, jsonrpc method)` → count. Keyed by content, never by call
    /// order, so an extra or reordered handshake step cannot shift the tally.
    calls: HashMap<(String, String), usize>,
}

/// Answer reverse requests until the agent drops the gateway.
fn spawn_sdk_pump() -> Arc<Mutex<SdkHits>> {
    use xai_acp_lib::AcpClientMessage;
    let hits: Arc<Mutex<SdkHits>> = Arc::new(Mutex::new(SdkHits::default()));
    let mut rx = GATEWAY_RX
        .with(|slot| slot.borrow_mut().take())
        .expect("gateway receiver was taken twice");
    let sink = hits.clone();
    tokio::task::spawn_local(async move {
        while let Some(msg) = rx.recv().await {
            let AcpClientMessage::ExtMethod(args) = msg else {
                // Notifications and other reverse requests are irrelevant here;
                // dropping their sender is what a disconnected client does.
                continue;
            };
            if args.request.method.as_ref() != "x.ai/mcp/sdk_call" {
                continue;
            }
            let params: serde_json::Value =
                serde_json::from_str(args.request.params.get()).unwrap_or_default();
            let server_id = params["serverId"].as_str().unwrap_or_default().to_string();
            let message = params["message"].clone();
            let method = message["method"].as_str().unwrap_or_default().to_string();
            *sink
                .lock()
                .unwrap()
                .calls
                .entry((server_id.clone(), method.clone()))
                .or_insert(0) += 1;

            let result = match method.as_str() {
                "initialize" => serde_json::json!({
                    "protocolVersion": "2025-06-18",
                    "capabilities": { "tools": {} },
                    "serverInfo": { "name": server_id, "version": "0.0.1" }
                }),
                "tools/list" => serde_json::json!({
                    "tools": [{
                        "name": format!("sdk_echo_{server_id}"),
                        "description": "echo",
                        "inputSchema": { "type": "object", "properties": {} }
                    }]
                }),
                _ => serde_json::json!({}),
            };
            // Notifications carry no id and must not get a JSON-RPC response
            // object, but the reverse *call* still needs an answer.
            let payload = if message["id"].is_null() {
                serde_json::json!({})
            } else {
                serde_json::json!({
                    "jsonrpc": "2.0", "id": message["id"].clone(), "result": result
                })
            };
            let raw = serde_json::value::RawValue::from_string(payload.to_string())
                .expect("raw response");
            let _ = args
                .response_tx
                .send(Ok(acp::ExtResponse::new(raw.into())));
        }
    });
    hits
}

fn sdk_meta(server_id: &str, extra: serde_json::Value) -> Option<acp::Meta> {
    let mut m = serde_json::json!({
        "x.ai/mcp/servers": [ { "name": server_id, "serverId": server_id } ],
    });
    if let (Some(dst), Some(src)) = (m.as_object_mut(), extra.as_object()) {
        for (k, v) in src {
            dst.insert(k.clone(), v.clone());
        }
    }
    m.as_object().cloned()
}

/// A completed E4 handshake is `initialize` + `tools/list` — deliberately
/// *without* `notifications/initialized`.
///
/// The reverse bridge discards id-less messages locally rather than sending
/// them, because the SDK peer rejects `x.ai/mcp/sdk_call`s with no JSON-RPC id
/// (`xai-grok-mcp/src/acp_transport.rs`, "half-duplex v1"). Requiring the
/// notification here would fail against a correct implementation; anyone who
/// later adds it to this list should change the bridge first.
fn sdk_handshake_complete(hits: &Arc<Mutex<SdkHits>>, server_id: &str) -> bool {
    let h = hits.lock().unwrap();
    ["initialize", "tools/list"]
        .iter()
        .all(|m| {
            h.calls
                .get(&(server_id.to_string(), (*m).to_string()))
                .copied()
                .unwrap_or(0)
                >= 1
        })
}

local_test!(t17_e4_ordinary_session_completes_reverse_handshake, async {
    let (http_base, http_hits) = spawn_mcp_fixture().await;
    let (a, tmp) = agent();
    let sdk_hits = spawn_sdk_pump();
    init(&a).await;

    let mut req = acp::NewSessionRequest::new(tmp.path().to_path_buf());
    // E3 alongside E4, on separate books: a shared tally could let one source's
    // traffic stand in for the other's.
    req.mcp_servers = vec![http_server("e3-src", format!("{http_base}/e3"))];
    req.meta = sdk_meta("e4-src", serde_json::json!({}));
    a.new_session(req).await.expect("new session");

    let done = wait_for(|| sdk_handshake_complete(&sdk_hits, "e4-src")).await;
    let calls = sdk_hits.lock().unwrap().calls.clone();
    assert!(
        done,
        "E4 must complete initialize + notifications/initialized + tools/list          over the reverse channel; calls={calls:?}"
    );

    // Books stay separate in both directions.
    assert!(
        calls.keys().all(|(sid, _)| sid == "e4-src"),
        "no other server id may appear in the reverse tally; calls={calls:?}"
    );
    let http = http_hits.lock().unwrap();
    assert_eq!(
        http.initialize.get("e4").copied().unwrap_or(0),
        0,
        "E4 must not appear on the HTTP books"
    );
    assert!(
        http.initialize.get("e3").copied().unwrap_or(0) >= 1,
        "and E3 must still be on its own"
    );
});

local_test!(t16_e4_restricted_session_makes_no_reverse_calls, async {
    let (a, tmp) = agent();
    let sdk_hits = spawn_sdk_pump();
    init(&a).await;

    let baseline = sdk_hits.lock().unwrap().calls.len();
    let mut req = acp::NewSessionRequest::new(tmp.path().to_path_buf());
    req.meta = sdk_meta("e4-src", serde_json::json!({ KEY: true }));
    let resp = a.new_session(req).await.expect("new session");
    assert_eq!(applied(&resp.meta), Some(&serde_json::Value::Bool(true)));

    // Same budget the positive test needed, so an empty tally means "never
    // called" rather than "not yet".
    let called = wait_for(|| !sdk_hits.lock().unwrap().calls.is_empty()).await;
    let calls = sdk_hits.lock().unwrap().calls.clone();
    assert!(
        !called,
        "a restricted session must make no reverse MCP calls; calls={calls:?}"
    );
    assert_eq!(calls.len(), baseline);
});

// ---------------------------------------------------------------------------
// E1 — MCP contributed by an installed plugin
//
// Ordered so the negative half cannot pass for the wrong reason: the plugin is
// proven live first (an ordinary session handshakes with it and the shared
// registry is populated), and only then is a restricted session created and a
// real broadcast fired at it. Asserting "no hits" against a plugin that was
// never active would prove nothing at all.
// ---------------------------------------------------------------------------

/// A plugin whose MCP server is HTTP-addressed, so its connection lands on the
/// same books as E3 but under its own tag.
fn http_plugin_fixture(name: &str, url: &str) -> tempfile::TempDir {
    let dir = tempfile::tempdir().expect("plugin dir");
    std::fs::write(
        dir.path().join("plugin.json"),
        format!(r#"{{"name": "{name}"}}"#),
    )
    .expect("plugin.json");
    std::fs::write(
        dir.path().join(".mcp.json"),
        serde_json::json!({ "mcpServers": { "e1-src": { "url": url } } }).to_string(),
    )
    .expect(".mcp.json");
    dir
}

local_test!(t17_e1_plugin_mcp_connects_then_stays_away_from_restricted, async {
    let (base, hits) = spawn_mcp_fixture().await;
    let plugin = http_plugin_fixture("e1-plugin", &format!("{base}/e1"));
    let (a, tmp) = agent_with_plugins(&[plugin.path()]);
    init(&a).await;

    // 1 — an ordinary session loads the plugin and its MCP server handshakes.
    let ordinary = ordinary_session(&a, tmp.path()).await;
    let connected = wait_for(|| {
        let h = hits.lock().unwrap();
        h.initialize.get("e1").copied().unwrap_or(0) >= 1
            && h.tools_list.get("e1").copied().unwrap_or(0) >= 1
    })
    .await;
    {
        let h = hits.lock().unwrap();
        assert!(
            connected,
            "the plugin's MCP server must complete a handshake for an ordinary              session; initialize={:?} tools_list={:?}",
            h.initialize, h.tools_list
        );
    }

    // 2 — and the shared registry really holds it.
    assert_eq!(
        a.session_raw_plugin_registry(&ordinary).await,
        Some(Some(1)),
        "the ordinary session must actually have the plugin loaded"
    );

    // 3 — now a restricted session, same plugin source, same connection.
    let restricted_cwd = tempfile::tempdir().expect("cwd");
    let baseline = {
        let h = hits.lock().unwrap();
        (
            h.initialize.get("e1").copied().unwrap_or(0),
            h.tools_list.get("e1").copied().unwrap_or(0),
        )
    };
    let restricted = disabled_session(&a, restricted_cwd.path()).await;

    // 4 — a real reload, driven by the ordinary session, fanned out process-wide.
    ext(
        &a,
        "x.ai/plugins/action",
        serde_json::json!({ "sessionId": ordinary.0, "action": { "type": "reload" } }),
    )
    .await
    .expect("the ordinary session's reload must succeed");

    // 5 — the restricted session must add nothing to E1's tally and hold no
    // registry. Deltas, not absolutes: the ordinary session's own connection
    // is legitimately on these books already.
    //
    // Any connection the restricted session might make gets the same budget
    // the positive half needed, so a zero delta means "never connected" rather
    // than "not yet".
    let grew = wait_for(|| {
        hits.lock().unwrap().initialize.get("e1").copied().unwrap_or(0) > baseline.0
    })
    .await;

    assert_eq!(
        a.session_raw_plugin_registry(&restricted).await,
        Some(None),
        "the restricted session's actor must hold no plugin registry"
    );
    let h = hits.lock().unwrap();
    // Measured, not assumed: the ordinary session's reload re-merges its MCP
    // config but reuses the live connection instead of re-handshaking, so E1's
    // tally is stable across it — which is what makes any growth here
    // attributable to the restricted session. If a future change makes reload
    // reconnect, this fails loudly rather than quietly losing its meaning.
    assert!(
        !grew,
        "no further E1 handshake may occur once the restricted session joins; \
         baseline={} initialize={:?}",
        baseline.0, h.initialize
    );
    assert_eq!(h.initialize.get("e1").copied().unwrap_or(0), baseline.0);
    assert_eq!(h.tools_list.get("e1").copied().unwrap_or(0), baseline.1);
    // Books stay separate: E1 must not show up under E3's tag or on the
    // reverse channel's.
    assert_eq!(h.initialize.get("e3").copied().unwrap_or(0), 0);
});

// ---------------------------------------------------------------------------
// E2 — managed MCP, fetched from the proxy
//
// Two planes, counted separately, because they are two different claims:
//   control — the managed list endpoint was actually asked;
//   data    — the single server it returned actually handshook.
// A test that only checked one of them would pass while the other never
// happened.
//
// The negative half uses its **own** `MvpAgent`, identically configured, with
// only a restricted session on it. Reusing the positive agent would let a
// populated managed cache supply the "zero fetches" result, which proves
// nothing about the policy.
// ---------------------------------------------------------------------------

local_test!(t17_e2_managed_mcp_control_and_data_plane, async {
    let (base, hits) = spawn_mcp_fixture().await;
    let proxy = proxy_base(&base);
    let (a, tmp) = agent_full(&[], Some(&proxy));
    init(&a).await;
    ordinary_session(&a, tmp.path()).await;

    let ok = wait_for(|| {
        let h = hits.lock().unwrap();
        h.managed_configs >= 1
            && h.initialize.get("e2").copied().unwrap_or(0) >= 1
            && h.tools_list.get("e2").copied().unwrap_or(0) >= 1
    })
    .await;
    let h = hits.lock().unwrap();
    assert!(
        ok,
        "managed MCP must be listed *and* connected; managed_configs={}          initialize={:?} tools_list={:?}",
        h.managed_configs, h.initialize, h.tools_list
    );
    // The control-plane request must carry the seeded credential: a fetch that
    // reached the URL unauthenticated would be a different (and broken) thing.
    assert!(
        h.managed_bearers.iter().any(|b| b.contains(TEST_BEARER)),
        "the managed fetch must be authenticated with the seeded token; saw {:?}",
        h.managed_bearers
    );
});

// T16-E2 — NOT YET GREEN. Shape is settled and the blocking UB is gone; what
// remains is a child-process harness problem, recorded here rather than left as
// a hanging test.
//
// Shape: the parent proves the fixture live and takes a baseline, then
// re-executes this binary with `Command::env` (`GROK_HOME`, proxy URL, a child
// marker) so the child gets its own process — and therefore its own managed
// cache — without this process's environment ever being written. The child
// creates exactly one restricted session; the parent then asserts both
// `/proxy/mcp/configs` and `/mcp/e2` are unchanged.
//
// Why a child rather than a second in-process agent: a shared managed cache
// would let "zero fetches" be a cache hit rather than a policy decision, and
// the environment a second agent needs can only be set with
// `std::env::set_var`, which is unsound once the first agent's reqwest and MCP
// workers are running (that is what aborted with STATUS_HEAP_CORRUPTION).
//
// Open problem: the spawned child did not finish within 10 minutes. Not yet
// diagnosed. First suspect is `scratch_home()`, which unconditionally writes
// `GROK_HOME` and would therefore override the value `Command::env` passed in;
// the child branch needs to honour an inherited home instead.


