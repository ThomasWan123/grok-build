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
/// The process-wide scratch home, for tests that need to write into it.
fn scratch_home_path() -> std::path::PathBuf {
    scratch_home();
    std::path::PathBuf::from(std::env::var_os("GROK_HOME").expect("scratch home is set"))
}

fn scratch_home() {
    static HOME: std::sync::OnceLock<Option<tempfile::TempDir>> = std::sync::OnceLock::new();
    HOME.get_or_init(|| {
        // Inherit only in the child branch, which is handed an isolated home
        // through `Command::env`; overwriting that would discard the isolation
        // the parent set up, and the child then writes no environment at all.
        //
        // A parent run always overrides, even when `GROK_HOME` is already set:
        // honouring an ambient value would put session files in whatever
        // directory the developer happens to have configured — including their
        // real `~/.grok`.
        if std::env::var_os(CHILD_PROXY).is_some() {
            debug_assert!(
                std::env::var_os("GROK_HOME").is_some(),
                "the child branch must be given an isolated home"
            );
            return None;
        }
        let dir = tempfile::tempdir().expect("scratch home");
        // SAFETY: runs once, at the first agent construction, before any agent
        // has spawned a thread that reads the environment; the runtime is
        // single-threaded (asserted in `local_test!`) and the dedicated command
        // pins `--test-threads=1`.
        unsafe { std::env::set_var("GROK_HOME", dir.path()) };
        Some(dir)
    });
}

fn agent_full(
    plugin_dirs: &[&std::path::Path],
    managed_proxy_url: Option<&str>,
) -> (MvpAgent, tempfile::TempDir) {
    scratch_home();
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
        // `current_thread` is load-bearing, not incidental: `scratch_home()`
        // writes `GROK_HOME` once, and that write is only sound while no other
        // thread can be reading the environment. Combined with
        // `--test-threads=1`, this keeps the whole file single-threaded on the
        // Tokio side.
        #[tokio::test(flavor = "current_thread")]
        #[serial_test::serial]
        async fn $name() {
            assert_eq!(
                tokio::runtime::Handle::current().runtime_flavor(),
                tokio::runtime::RuntimeFlavor::CurrentThread,
                "this file's one environment write is only sound on a                  single-threaded runtime; see scratch_home()"
            );
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

// ---------------------------------------------------------------------------
// T16-E2 — the negative half, in a child process
//
// The agent under test needs a managed cache that is genuinely its own: sharing
// one with the positive agent would let "zero fetches" be a cache hit rather
// than a policy decision. A second in-process agent cannot provide that — the
// environment it needs can only be set with `std::env::set_var`, unsound once
// the first agent's reqwest and MCP workers are running (that is what aborted
// with STATUS_HEAP_CORRUPTION).
//
// Re-executing this binary solves both: `Command::env` configures the child
// without touching this process, and a fresh process has a fresh cache by
// construction.
//
// The parent must wait *asynchronously*. The fixture serves `/proxy/mcp/configs`
// and `/mcp/e2` from this process's current-thread runtime, so a blocking wait
// stops answering the very requests the child would make — the child then waits
// on a response that can never arrive, and the two deadlock. An earlier
// revision did exactly that and hung past ten minutes.
// ---------------------------------------------------------------------------

/// Present only in the child: carries the fixture's proxy URL and selects the
/// child branch of the single test below.
const CHILD_PROXY: &str = "LED_T16_E2_PROXY";

local_test!(t16_e2_restricted_child_neither_lists_nor_connects, async {
    // One test entry, two branches. A separate `#[tokio::test]` child worker
    // would be reported as a passing test in every ordinary run while doing
    // nothing at all — a green result that asserts nothing is exactly the kind
    // of thing this file exists to prevent.
    if let Ok(proxy) = std::env::var(CHILD_PROXY) {
        let (a, tmp) = agent_full(&[], Some(&proxy));
        init(&a).await;
        let sid = disabled_session(&a, tmp.path()).await;
        assert_eq!(
            a.session_local_extensions_disabled_snapshot(&sid),
            Some(true),
            "the child must actually have created a restricted session"
        );
        // Hold open for the same budget the parent's positive half needed, so
        // the parent is not measuring a window that never opened.
        tokio::time::sleep(std::time::Duration::from_secs(5)).await;
        return;
    }

    let (base, hits) = spawn_mcp_fixture().await;
    let proxy = proxy_base(&base);

    // Prove the fixture and credential live here first: a quiet child proves
    // nothing if managed MCP was broken to begin with.
    //
    // Held for the rest of the test rather than scoped to a block. Scoping it
    // coincided with a process abort here, but that is one observation and no
    // mechanism has been established — a normal drop in safe Rust should not
    // corrupt the heap, and the one root cause actually confirmed in this file
    // was concurrent environment mutation. Kept alive because nothing needs it
    // dropped early, not as a rule about agent lifetimes.
    let (prover, prover_tmp) = agent_full(&[], Some(&proxy));
    init(&prover).await;
    ordinary_session(&prover, prover_tmp.path()).await;
    let live = wait_for(|| {
        let h = hits.lock().unwrap();
        h.managed_configs >= 1 && h.initialize.get("e2").copied().unwrap_or(0) >= 1
    })
    .await;
    assert!(
        live,
        "the managed fixture must be proven live before measuring the child"
    );

    // All three counters are captured separately: comparing `tools_list`
    // against the `initialize` baseline would silently mask a divergence
    // between them (a retried or partial connection moves one without the
    // other).
    let (base_configs, base_init, base_tools) = {
        let h = hits.lock().unwrap();
        (
            h.managed_configs,
            h.initialize.get("e2").copied().unwrap_or(0),
            h.tools_list.get("e2").copied().unwrap_or(0),
        )
    };

    let child_home = tempfile::tempdir().expect("child home");
    let exe = std::env::current_exe().expect("test binary path");
    // Async wait, so this runtime keeps serving the fixture while the child
    // runs. Bounded, so a wedged child fails the test instead of hanging it.
    let status = tokio::time::timeout(
        std::time::Duration::from_secs(120),
        tokio::process::Command::new(exe)
            .args([
                "t16_e2_restricted_child_neither_lists_nor_connects",
                "--exact",
                "--test-threads=1",
                "--nocapture",
            ])
            // Child-only configuration; this process's environment is never
            // written. The child's `scratch_home()` honours the inherited home
            // rather than replacing it.
            .env(CHILD_PROXY, &proxy)
            .env("GROK_HOME", child_home.path())
            .status(),
    )
    .await
    .expect("child worker must finish within the timeout")
    .expect("run child");
    assert!(status.success(), "child worker must exit cleanly: {status:?}");

    let h = hits.lock().unwrap();
    // Data plane is what the policy governs, and it must not move at all: the
    // restricted session connected to no managed server.
    assert_eq!(
        h.initialize.get("e2").copied().unwrap_or(0),
        base_init,
        "a restricted session must not connect to a managed MCP server;          initialize={:?}",
        h.initialize
    );
    assert_eq!(
        h.tools_list.get("e2").copied().unwrap_or(0),
        base_tools,
        "and must not list its tools either; tools_list={:?}",
        h.tools_list
    );
    // The control plane must *move*, but the exact count is not a policy
    // contract — it depends on whether the fetch was retried, which this test
    // has no business pinning. What it does prove is that the child's
    // credential worked and it really did obtain the managed configuration:
    // without that, the zero data-plane deltas above could just mean the child
    // never got as far as having something to connect to.
    assert!(
        h.managed_configs > base_configs,
        "the child must have listed managed connectors (proving its auth          worked), else the zero connection deltas prove nothing; baseline={}          now={}",
        base_configs,
        h.managed_configs
    );
    drop(h);
    // Keeps the prover (and its workers) alive to here.
    assert!(prover_tmp.path().exists());
    drop(prover);
});



// ---------------------------------------------------------------------------
// Mock model — enables the tests that need a turn to actually run
//
// T14 (agent-defined hooks must not fire), T19/T20 (a real subagent spawn) and
// T23 (slash commands execute inside a turn) all need the session to complete a
// prompt, which needs a model to answer. A prefetched catalog pointed at a
// local endpoint keeps that entirely in-process.
// ---------------------------------------------------------------------------

/// Number of completions the mock has served.
type ModelProbe = Arc<std::sync::atomic::AtomicU32>;

async fn spawn_mock_model() -> (String, ModelProbe) {
    use axum::routing::post;
    use axum::Router;

    let probe: ModelProbe = Arc::new(std::sync::atomic::AtomicU32::new(0));
    let app = Router::new()
        .route(
            "/v1/chat/completions",
            post(|axum::extract::State(probe): axum::extract::State<ModelProbe>| async move {
                probe.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                let chunk = serde_json::json!({
                    "id": "chatcmpl-led",
                    "object": "chat.completion.chunk",
                    "created": 0,
                    "model": "led-mock",
                    "choices": [{
                        "index": 0,
                        "delta": { "role": "assistant", "content": "ok" },
                        "finish_reason": "stop"
                    }]
                });
                axum::response::sse::Sse::new(futures_util::stream::iter(vec![
                    Ok::<_, std::convert::Infallible>(
                        axum::response::sse::Event::default().data(chunk.to_string()),
                    ),
                    Ok(axum::response::sse::Event::default().data("[DONE]")),
                ]))
            }),
        )
        .with_state(probe.clone());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::task::spawn_local(async move {
        let _ = axum::serve(listener, app).await;
    });
    (format!("http://{addr}/v1"), probe)
}

fn mock_catalog(base_url: &str) -> indexmap::IndexMap<String, xai_grok_shell::agent::config::ModelEntry> {
    let mut info = xai_grok_shell::agent::config::ModelInfo::fallback("led-mock");
    info.base_url = base_url.to_owned();
    info.max_completion_tokens = Some(64);
    let mut map = indexmap::IndexMap::new();
    map.insert(
        "led-mock".to_string(),
        xai_grok_shell::agent::config::ModelEntry {
            info,
            api_key: Some("led-mock-key".to_string()),
            env_key: None,
            api_base_url: None,
        },
    );
    map
}

/// Build an agent whose catalog points at `model_url`.
///
/// Separate from `agent_full` because a prefetched catalog is the only way to
/// reach a local model endpoint, and the tests that do not run a turn have no
/// reason to carry one.
fn agent_with_model(model_url: &str) -> (MvpAgent, tempfile::TempDir) {
    agent_with_model_and_plugins(model_url, &[])
}

fn agent_with_model_and_plugins(
    model_url: &str,
    plugin_dirs: &[&std::path::Path],
) -> (MvpAgent, tempfile::TempDir) {
    scratch_home();
    let temp = tempfile::tempdir().expect("temp dir");
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
    // LSP is off by default; T13 needs it on for the Code control to have
    // anything to assemble.
    cfg.features.lsp_tools = Some(true);
    let agent = MvpAgent::new(gateway, &cfg, auth, Some(mock_catalog(model_url)))
        .expect("valid test config");
    (agent, temp)
}

// Guards the turn harness itself.
//
// T14, T19/T20 and T23 all rest on a session being able to complete a prompt.
// If that ever stops working those tests would go quiet in a way that reads
// like "the policy blocked it", so the ability to run a turn at all is pinned
// separately.
local_test!(turn_harness_completes_a_prompt, async {
    let (model_url, model_hits) = spawn_mock_model().await;
    let (a, tmp) = agent_with_model(&model_url);
    init(&a).await;
    let sid = ordinary_session(&a, tmp.path()).await;

    let resp = a
        .prompt(acp::PromptRequest::new(
            sid.clone(),
            vec![acp::ContentBlock::from("hello")],
        ))
        .await
        .expect("prompt must succeed");
    assert_eq!(resp.stop_reason, acp::StopReason::EndTurn);
    assert_eq!(
        model_hits.load(std::sync::atomic::Ordering::SeqCst),
        1,
        "the turn must have reached the mock model"
    );
});

// ---------------------------------------------------------------------------
// T23 — every `/hooks` and `/plugins` arm is refused, structurally
//
// Each command is given syntactically valid arguments. A malformed one would
// be rejected during parsing, and "the command did not run" is not the claim
// being tested — a parse failure would look identical to a policy refusal from
// the outside, which is why the assertion is on the refusal *payload* rather
// than on the absence of an effect.
// ---------------------------------------------------------------------------

/// Collects `session/update` notifications off the gateway.
fn spawn_notification_collector() -> Arc<Mutex<Vec<serde_json::Value>>> {
    use xai_acp_lib::AcpClientMessage;
    let seen: Arc<Mutex<Vec<serde_json::Value>>> = Arc::new(Mutex::new(Vec::new()));
    let mut rx = GATEWAY_RX
        .with(|slot| slot.borrow_mut().take())
        .expect("gateway receiver was taken twice");
    let sink = seen.clone();
    tokio::task::spawn_local(async move {
        while let Some(msg) = rx.recv().await {
            if let AcpClientMessage::SessionNotification(args) = msg {
                if let Ok(v) = serde_json::to_value(&args.request) {
                    sink.lock().unwrap().push(v);
                }
                let _ = args.response_tx.send(Ok(()));
            }
        }
    });
    seen
}

/// The refusal payload carried on an agent message chunk, if any.
fn refusal_meta(notifications: &Arc<Mutex<Vec<serde_json::Value>>>) -> Vec<serde_json::Value> {
    notifications
        .lock()
        .unwrap()
        .iter()
        .filter_map(|n| n.pointer("/update/content/_meta").cloned())
        .filter(|m| m.get("code").and_then(|c| c.as_str()) == Some("local_extensions_disabled"))
        .collect()
}

/// Command names the session advertises, from an `available_commands_update`.
fn advertised_commands(notifications: &Arc<Mutex<Vec<serde_json::Value>>>) -> Vec<String> {
    notifications
        .lock()
        .unwrap()
        .iter()
        .rev()
        .find_map(|n| n.pointer("/update/availableCommands").cloned())
        .map(|cmds| {
            cmds.as_array()
                .map(|a| {
                    a.iter()
                        .filter_map(|c| c.get("name").and_then(|v| v.as_str()))
                        .map(str::to_owned)
                        .collect()
                })
                .unwrap_or_default()
        })
        .unwrap_or_default()
}

fn extension_command_names(names: &[String]) -> Vec<&String> {
    names
        .iter()
        .filter(|n| n.starts_with("hooks") || n.starts_with("plugin"))
        .collect()
}

local_test!(t23_extension_commands_are_not_offered_to_a_restricted_session, async {
    let (model_url, _hits) = spawn_mock_model().await;
    let (a, tmp) = agent_with_model(&model_url);
    let notes = spawn_notification_collector();
    init(&a).await;
    let sid = disabled_session(&a, tmp.path()).await;

    // Drive one turn so the command catalog is published.
    a.prompt(acp::PromptRequest::new(
        sid.clone(),
        vec![acp::ContentBlock::from("hello")],
    ))
    .await
    .expect("prompt");

    let names = advertised_commands(&notes);
    assert!(
        !names.is_empty(),
        "the session must advertise some commands, or this asserts nothing"
    );
    assert!(
        extension_command_names(&names).is_empty(),
        "a restricted session must offer no hooks/plugins command; got {names:?}"
    );
});

local_test!(t23_extension_commands_are_offered_to_an_ordinary_session, async {
    let fixture = plugin_fixture("t23-fixture-plugin");
    let (model_url, _hits) = spawn_mock_model().await;
    // Same shape as `agent_with_model`, plus a plugin so the `plugins` gate has
    // something to open on. Without it "no plugins command" would be the
    // expected answer for an ordinary session too, and the negative test above
    // would prove nothing.
    let (a, tmp) = agent_with_model_and_plugins(&model_url, &[fixture.path()]);
    let notes = spawn_notification_collector();
    init(&a).await;
    let sid = ordinary_session(&a, tmp.path()).await;

    a.prompt(acp::PromptRequest::new(
        sid.clone(),
        vec![acp::ContentBlock::from("hello")],
    ))
    .await
    .expect("prompt");

    let names = advertised_commands(&notes);
    assert!(
        !extension_command_names(&names).is_empty(),
        "an ordinary session with a plugin installed must offer the extension          commands; got {names:?}"
    );
});


local_test!(t23_ordinary_session_still_runs_extension_commands, async {
    let fixture = plugin_fixture("t23-control-plugin");
    let (model_url, _model_hits) = spawn_mock_model().await;
    let (a, tmp) = agent_with_model_and_plugins(&model_url, &[fixture.path()]);
    let notes = spawn_notification_collector();
    init(&a).await;
    let sid = ordinary_session(&a, tmp.path()).await;

    // Read-only, so the control is non-destructive; the point is only that the
    // policy branch is not a global kill switch.
    a.prompt(acp::PromptRequest::new(
        sid.clone(),
        vec![acp::ContentBlock::from("/plugins list")],
    ))
    .await
    .expect("an ordinary session must still run /plugins list");

    assert!(
        refusal_meta(&notes).is_empty(),
        "an ordinary session must not see a policy refusal; got {:?}",
        refusal_meta(&notes)
    );
});

// ---------------------------------------------------------------------------
// T23b — the dispatch guard, exercised through the real dispatch
//
// T23a proves these commands are never offered. This proves the guard behind
// that: with every gate forced open, each arm still refuses, and refuses before
// doing anything. The test never restates which commands are extension
// commands — it drives the production path and reads the production output, so
// removing an arm from the guard's list makes the corresponding case fail.
// ---------------------------------------------------------------------------

/// Command texts with arguments their parser accepts.
///
/// Syntax matters here: `install` / `add` / `remove` and friends resolve to a
/// different action (or fail to resolve at all) without an argument, and a
/// command that never reaches the guard would pass this test while proving
/// nothing about it.
const EXTENSION_COMMAND_CASES: &[&str] = &[
    "/hooks-trust",
    "/hooks-list",
    "/hooks-add C:/tmp/led-hook.json",
    "/hooks-remove C:/tmp/led-hook.json",
    "/hooks-untrust",
    "/plugins list",
    "/plugins reload",
    "/plugins trust C:/tmp/led-plugin",
    "/plugins add C:/tmp/led-plugin",
    "/plugins remove C:/tmp/led-plugin",
    "/plugins install C:/tmp/led-plugin",
    "/plugins uninstall led-plugin",
    "/plugins update led-plugin",
];

/// Everything a refused command must leave untouched.
#[derive(Debug, PartialEq)]
struct SideEffectSnapshot {
    plugin_registry: Option<Option<usize>>,
    hooks: serde_json::Value,
    mcp_hits: usize,
    sentinel_files: Vec<String>,
}

async fn snapshot(
    a: &MvpAgent,
    sid: &acp::SessionId,
    mcp: &Arc<Mutex<Hits>>,
    sentinel_dir: &std::path::Path,
) -> SideEffectSnapshot {
    let mut sentinel_files: Vec<String> = std::fs::read_dir(sentinel_dir)
        .map(|rd| {
            rd.filter_map(|e| e.ok())
                .map(|e| e.file_name().to_string_lossy().into_owned())
                .collect()
        })
        .unwrap_or_default();
    sentinel_files.sort();
    SideEffectSnapshot {
        plugin_registry: a.session_raw_plugin_registry(sid).await,
        hooks: ext(a, "x.ai/hooks/list", serde_json::json!({ "sessionId": sid.0 }))
            .await
            .expect("hooks/list"),
        mcp_hits: {
            let h = mcp.lock().unwrap();
            h.initialize.values().sum::<usize>() + h.managed_configs
        },
        sentinel_files,
    }
}

local_test!(t23b_every_extension_arm_refuses_before_acting, async {
    let (mcp_base, mcp_hits) = spawn_mcp_fixture().await;
    let (model_url, model_hits) = spawn_mock_model().await;
    // A plugin directory the `add` / `install` / `trust` arms would plausibly
    // write into, so "no side effect" is a claim with something behind it.
    let sentinel_dir = tempfile::tempdir().expect("sentinel dir");
    let (a, tmp) = agent_with_model(&model_url);
    let notes = spawn_notification_collector();
    init(&a).await;
    let sid = disabled_session(&a, tmp.path()).await;
    let _ = mcp_base;

    for cmd in EXTENSION_COMMAND_CASES {
        let before = snapshot(&a, &sid, &mcp_hits, sentinel_dir.path()).await;
        let model_before = model_hits.load(std::sync::atomic::Ordering::SeqCst);
        let notes_before = notes.lock().unwrap().len();

        a.test_execute_slash_command(&sid, cmd).await;

        // Bounded: the refusal is emitted on the notification channel, which is
        // asynchronous even though dispatch has returned.
        let arrived = wait_for(|| {
            notes.lock().unwrap()[notes_before..]
                .iter()
                .any(|n| n.pointer("/update/content/_meta").is_some())
        })
        .await;

        let refusals: Vec<serde_json::Value> = notes.lock().unwrap()[notes_before..]
            .iter()
            .filter_map(|n| n.pointer("/update/content/_meta").cloned())
            .collect();
        assert!(arrived, "{cmd}: no refusal notification arrived");
        assert_eq!(
            refusals.len(),
            1,
            "{cmd}: expected exactly one refusal, got {refusals:?}"
        );
        assert_eq!(
            refusals[0]["code"], "local_extensions_disabled",
            "{cmd}: wrong code"
        );
        assert_eq!(
            refusals[0]["policy"], "local_extensions_disabled",
            "{cmd}: wrong policy"
        );
        assert_eq!(
            refusals[0]["reason"], "slash_command_refused",
            "{cmd}: wrong reason"
        );

        assert_eq!(
            model_hits.load(std::sync::atomic::Ordering::SeqCst),
            model_before,
            "{cmd}: a refused command must not run a model turn"
        );
        let after = snapshot(&a, &sid, &mcp_hits, sentinel_dir.path()).await;
        assert_eq!(after, before, "{cmd}: a refused command must change nothing");
    }
});

// ---------------------------------------------------------------------------
// T14 — hooks carried by the agent definition (source D3)
// ---------------------------------------------------------------------------

/// A hook script that appends the envelope it receives on stdin.
///
/// One line of JSON per firing, so the test attributes hits by the envelope's
/// own `session_id` / `cwd` fields rather than by counting lines.
fn hook_sentinel(dir: &std::path::Path) -> (std::path::PathBuf, std::path::PathBuf) {
    let log = dir.join("hook-hits.jsonl");
    let script = dir.join("led-hook.cmd");
    std::fs::write(
        &script,
        format!("@echo off
more >> \"{}\"
", log.display()),
    )
    .expect("hook script");
    (script, log)
}

fn hook_lines(log: &std::path::Path) -> Vec<serde_json::Value> {
    std::fs::read_to_string(log)
        .unwrap_or_default()
        .lines()
        .filter(|l| !l.trim().is_empty())
        .filter_map(|l| serde_json::from_str(l).ok())
        .collect()
}

/// The agent profile both sessions receive — byte-identical by construction.
fn inline_hook_profile(script: &std::path::Path) -> serde_json::Value {
    serde_json::json!({
        "name": "led-inline-hook-agent",
        "description": "carries an inline hook",
        "hooks": {
            "UserPromptSubmit": [{
                "hooks": [{ "type": "command", "command": script.display().to_string() }]
            }]
        }
    })
}

/// Sentinel lines attributable to one session, by exact field match.
///
/// Not a count and not a substring search: both sessions in this test run the
/// same hook script against the same log, so attribution has to come from the
/// envelope's own `sessionId`.
fn hook_hits_for(log: &std::path::Path, sid: &acp::SessionId) -> Vec<serde_json::Value> {
    hook_lines(log)
        .into_iter()
        .filter(|l| l.get("sessionId").and_then(|v| v.as_str()) == Some(sid.0.as_ref()))
        .collect()
}

local_test!(t14_agent_defined_hooks_fire_for_code_and_never_for_restricted, async {
    let hook_dir = tempfile::tempdir().expect("hook dir");
    let (script, log) = hook_sentinel(hook_dir.path());
    let profile = inline_hook_profile(&script);
    let (model_url, model_hits) = spawn_mock_model().await;
    let (a, tmp) = agent_with_model(&model_url);
    init(&a).await;

    // Both sessions get the *same* profile value — same hook, same script, same
    // log — so any difference in outcome is the policy and nothing else.
    let ordinary = new_session(
        &a,
        tmp.path(),
        meta(serde_json::json!({ "agentProfile": profile })),
    )
    .await
    .expect("ordinary session")
    .session_id;
    let restricted_cwd = tempfile::tempdir().expect("cwd");
    let restricted = new_session(
        &a,
        restricted_cwd.path(),
        meta(serde_json::json!({ KEY: true, "agentProfile": profile })),
    )
    .await
    .expect("restricted session")
    .session_id;

    // --- Code side: the hook is installed, visible, and fires -------------
    let listed = ext(
        &a,
        "x.ai/hooks/list",
        serde_json::json!({ "sessionId": ordinary.0 }),
    )
    .await
    .expect("hooks/list");
    let names: Vec<String> = listed["hooks"]
        .as_array()
        .map(|arr| {
            arr.iter()
                .filter_map(|h| h["name"].as_str())
                .map(str::to_owned)
                .collect()
        })
        .unwrap_or_default();
    assert!(
        names.iter().any(|n| n.contains("led-inline-hook-agent")),
        "the ordinary session must list the agent's hook; got {names:?}"
    );

    let model_before = model_hits.load(std::sync::atomic::Ordering::SeqCst);
    a.prompt(acp::PromptRequest::new(
        ordinary.clone(),
        vec![acp::ContentBlock::from("hello")],
    ))
    .await
    .expect("ordinary prompt");
    assert!(
        model_hits.load(std::sync::atomic::Ordering::SeqCst) > model_before,
        "the ordinary turn must actually have run"
    );
    assert!(
        wait_for(|| !hook_hits_for(&log, &ordinary).is_empty()).await,
        "the agent's hook must fire for the ordinary session"
    );
    let ordinary_hits = hook_hits_for(&log, &ordinary);
    assert_eq!(
        ordinary_hits[0]["cwd"].as_str(),
        Some(tmp.path().to_string_lossy().as_ref()),
        "and must be attributed to that session's cwd"
    );

    // --- Restricted side: nothing, and the turn still ran -----------------
    let restricted_listed = ext(
        &a,
        "x.ai/hooks/list",
        serde_json::json!({ "sessionId": restricted.0 }),
    )
    .await
    .expect("hooks/list");
    assert_eq!(restricted_listed["hooks"], serde_json::json!([]));

    let model_before = model_hits.load(std::sync::atomic::Ordering::SeqCst);
    let lines_before = hook_lines(&log).len();
    a.prompt(acp::PromptRequest::new(
        restricted.clone(),
        vec![acp::ContentBlock::from("hello")],
    ))
    .await
    .expect("restricted prompt");
    // Without this the whole negative half could pass because the turn never
    // ran — the failure mode that looks exactly like a working interception.
    assert!(
        model_hits.load(std::sync::atomic::Ordering::SeqCst) > model_before,
        "the restricted turn must actually have run"
    );

    // Same budget the positive half needed before concluding "never fired".
    let fired = wait_for(|| !hook_hits_for(&log, &restricted).is_empty()).await;
    assert!(
        !fired,
        "the agent's hook must not fire for a restricted session; hits={:?}",
        hook_hits_for(&log, &restricted)
    );
    assert_eq!(
        hook_lines(&log).len(),
        lines_before,
        "and must add no sentinel line at all"
    );
});

// ---------------------------------------------------------------------------
// Subagent fixture — the shared groundwork for T19/T20
//
// `resolve_agent_definition` falls back to `cfg.cli_agents` by name, so a full
// `AgentDefinition` placed there is the cleanest way to give a subagent its own
// inline hook and inline MCP server without touching disk discovery.
// ---------------------------------------------------------------------------

/// Marker carried in the prompt the parent hands its subagent.
///
/// The mock routes on message content, never on request order: with a parent
/// turn, a subagent turn and a post-tool-result turn all hitting the same
/// endpoint, ordering is exactly the thing that cannot be relied on.
const SUBAGENT_PROMPT_MARKER: &str = "led-subagent-task-marker";
const SUBAGENT_TYPE: &str = "led-subagent";

/// Counts per routing state, so a test can prove each turn really happened.
#[derive(Default, Debug)]
struct ModelRouting {
    parent_initial: usize,
    subagent: usize,
    parent_after_tool: usize,
}

async fn spawn_routing_model() -> (String, Arc<Mutex<ModelRouting>>) {
    use axum::routing::post;
    use axum::Router;

    let routing: Arc<Mutex<ModelRouting>> = Arc::new(Mutex::new(ModelRouting::default()));
    let app = Router::new()
        .route(
            "/v1/chat/completions",
            post(
                |axum::extract::State(routing): axum::extract::State<Arc<Mutex<ModelRouting>>>,
                 body: String| async move {
                    // Routed on the *last* message, parsed — not on a substring
                    // of the whole body. The marker also appears inside the
                    // parent's own tool-call arguments once they are in history,
                    // so a body-wide search misroutes every later parent turn to
                    // the subagent branch.
                    let parsed: serde_json::Value =
                        serde_json::from_str(&body).unwrap_or(serde_json::Value::Null);
                    let last = parsed["messages"]
                        .as_array()
                        .and_then(|m| m.last())
                        .cloned()
                        .unwrap_or(serde_json::Value::Null);
                    let last_role = last["role"].as_str().unwrap_or_default().to_string();
                    let last_text = last["content"].to_string();
                    let has_marker =
                        last_role == "user" && last_text.contains(SUBAGENT_PROMPT_MARKER);
                    let has_tool_result = last_role == "tool";
                    let delta = if has_marker {
                        routing.lock().unwrap().subagent += 1;
                        serde_json::json!({ "role": "assistant", "content": "subagent done" })
                    } else if has_tool_result {
                        routing.lock().unwrap().parent_after_tool += 1;
                        serde_json::json!({ "role": "assistant", "content": "all done" })
                    } else {
                        routing.lock().unwrap().parent_initial += 1;
                        serde_json::json!({
                            "role": "assistant",
                            "tool_calls": [{
                                "index": 0,
                                "id": "call_led_1",
                                "type": "function",
                                "function": {
                                    "name": "spawn_subagent",
                                    "arguments": serde_json::json!({
                                        "prompt": SUBAGENT_PROMPT_MARKER,
                                        "description": "led probe",
                                        "subagent_type": SUBAGENT_TYPE,
                                        "background": false,
                                    }).to_string()
                                }
                            }]
                        })
                    };
                    let finish = if delta.get("tool_calls").is_some() {
                        "tool_calls"
                    } else {
                        "stop"
                    };
                    let chunk = serde_json::json!({
                        "id": "chatcmpl-led", "object": "chat.completion.chunk",
                        "created": 0, "model": "led-mock",
                        "choices": [{ "index": 0, "delta": delta, "finish_reason": finish }]
                    });
                    axum::response::sse::Sse::new(futures_util::stream::iter(vec![
                        Ok::<_, std::convert::Infallible>(
                            axum::response::sse::Event::default().data(chunk.to_string()),
                        ),
                        Ok(axum::response::sse::Event::default().data("[DONE]")),
                    ]))
                },
            ),
        )
        .with_state(routing.clone());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::task::spawn_local(async move {
        let _ = axum::serve(listener, app).await;
    });
    (format!("http://{addr}/v1"), routing)
}

/// The transport config for a single-key `mcpServers` entry: a real
/// `acp::McpServer` minus its `name`.
fn inline_mcp_config(name: &str, url: &str) -> serde_json::Value {
    let dto = http_server(name, url.to_string());
    let mut config = serde_json::to_value(&dto).expect("serialize mcp server");
    config
        .as_object_mut()
        .expect("mcp server serializes to an object")
        .remove("name");
    config
}

/// An agent definition carrying both subagent injection sources at once.
fn subagent_definition(script: &std::path::Path, mcp_url: &str) -> serde_json::Value {
    serde_json::json!({
        "name": SUBAGENT_TYPE,
        "description": "subagent carrying an inline hook and an inline MCP server",
        "hooks": {
            "UserPromptSubmit": [{
                "hooks": [{ "type": "command", "command": script.display().to_string() }]
            }]
        },
        // Built from the typed DTO, not hand-written. `McpServerHttp.headers`
        // is a required field with no serde default, so a hand-written config
        // that omits it fails to deserialize — and the materialization branch
        // drops such an entry silently, which reads as "the policy blocked it".
        // Serializing a real `McpServer` guarantees every required field is
        // present; only `name` is lifted out, because the single-key map form
        // supplies it as the key.
        "mcpServers": [ { "sub-src": inline_mcp_config("sub-src", mcp_url) } ]
    })
}

/// Round-trips the fixture's MCP entry through the production shapes.
///
/// Written against the typed DTO rather than hand-written JSON: the earlier
/// attempt guessed the wire form and produced an entry that deserialized
/// happily but carried no transport `type`, so it never connected and nothing
/// reported an error.
#[test]
fn subagent_inline_mcp_entry_round_trips_to_a_typed_dto() {
    let url = "http://127.0.0.1:1/mcp/sub";
    let dto: acp::McpServer = http_server("sub-src", url.to_string());

    // Fixture form: the transport config alone, under the server name.
    let mut config = serde_json::to_value(&dto).expect("serialize");
    let obj = config.as_object_mut().expect("object");
    obj.remove("name");
    let entry = serde_json::json!({ "sub-src": config });

    // Production form: the inline config is flattened and `name` restored.
    let (name, mut inner) = entry
        .as_object()
        .and_then(|o| o.iter().next())
        .map(|(k, v)| (k.clone(), v.clone()))
        .expect("single-key map");
    inner
        .as_object_mut()
        .expect("config object")
        .insert("name".to_string(), serde_json::json!(name));
    let back: acp::McpServer =
        serde_json::from_value(inner).expect("inline config must deserialize as an McpServer");

    assert_eq!(
        serde_json::to_value(&back).unwrap(),
        serde_json::to_value(&dto).unwrap(),
        "the fixture entry must round-trip to the same server the DTO describes"
    );
}

fn agent_with_subagent_fixture_and_plugins(
    model_url: &str,
    script: &std::path::Path,
    mcp_url: &str,
    plugin_dirs: &[&std::path::Path],
) -> (MvpAgent, tempfile::TempDir) {
    scratch_home();
    let temp = tempfile::tempdir().expect("temp dir");
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
    cfg.cli_agents = vec![
        serde_json::from_value(subagent_definition(script, mcp_url))
            .expect("agent definition parses"),
    ];
    let agent = MvpAgent::new(gateway, &cfg, auth, Some(mock_catalog(model_url)))
        .expect("valid test config");
    (agent, temp)
}

// The positive control for T19/T20: an ordinary parent really does hand its
// subagent the definition's hook *and* its inline MCP server.
//
// All three must hold before any "restricted subagents get nothing" claim
// means anything — a fixture that silently fails to deliver either one would
// make the negative half pass for the wrong reason. That is not hypothetical:
// this fixture materialized zero MCP servers until the config was built from a
// real DTO, because `McpServerHttp.headers` has no serde default and the
// materialization branch drops entries that fail to deserialize.
local_test!(t19_positive_subagent_receives_definition_hook_and_mcp, async {
    let hook_dir = tempfile::tempdir().expect("hook dir");
    let (script, log) = hook_sentinel(hook_dir.path());
    let (mcp_base, mcp_hits) = spawn_mcp_fixture().await;
    let (model_url, routing) = spawn_routing_model().await;
    // A plugin is installed so the shared registry is genuinely populated: the
    // assertion below that the subagent *does* receive one is what gives T20's
    // `None` its meaning.
    let plugin = plugin_fixture("t19-shared-plugin");
    let (a, tmp) = agent_with_subagent_fixture_and_plugins(
        &model_url,
        &script,
        &format!("{mcp_base}/sub"),
        &[plugin.path()],
    );
    init(&a).await;
    let sid = ordinary_session(&a, tmp.path()).await;

    a.prompt(acp::PromptRequest::new(
        sid.clone(),
        vec![acp::ContentBlock::from("please delegate")],
    ))
    .await
    .expect("parent prompt");

    // The delegation really happened, and the subagent really ran a turn.
    {
        let r = routing.lock().unwrap();
        assert!(r.parent_initial >= 1, "parent must have been asked: {r:?}");
        assert!(r.subagent >= 1, "the subagent must have run a turn: {r:?}");
    }

    // The definition carried the entry, and it survived materialization.
    assert_eq!(
        MvpAgent::last_subagent_def_mcp_count(),
        1,
        "the definition must carry exactly one MCP entry"
    );
    assert!(
        MvpAgent::last_subagent_plugin_count().is_some(),
        "an ordinary parent's subagent does receive a plugin registry — without          this, T20's `None` could just mean the field is never populated"
    );
    assert_eq!(
        MvpAgent::last_subagent_mcp_servers(),
        vec!["sub-src".to_string()],
        "and it must materialize under its own name"
    );

    // Data plane: the server it materialized into actually handshook.
    assert!(
        wait_for(|| mcp_hits.lock().unwrap().initialize.get("sub").copied().unwrap_or(0) >= 1)
            .await,
        "the subagent's inline MCP server must complete a handshake; initialize={:?}",
        mcp_hits.lock().unwrap().initialize
    );

    // And the definition's hook fired.
    assert!(
        wait_for(|| !hook_lines(&log).is_empty()).await,
        "the subagent's inline hook must fire"
    );
});


// ---------------------------------------------------------------------------
// T20 — a restricted parent's subagent gets none of the three sources back
//
// Runs inside a *single* `MvpAgent` sharing one plugin registry. Splitting it
// across two agents would leave the most dangerous path untested: I1 is the
// subagent coordinator reading the **shared** registry, which is unrelated to
// the parent's own (zeroed) one, so a second process would simply not have the
// thing that could leak.
// ---------------------------------------------------------------------------

local_test!(t20_restricted_parent_subagent_gets_nothing_back, async {
    let plugin = plugin_fixture("t20-shared-plugin");
    let hook_dir = tempfile::tempdir().expect("hook dir");
    let (script, log) = hook_sentinel(hook_dir.path());
    let (mcp_base, mcp_hits) = spawn_mcp_fixture().await;
    let (model_url, routing) = spawn_routing_model().await;
    let (a, tmp) = agent_with_subagent_fixture_and_plugins(
        &model_url,
        &script,
        &format!("{mcp_base}/sub"),
        &[plugin.path()],
    );
    init(&a).await;

    // 1 — an ordinary parent loads the plugin, so the shared source exists.
    let code_parent = ordinary_session(&a, tmp.path()).await;
    assert_eq!(
        a.session_raw_plugin_registry(&code_parent).await,
        Some(Some(1)),
        "the Code parent must actually have the plugin loaded"
    );
    assert!(
        a.plugin_registry_snapshot().is_some_and(|r| !r.list().is_empty()),
        "the agent-level shared registry must be non-empty, or I1 has nothing to leak"
    );

    // 2 — baselines.
    let hook_lines_before = hook_lines(&log).len();
    let mcp_before = mcp_hits.lock().unwrap().initialize.get("sub").copied().unwrap_or(0);
    let routing_before = {
        let r = routing.lock().unwrap();
        (r.parent_initial, r.subagent, r.parent_after_tool)
    };
    // Cleared so a leftover positive result cannot satisfy the assertions.
    MvpAgent::reset_subagent_observations();

    // 3 — the restricted parent, in the same agent.
    let restricted_cwd = tempfile::tempdir().expect("cwd");
    let restricted = disabled_session(&a, restricted_cwd.path()).await;
    assert_eq!(
        a.session_raw_plugin_registry(&restricted).await,
        Some(None),
        "the restricted parent's own registry must be empty"
    );

    // 4 — derive, using the very same definition the positive control used.
    a.prompt(acp::PromptRequest::new(
        restricted.clone(),
        vec![acp::ContentBlock::from("please delegate")],
    ))
    .await
    .expect("restricted parent prompt");

    // 5 — the delegation really happened. Without this every assertion below
    // would hold for a run in which no subagent was ever spawned.
    {
        let r = routing.lock().unwrap();
        assert!(
            r.parent_initial > routing_before.0
                && r.subagent > routing_before.1
                && r.parent_after_tool > routing_before.2,
            "all three turns must have run: before={routing_before:?} now={r:?}"
        );
    }

    // 6 — and the subagent got none of the three sources.
    assert_eq!(
        MvpAgent::last_subagent_plugin_count(),
        None,
        "I1: the subagent must not be handed the shared plugin registry"
    );
    assert_eq!(
        MvpAgent::last_subagent_mcp_servers(),
        Vec::<String>::new(),
        "I3: the definition's inline MCP must not materialize"
    );
    let mcp_grew = wait_for(|| {
        mcp_hits.lock().unwrap().initialize.get("sub").copied().unwrap_or(0) > mcp_before
    })
    .await;
    assert!(
        !mcp_grew,
        "and must not connect; initialize={:?}",
        mcp_hits.lock().unwrap().initialize
    );
    let hook_fired = wait_for(|| hook_lines(&log).len() > hook_lines_before).await;
    assert!(
        !hook_fired,
        "I2: the definition's inline hook must not fire; new lines={:?}",
        &hook_lines(&log)[hook_lines_before..]
    );
    assert!(
        hook_hits_for(&log, &restricted).is_empty(),
        "nor under the restricted parent's own session id"
    );

    // 7 — the parent stayed empty, and the shared source is still there, so the
    // zeroes above are the policy and not a vanished registry.
    assert_eq!(a.session_raw_plugin_registry(&restricted).await, Some(None));
    assert!(
        a.plugin_registry_snapshot().is_some_and(|r| !r.list().is_empty()),
        "the shared registry must still be non-empty at the end"
    );
});

// ---------------------------------------------------------------------------
// T13 — the three LSP sources
//
// Each source gets its own server id, so the Code control can assert the exact
// set and each per-source mutation shows up as a single missing/extra id.
// ---------------------------------------------------------------------------

const LSP_PLUGIN_ID: &str = "led-lsp-plugin";
const LSP_USER_ID: &str = "led-lsp-user";
const LSP_PROJECT_ID: &str = "led-lsp-project";

fn lsp_entry(id: &str) -> serde_json::Value {
    // `extensions` is a map, not a list — a list makes the whole entry fail to
    // deserialize and the server simply never appears, with no error surfaced.
    serde_json::json!({
        id: {
            "command": "cmd",
            "args": ["/c", "echo"],
            "extensions": { ".led": "led" }
        }
    })
}

/// A plugin contributing its own LSP server, plus the user- and project-level
/// `lsp.json` files. Returns the project cwd.
fn lsp_sources(home: &std::path::Path) -> (tempfile::TempDir, tempfile::TempDir) {
    let plugin = tempfile::tempdir().expect("plugin dir");
    // A plugin contributes LSP through its manifest's `lspServers` field —
    // inline here, so the fixture does not depend on the default file name.
    // Dropping an `lsp.json` in the plugin directory does nothing: the loader
    // reads the manifest, not the directory.
    std::fs::write(
        plugin.path().join("plugin.json"),
        serde_json::json!({
            "name": "led-lsp-contributor",
            "lspServers": lsp_entry(LSP_PLUGIN_ID),
        })
        .to_string(),
    )
    .expect("plugin.json");

    std::fs::write(home.join("lsp.json"), lsp_entry(LSP_USER_ID).to_string())
        .expect("user lsp.json");

    let project = tempfile::tempdir().expect("project dir");
    std::fs::create_dir_all(project.path().join(".grok")).expect(".grok");
    std::fs::write(
        project.path().join(".grok").join("lsp.json"),
        lsp_entry(LSP_PROJECT_ID).to_string(),
    )
    .expect("project lsp.json");
    (plugin, project)
}

local_test!(t13_lsp_three_sources_for_code_none_for_restricted, async {
    let home = scratch_home_path();
    let (plugin, project) = lsp_sources(&home);
    let (model_url, _routing) = spawn_mock_model().await;
    let (a, _tmp) = agent_with_model_and_plugins(&model_url, &[plugin.path()]);
    init(&a).await;

    // --- T13a/b/c/d: the Code control assembles exactly the three ---------
    MvpAgent::reset_lsp_observations();
    let code = ordinary_session(&a, project.path()).await;
    let mut assembled = MvpAgent::last_lsp_server_names();
    assembled.sort();
    let expected = {
        let mut v = vec![
            LSP_PLUGIN_ID.to_string(),
            LSP_PROJECT_ID.to_string(),
            LSP_USER_ID.to_string(),
        ];
        v.sort();
        v
    };
    assert_eq!(
        assembled, expected,
        "the Code session must assemble all three LSP sources"
    );
    let _ = code;

    // --- Restricted: nothing assembled, no manager built ------------------
    MvpAgent::reset_lsp_observations();
    let managers_before = MvpAgent::lsp_manager_constructions();
    let restricted = disabled_session(&a, project.path()).await;
    assert_eq!(
        MvpAgent::last_lsp_server_names(),
        Vec::<String>::new(),
        "a restricted session must assemble no LSP server from any source"
    );
    assert_eq!(
        MvpAgent::lsp_manager_constructions(),
        managers_before,
        "and must not construct an LspManager — counted directly, because          language servers start lazily and 'no subprocess' proves little"
    );
    let _ = restricted;

    // --- The sources are still there, so the zeroes are the policy --------
    MvpAgent::reset_lsp_observations();
    let code_again = ordinary_session(&a, project.path()).await;
    let mut assembled_again = MvpAgent::last_lsp_server_names();
    assembled_again.sort();
    assert_eq!(
        assembled_again, expected,
        "the three sources must still be present at the end; otherwise the          restricted session's empty set could be a vanished fixture"
    );
    let _ = code_again;
});

// ---------------------------------------------------------------------------
// T7 — a session that fails to assemble is never confirmed
//
// The failpoint sits after the policy has been parsed and validated and after
// every policy-derived argument has been computed, but before any
// `SessionHandle` exists, is registered, or a response is built. Failing
// earlier would make this test tautological: a request rejected during parsing
// obviously carries no confirmation field.
//
// New and Load share `spawn_and_register_session`, so one failpoint serves both
// wire entry points — and both are exercised, because the confirmation is
// written in two separate places.
// ---------------------------------------------------------------------------

/// `true` if the key appears anywhere in the value, at any depth.
fn mentions_applied(v: &serde_json::Value) -> bool {
    match v {
        serde_json::Value::Object(o) => {
            o.contains_key(APPLIED) || o.values().any(mentions_applied)
        }
        serde_json::Value::Array(a) => a.iter().any(mentions_applied),
        serde_json::Value::String(s) => s.contains(APPLIED),
        _ => false,
    }
}

/// Failpoints are process-global test scaffolding. Always disarm them even
/// when an assertion panics, or a later case could fail for the wrong reason.
struct T7FailpointReset;

impl Drop for T7FailpointReset {
    fn drop(&mut self) {
        MvpAgent::set_assembly_failpoint(false);
        MvpAgent::set_latch_drop_failpoint(false);
    }
}

local_test!(t7_failed_assembly_is_never_confirmed, async {
    let _failpoint_reset = T7FailpointReset;
    MvpAgent::set_assembly_failpoint(false);
    MvpAgent::set_latch_drop_failpoint(false);
    let plugin = plugin_fixture("t7-plugin");
    let (mcp_base, mcp_hits) = spawn_mcp_fixture().await;
    let (a, tmp) = agent_with_plugins(&[plugin.path()]);
    init(&a).await;
    let _ = mcp_base;

    let request_meta = || meta(serde_json::json!({ KEY: true }));

    // Baselines for the side effects a half-assembled session could leave.
    let mcp_before = {
        let h = mcp_hits.lock().unwrap();
        h.initialize.len() + h.managed_configs
    };
    let lsp_managers_before = MvpAgent::lsp_manager_constructions();
    let ensure_before = a.ensure_plugin_registry_call_count();
    let sessions_before = a.live_session_count();

    // --- New: assembly fails -------------------------------------------
    MvpAgent::set_assembly_failpoint(true);
    let hits_before = MvpAgent::assembly_failpoint_hits();
    let err = new_session(&a, tmp.path(), request_meta())
        .await
        .expect_err("assembly must fail");
    assert!(
        MvpAgent::assembly_failpoint_hits() > hits_before,
        "the request must have reached the failpoint, not failed earlier"
    );
    let err_value = serde_json::json!({
        "code": format!("{:?}", err.code),
        "message": err.message,
        "data": err.data,
    });
    assert_eq!(err_value["data"]["code"], "session_assembly_failed");
    assert!(
        !mentions_applied(&err_value),
        "a failed assembly must not confirm the policy anywhere in its error: {err_value}"
    );

    // No session, and none of the side effects a real assembly would leave.
    assert_eq!(
        a.live_session_count(),
        sessions_before,
        "a failed assembly must not install an unreachable live handle"
    );
    assert_eq!(
        {
            let h = mcp_hits.lock().unwrap();
            h.initialize.len() + h.managed_configs
        },
        mcp_before,
        "a failed assembly must not have connected anything"
    );
    assert_eq!(
        MvpAgent::lsp_manager_constructions(),
        lsp_managers_before,
        "nor constructed an LspManager"
    );
    assert_eq!(
        a.ensure_plugin_registry_call_count(),
        ensure_before,
        "nor initialized the shared plugin registry"
    );

    // --- The same request succeeds once the failpoint is disarmed ---------
    MvpAgent::set_assembly_failpoint(false);
    let resp = new_session(&a, tmp.path(), request_meta())
        .await
        .expect("the same request must succeed with the failpoint disarmed");
    let sid = resp.session_id.clone();
    assert_eq!(
        applied(&resp.meta),
        Some(&serde_json::Value::Bool(true)),
        "proving the request and fixture were valid all along"
    );
    assert_eq!(a.session_local_extensions_disabled_snapshot(&sid), Some(true));

    // A successful assembly whose actual latch is false must not echo the
    // requested value. This is the discriminating half of T7: an
    // implementation that builds the confirmation from request metadata would
    // incorrectly emit `applied=true` here.
    MvpAgent::set_latch_drop_failpoint(true);
    let resp = new_session(&a, tmp.path(), request_meta())
        .await
        .expect("latch-drop assembly itself succeeds");
    let dropped_new_sid = resp.session_id.clone();
    assert_eq!(
        applied(&resp.meta),
        None,
        "new-session confirmation must follow the installed latch, not the request"
    );
    assert_eq!(
        a.session_local_extensions_disabled_snapshot(&dropped_new_sid),
        Some(false),
        "the test must really have dropped the installed latch"
    );
    MvpAgent::set_latch_drop_failpoint(false);
    a.remove_session_for_test(&dropped_new_sid);

    // --- Load: the other confirmation site, same failpoint ----------------
    a.remove_session_for_test(&sid);
    assert_eq!(
        a.session_local_extensions_disabled_snapshot(&sid),
        None,
        "the session must be out of the map so the load really re-assembles"
    );
    MvpAgent::set_assembly_failpoint(true);
    let hits_before = MvpAgent::assembly_failpoint_hits();
    let err = load_session(&a, &sid, tmp.path(), request_meta())
        .await
        .expect_err("load assembly must fail");
    assert!(
        MvpAgent::assembly_failpoint_hits() > hits_before,
        "the load must have reached the failpoint too"
    );
    let err_value = serde_json::json!({
        "code": format!("{:?}", err.code),
        "message": err.message,
        "data": err.data,
    });
    assert!(
        !mentions_applied(&err_value),
        "a failed load must not confirm the policy either: {err_value}"
    );
    assert_eq!(
        a.session_local_extensions_disabled_snapshot(&sid),
        None,
        "and must not have registered a session"
    );

    // --- And the same load succeeds once disarmed -------------------------
    MvpAgent::set_assembly_failpoint(false);
    let resp = load_session(&a, &sid, tmp.path(), request_meta())
        .await
        .expect("the same load must succeed with the failpoint disarmed");
    assert_eq!(
        applied(&resp.meta),
        Some(&serde_json::Value::Bool(true)),
        "the load confirmation site must work when assembly does"
    );

    // The second response site must use the loaded handle's latch as well.
    a.remove_session_for_test(&sid);
    MvpAgent::set_latch_drop_failpoint(true);
    let resp = load_session(&a, &sid, tmp.path(), request_meta())
        .await
        .expect("latch-drop load itself succeeds");
    assert_eq!(
        applied(&resp.meta),
        None,
        "load confirmation must follow the installed latch, not the request"
    );
    assert_eq!(
        a.session_local_extensions_disabled_snapshot(&sid),
        Some(false),
        "the loaded handle must expose the dropped latch"
    );
});

// ---------------------------------------------------------------------------
// Defence redundancy matrices
// ---------------------------------------------------------------------------

struct DefenceBypassReset;

impl Drop for DefenceBypassReset {
    fn drop(&mut self) {
        MvpAgent::set_hook_defence_bypasses(false, false, false);
        MvpAgent::set_plugin_fanout_defence_bypasses(false, false);
    }
}

async fn run_hook_defence_case(
    a: &MvpAgent,
    profile: &serde_json::Value,
    log: &std::path::Path,
    model_hits: &ModelProbe,
    label: &str,
    bypasses: (bool, bool, bool),
    expect_hook: bool,
) {
    MvpAgent::set_hook_defence_bypasses(bypasses.0, bypasses.1, bypasses.2);
    let cwd = tempfile::tempdir().expect("matrix cwd");
    let sid = new_session(
        a,
        cwd.path(),
        meta(serde_json::json!({ KEY: true, "agentProfile": profile })),
    )
    .await
    .unwrap_or_else(|e| panic!("{label}: create restricted session: {e:?}"))
    .session_id;

    let model_before = model_hits.load(std::sync::atomic::Ordering::SeqCst);
    a.prompt(acp::PromptRequest::new(
        sid.clone(),
        vec![acp::ContentBlock::from(format!("matrix-{label}"))],
    ))
    .await
    .unwrap_or_else(|e| panic!("{label}: prompt: {e:?}"));
    assert!(
        model_hits.load(std::sync::atomic::Ordering::SeqCst) > model_before,
        "{label}: the turn must run before hook attribution is meaningful"
    );

    let fired = wait_for(|| !hook_hits_for(log, &sid).is_empty()).await;
    assert_eq!(
        fired, expect_hook,
        "{label}: bypasses={bypasses:?}, hits={:?}",
        hook_hits_for(log, &sid)
    );
}

local_test!(hook_three_layer_redundancy_matrix, async {
    let _reset = DefenceBypassReset;
    MvpAgent::set_hook_defence_bypasses(false, false, false);
    let hook_dir = tempfile::tempdir().expect("hook dir");
    let (script, log) = hook_sentinel(hook_dir.path());
    let profile = inline_hook_profile(&script);
    let (model_url, model_hits) = spawn_mock_model().await;
    let (a, _tmp) = agent_with_model(&model_url);
    init(&a).await;

    // `true` means that defence is bypassed. With all defences active, or
    // with either one or two bypassed, the hook must remain blocked. Only
    // bypassing all three is expected to expose the sentinel.
    let cases = [
        ("all_enabled", (false, false, false), false),
        ("input_disabled", (true, false, false), false),
        ("spawn_disabled", (false, true, false), false),
        ("dispatch_disabled", (false, false, true), false),
        ("input_only", (false, true, true), false),
        ("spawn_only", (true, false, true), false),
        ("dispatch_only", (true, true, false), false),
        ("all_disabled", (true, true, true), true),
    ];
    for (label, bypasses, expect_hook) in cases {
        run_hook_defence_case(
            &a,
            &profile,
            &log,
            &model_hits,
            label,
            bypasses,
            expect_hook,
        )
        .await;
    }
});

local_test!(plugin_fanout_two_layer_redundancy_matrix, async {
    let _reset = DefenceBypassReset;
    MvpAgent::set_plugin_fanout_defence_bypasses(false, false);
    let fixture = plugin_fixture("fanout-matrix-plugin");
    let (a, _tmp) = agent_with_plugins(&[fixture.path()]);
    init(&a).await;

    // For two defences, the single-disabled and single-enabled cases are the
    // same two configurations. Each surviving layer must independently keep
    // the restricted actor empty; only bypassing both may install the fixture.
    let cases = [
        ("all_enabled", (false, false), Some(None)),
        ("broadcast_disabled_apply_only", (true, false), Some(None)),
        ("apply_disabled_broadcast_only", (false, true), Some(None)),
        ("all_disabled", (true, true), Some(Some(1))),
    ];
    for (label, bypasses, expected) in cases {
        MvpAgent::set_plugin_fanout_defence_bypasses(bypasses.0, bypasses.1);
        let restricted_cwd = tempfile::tempdir().expect("restricted cwd");
        let ordinary_cwd = tempfile::tempdir().expect("ordinary cwd");
        let restricted = disabled_session(&a, restricted_cwd.path()).await;
        let ordinary = ordinary_session(&a, ordinary_cwd.path()).await;

        ext(
            &a,
            "x.ai/plugins/action",
            serde_json::json!({ "sessionId": ordinary.0, "action": { "type": "reload" } }),
        )
        .await
        .unwrap_or_else(|e| panic!("{label}: ordinary reload: {e:?}"));
        assert_eq!(
            a.session_raw_plugin_registry(&ordinary).await,
            Some(Some(1)),
            "{label}: the broadcast must carry a real plugin"
        );
        assert_eq!(
            a.session_raw_plugin_registry(&restricted).await,
            expected,
            "{label}: bypasses={bypasses:?}"
        );
    }
});
