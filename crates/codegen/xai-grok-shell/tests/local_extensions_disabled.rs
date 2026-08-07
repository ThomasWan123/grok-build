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

const KEY: &str = "x.ai/localExtensionsDisabled";
const APPLIED: &str = "localExtensionsDisabledApplied";

// ---------------------------------------------------------------------------
// fixtures
// ---------------------------------------------------------------------------

/// A live agent with just enough auth to get past the `session/new` gate.
///
/// `XAI_API_KEY` is what makes `initialize` select a default auth method; the
/// key is never used for a request because no test here runs a turn.
fn agent() -> (MvpAgent, tempfile::TempDir) {
    let temp = tempfile::tempdir().expect("temp dir");
    unsafe {
        std::env::set_var("XAI_API_KEY", "test-key");
        std::env::set_var("GROK_HOME", temp.path());
    }
    let auth = std::sync::Arc::new(xai_grok_shell::auth::AuthManager::new(
        temp.path(),
        xai_grok_shell::auth::GrokComConfig::default(),
    ));
    let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
    let gateway = xai_acp_lib::AcpAgentGatewaySender::new(tx);
    let cfg = xai_grok_shell::agent::config::Config::default();
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
    let (a, tmp) = agent();
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

    // The ordinary session still answers from its own registry rather than
    // being caught by the policy branch.
    let plugins = ext(
        &a,
        "x.ai/plugins/list",
        serde_json::json!({ "sessionId": ordinary.0 }),
    )
    .await
    .expect("plugins/list");
    assert!(plugins.get("plugins").is_some());

    // And its extension actions are not refused by the policy gate.
    let err = ext(
        &a,
        "x.ai/plugins/action",
        serde_json::json!({ "sessionId": ordinary.0, "action": { "type": "reload" } }),
    )
    .await
    .err();
    if let Some(e) = err {
        let data = err_data(&e);
        assert_ne!(data["code"], "local_extensions_disabled", "{data}");
    }
});

// ---------------------------------------------------------------------------
// T22 — the process-wide reload is conditional, not refused outright
// ---------------------------------------------------------------------------

local_test!(t22_global_reload_requires_an_eligible_session, async {
    let (a, tmp) = agent();
    init(&a).await;

    // Only a built-in-tools-only session exists: nobody could consume the
    // rebuild, so it is refused rather than run for no one.
    let disabled = disabled_session(&a, tmp.path()).await;
    let err = ext(&a, "x.ai/plugins/reload", serde_json::json!({}))
        .await
        .expect_err("reload must be refused when no session may use plugins");
    assert_eq!(err_data(&err)["reason"], "no_eligible_session");

    // Add an ordinary session and the same call succeeds — the endpoint stays
    // usable for the sessions it exists for.
    let ordinary_cwd = tempfile::tempdir().expect("cwd");
    let _ordinary = ordinary_session(&a, ordinary_cwd.path()).await;
    ext(&a, "x.ai/plugins/reload", serde_json::json!({}))
        .await
        .expect("reload must succeed once an eligible session exists");

    // …and the fan-out it triggers still does not reach the disabled session.
    let plugins = ext(
        &a,
        "x.ai/plugins/list",
        serde_json::json!({ "sessionId": disabled.0 }),
    )
    .await
    .expect("plugins/list");
    assert_eq!(plugins, serde_json::json!({ "plugins": [] }));
    let hooks = ext(
        &a,
        "x.ai/hooks/list",
        serde_json::json!({ "sessionId": disabled.0 }),
    )
    .await
    .expect("hooks/list");
    assert_eq!(hooks["hooks"], serde_json::json!([]));
});

// ---------------------------------------------------------------------------
// T3 — a reload driven by another session does not reach a disabled one
// ---------------------------------------------------------------------------

local_test!(t3_broadcast_does_not_reach_disabled_session, async {
    let (a, tmp) = agent();
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

    let plugins = ext(
        &a,
        "x.ai/plugins/list",
        serde_json::json!({ "sessionId": disabled.0 }),
    )
    .await
    .expect("plugins/list");
    assert_eq!(
        plugins,
        serde_json::json!({ "plugins": [] }),
        "a reload driven by another session must not reach this one"
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
