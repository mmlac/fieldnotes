"""``fieldnotes doctor`` — pre-flight checks for a healthy setup."""

from __future__ import annotations

import shutil
from pathlib import Path

from worker.config import DEFAULT_CONFIG_PATH, SlackSourceConfig, load_config


def _ok(msg: str) -> None:
    print(f"  \u2713 {msg}")


def _warn(msg: str) -> None:
    print(f"  ! {msg}")


def _fail(msg: str) -> None:
    print(f"  \u2717 {msg}")


def check_slack_auth(token_path: Path | None = None) -> int:
    """Check Slack auth status and print a result line.

    Returns 0 on OK / not-configured, 1 if the token is rejected (reauth
    required) — those are surfaced as errors so ``fieldnotes doctor``
    exits non-zero and the user knows to re-run the install flow.
    """
    from slack_sdk import WebClient
    from slack_sdk.errors import SlackApiError

    from worker.sources.slack_auth import (
        DEFAULT_TOKEN_PATH,
        REAUTH_ERRORS,
        _load_token,
        _slack_error_code,
    )

    path = token_path or DEFAULT_TOKEN_PATH
    if not path.exists():
        _warn("Slack auth: not configured (run install flow)")
        return 0
    try:
        token = _load_token(path)
    except Exception as exc:
        _fail(f"Slack auth: token unreadable ({exc}) — re-run install flow")
        return 1
    if token is None:
        _warn("Slack auth: not configured (run install flow)")
        return 0
    try:
        WebClient(token=token.bot_token).auth_test()
    except SlackApiError as exc:
        err = _slack_error_code(exc)
        if err in REAUTH_ERRORS:
            _fail(f"Slack auth: reauth required ({err}) — re-run install flow")
        else:
            _fail(f"Slack auth: API error ({err or exc})")
        return 1
    except Exception as exc:
        _fail(f"Slack auth: error ({exc})")
        return 1
    _ok("Slack auth: OK")
    return 0


def check_slack(slack_cfg: SlackSourceConfig) -> int:
    """Run Slack source pre-flight checks. Returns the number of errors."""
    if not slack_cfg.enabled:
        _ok("Slack disabled (set [sources.slack] enabled = true to enable)")
        return 0

    secrets_path = Path(slack_cfg.client_secrets_path).expanduser()
    if not secrets_path.is_file():
        _fail(f"Slack client_secrets_path missing: {secrets_path}")
        return 1

    _ok(f"Slack client secrets present ({secrets_path})")

    try:
        import worker.sources.slack_auth  # noqa: F401
    except ImportError:
        _warn(
            "Slack auth module not available "
            "(worker.sources.slack_auth) — skipping auth probe"
        )
        return 0

    return check_slack_auth()


def doctor(config_path: Path | None = None) -> int:
    """Run pre-flight checks and print results.  Returns 0 if all pass."""
    path = config_path or DEFAULT_CONFIG_PATH
    errors = 0

    # ── 1. Config file ──────────────────────────────────────────────
    print("Config")
    if not path.exists():
        _fail(f"Config file not found: {path}")
        print("\n  Run 'fieldnotes init' to create one.\n")
        return 1
    _ok(f"Config file exists ({path})")

    try:
        cfg = load_config(path)
        _ok("Config parses without errors")
    except Exception as exc:
        _fail(f"Config parse error: {exc}")
        return 1

    # Print any validation warnings
    warnings = cfg.validate()
    for w in warnings:
        _warn(w)

    # ── 2. Model provider chain ─────────────────────────────────────
    print("\nModels")
    for role, alias in cfg.roles.mapping.items():
        if alias not in cfg.models:
            _fail(f"Role {role!r} → model {alias!r} (not defined)")
            errors += 1
        else:
            m = cfg.models[alias]
            if m.provider not in cfg.providers:
                _fail(
                    f"Role {role!r} → model {alias!r} → provider "
                    f"{m.provider!r} (not defined)"
                )
                errors += 1
            else:
                _ok(f"Role {role!r} → {alias!r} ({m.model} via {m.provider})")

    # ── 3. Ollama reachability ──────────────────────────────────────
    print("\nProviders")
    for name, prov in cfg.providers.items():
        if prov.type == "ollama":
            base_url = prov.settings.get("base_url", "http://localhost:11434")
            # Validate URL scheme to prevent SSRF.
            from urllib.parse import urlparse

            parsed = urlparse(base_url)
            if parsed.scheme not in ("http", "https"):
                _fail(f"Ollama ({name}) invalid URL scheme: {base_url}")
                errors += 1
                continue
            try:
                import urllib.request

                req = urllib.request.Request(f"{base_url}/api/tags", method="GET")
                with urllib.request.urlopen(req, timeout=5) as resp:
                    if resp.status == 200:
                        import json

                        data = json.loads(resp.read())
                        available = [m["name"] for m in data.get("models", [])]
                        _ok(f"Ollama ({name}) reachable at {base_url}")
                        # Check that configured models are pulled
                        for alias, mcfg in cfg.models.items():
                            if mcfg.provider == name:
                                # Ollama names may or may not include :latest
                                model_name = mcfg.model
                                matched = any(
                                    a == model_name
                                    or a == f"{model_name}:latest"
                                    or a.startswith(f"{model_name}:")
                                    for a in available
                                )
                                if matched:
                                    _ok(f"  Model {model_name!r} available")
                                else:
                                    _warn(
                                        f"  Model {model_name!r} not found "
                                        f"— run: ollama pull {model_name}"
                                    )
                    else:
                        _fail(f"Ollama ({name}) returned HTTP {resp.status}")
                        errors += 1
            except Exception as exc:
                _fail(f"Ollama ({name}) unreachable at {base_url}: {exc}")
                errors += 1
        elif prov.type == "openai":
            api_key = prov.settings.get("api_key") or ""
            import os

            if not api_key:
                api_key = os.environ.get("OPENAI_API_KEY", "")
            if api_key:
                _ok(f"OpenAI ({name}) API key configured")
            else:
                _warn(
                    f"OpenAI ({name}) no API key set "
                    "(set OPENAI_API_KEY or api_key in config)"
                )
        elif prov.type == "anthropic":
            api_key = prov.settings.get("api_key") or ""
            import os

            if not api_key:
                api_key = os.environ.get("ANTHROPIC_API_KEY", "")
            if api_key:
                _ok(f"Anthropic ({name}) API key configured")
            else:
                _warn(
                    f"Anthropic ({name}) no API key set "
                    "(set ANTHROPIC_API_KEY or api_key in config)"
                )
        else:
            _ok(f"Provider {name!r} (type={prov.type})")

    # ── 4. Neo4j ────────────────────────────────────────────────────
    print("\nInfrastructure")
    try:
        from neo4j import GraphDatabase

        driver = GraphDatabase.driver(
            cfg.neo4j.uri,
            auth=(cfg.neo4j.user, cfg.neo4j.password),
        )
        try:
            driver.verify_connectivity()
            _ok(f"Neo4j reachable ({cfg.neo4j.uri})")
        finally:
            driver.close()
    except Exception as exc:
        _fail(f"Neo4j unreachable ({cfg.neo4j.uri}): {exc}")
        errors += 1

    # ── 5. Qdrant ───────────────────────────────────────────────────
    try:
        from qdrant_client import QdrantClient

        client = QdrantClient(host=cfg.qdrant.host, port=cfg.qdrant.port)
        try:
            client.get_collections()
            _ok(f"Qdrant reachable ({cfg.qdrant.host}:{cfg.qdrant.port})")
        finally:
            client.close()
    except Exception as exc:
        _fail(f"Qdrant unreachable ({cfg.qdrant.host}:{cfg.qdrant.port}): {exc}")
        errors += 1

    # ── 6. Source paths ─────────────────────────────────────────────
    print("\nSources")
    if not cfg.sources:
        _warn("No sources configured")
    for name, src in cfg.sources.items():
        settings = src.settings
        # Check common path settings
        path_keys = [
            "watch_paths",
            "vault_paths",
            "repo_roots",
            "scan_dirs",
        ]
        found_any = False
        for key in path_keys:
            if key not in settings:
                continue
            found_any = True
            val = settings[key]
            paths = val if isinstance(val, list) else [val]
            for p in paths:
                expanded = Path(p).expanduser()
                if expanded.exists():
                    _ok(f"{name}.{key}: {expanded}")
                else:
                    _warn(f"{name}.{key}: {expanded} (does not exist)")
        if not found_any:
            _ok(f"{name} configured")

    # ── 6b. Slack source ────────────────────────────────────────────
    print("\nSlack")
    errors += check_slack(cfg.slack)

    # ── 7. Reranker ──────────────────────────────────────────────────
    print("\nReranker")
    if not cfg.reranker.enabled:
        _ok("Reranker disabled (set [reranker] enabled = true to enable)")
    elif "rerank" not in cfg.roles.mapping:
        _warn(
            "Reranker enabled but no model bound to the 'rerank' role — "
            "vector results will not be reranked"
        )
    else:
        rerank_alias = cfg.roles.mapping["rerank"]
        try:
            from worker.models.resolver import ModelRegistry

            registry = ModelRegistry(cfg)
            model = registry.for_role("rerank")
            try:
                # Smoke test: one tiny pair.  This forces the cross-encoder to
                # download (if missing) and run a single inference; for an
                # LLM-backed provider this raises NotImplementedError, which
                # we surface as a configuration error.
                model.rerank("ping", ["pong"])
                _ok(
                    f"Reranker model {rerank_alias!r} ({model.model}) loads and "
                    f"scores a smoke pair"
                )
            except NotImplementedError:
                _fail(
                    f"Reranker model {rerank_alias!r} → provider "
                    f"{model.provider.provider_type!r} does not implement rerank()"
                )
                errors += 1
        except Exception as exc:
            _fail(f"Reranker smoke test failed: {exc}")
            errors += 1

    # ── 8. Tools ─────────────────────────────────────────────────────
    print("\nTools")
    if shutil.which("ollama"):
        _ok("ollama binary on PATH")
    else:
        _warn("ollama not found on PATH")
    if shutil.which("docker"):
        _ok("docker binary on PATH")
    else:
        _warn("docker not found on PATH")

    # ── Summary ──────────────────────────────────────────────────────
    print()
    if errors:
        print(f"{errors} check(s) failed. Fix the issues above and re-run.")
        return 1
    print("All checks passed.")
    return 0
