"""``fieldnotes doctor`` — pre-flight checks for a healthy setup."""

from __future__ import annotations

import shutil
from pathlib import Path

from worker.config import DEFAULT_CONFIG_PATH, load_config


def _ok(msg: str) -> None:
    print(f"  \u2713 {msg}")


def _warn(msg: str) -> None:
    print(f"  ! {msg}")


def _fail(msg: str) -> None:
    print(f"  \u2717 {msg}")


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

    # ── 7. Tools ─────────────────────────────────────────────────────
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
