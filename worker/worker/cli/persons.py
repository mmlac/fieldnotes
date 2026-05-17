"""CLI handlers for ``fieldnotes persons {inspect,split,confirm,merge}``."""

from __future__ import annotations

import json
import sys
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Iterator

from neo4j import Driver

from worker.config import load_config
from worker.neo4j_driver import build_driver
from worker.curation import (
    AuditLog,
    CurationError,
    PersonCurator,
)
from worker.curation.persons import InspectResult as _InspectResult


@contextmanager
def _open_driver(config_path: Path | None) -> Iterator[tuple[Driver, str]]:
    """Open a Neo4j driver from ``config.toml`` and yield ``(driver, data_dir)``."""
    cfg = load_config(config_path)
    cfg.neo4j.validate()
    driver = build_driver(cfg.neo4j.uri, cfg.neo4j.user, cfg.neo4j.password)
    try:
        yield driver, cfg.core.data_dir
    finally:
        driver.close()


def _make_curator(driver: Driver, data_dir: str) -> PersonCurator:
    audit = AuditLog.from_data_dir(data_dir)
    return PersonCurator(driver, audit=audit)


# ---------------------------------------------------------------------------
# inspect
# ---------------------------------------------------------------------------


def run_inspect(
    *,
    identifier: str,
    config_path: Path | None = None,
    json_output: bool = False,
) -> int:
    try:
        with _open_driver(config_path) as (driver, data_dir):
            curator = _make_curator(driver, data_dir)
            result = curator.inspect(identifier)
    except CurationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if json_output:
        print(json.dumps(_inspect_to_dict(result), indent=2, sort_keys=True))
        return 0

    _print_inspect(result)
    return 0


def _inspect_to_dict(result: _InspectResult) -> dict:
    return {
        "focal": asdict(result.focal) if result.focal else None,
        "same_as": [asdict(e) for e in result.same_as],
        "never_same_as": [asdict(e) for e in result.never_same_as],
    }


def _print_inspect(result: _InspectResult) -> None:
    focal = result.focal
    if focal is None:
        print("No matching person found.")
        return

    flag = " [self]" if focal.is_self else ""
    print(f"{focal.label}: {focal.name or '<no name>'}{flag}")
    if focal.email:
        print(f"  email: {focal.email}")
    if focal.slack_user_id and focal.team_id:
        print(f"  slack: {focal.team_id}/{focal.slack_user_id}")

    if result.same_as:
        print(f"\nSAME_AS edges ({len(result.same_as)}):")
        for edge in result.same_as:
            _print_edge(edge)
    else:
        print("\nSAME_AS edges: none")

    if result.never_same_as:
        print(f"\nNEVER_SAME_AS blocks ({len(result.never_same_as)}):")
        for edge in result.never_same_as:
            _print_edge(edge)


def _print_edge(edge) -> None:
    name = edge.other_name or "<no name>"
    arrow = "→" if edge.direction == "out" else "←"
    parts = [f"  {arrow} {edge.other_label}: {name}"]
    if edge.other_email:
        parts.append(f"<{edge.other_email}>")
    if edge.other_slack_user_id and edge.other_team_id:
        parts.append(f"slack:{edge.other_team_id}/{edge.other_slack_user_id}")
    print(" ".join(parts))
    meta_bits = []
    if edge.match_type:
        meta_bits.append(f"match_type={edge.match_type}")
    if edge.confidence is not None:
        meta_bits.append(f"confidence={edge.confidence:.2f}")
    if edge.cross_source:
        meta_bits.append("cross_source=true")
    if meta_bits:
        print("      " + ", ".join(meta_bits))


# ---------------------------------------------------------------------------
# split / confirm / merge
# ---------------------------------------------------------------------------


def run_split(
    *,
    identifier: str,
    member: str,
    config_path: Path | None = None,
    json_output: bool = False,
) -> int:
    return _run_action(
        action="split",
        config_path=config_path,
        json_output=json_output,
        invoke=lambda c: c.split(identifier, member),
    )


def run_confirm(
    *,
    a: str,
    b: str,
    config_path: Path | None = None,
    json_output: bool = False,
) -> int:
    return _run_action(
        action="confirm",
        config_path=config_path,
        json_output=json_output,
        invoke=lambda c: c.confirm(a, b),
    )


def run_merge(
    *,
    a: str,
    b: str,
    config_path: Path | None = None,
    json_output: bool = False,
) -> int:
    return _run_action(
        action="merge",
        config_path=config_path,
        json_output=json_output,
        invoke=lambda c: c.merge(a, b),
    )


def _run_action(
    *,
    action: str,
    config_path: Path | None,
    json_output: bool,
    invoke,
) -> int:
    try:
        with _open_driver(config_path) as (driver, data_dir):
            curator = _make_curator(driver, data_dir)
            result = invoke(curator)
    except CurationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if json_output:
        print(
            json.dumps(
                {"action": result.action, "detail": result.detail},
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    print(f"{action}: ok")
    for k, v in result.detail.items():
        print(f"  {k}: {v}")
    return 0
