"""Curation primitives for the person identity graph.

The reconcile pipeline (email → slack → fuzzy name → entity-bridge →
cross-source → transitive closure → self-identity) makes mistakes over
time.  This module exposes the operations a human uses to fix them:

* :meth:`PersonCurator.inspect` — surface every ``SAME_AS`` edge
  incident on the cluster a Person belongs to, with provenance.
* :meth:`PersonCurator.split` — break a single bad ``SAME_AS`` edge and
  install a ``NEVER_SAME_AS`` block so the next reconcile pass does not
  recreate it.
* :meth:`PersonCurator.confirm` — lock a good merge by writing a
  ``user_confirmed`` ``SAME_AS`` edge.  Reconcile steps treat
  ``user_confirmed`` as ground truth and never overwrite it (because all
  reconcile steps create-only via ``MERGE`` + a ``NOT (a)-[:SAME_AS]-(b)``
  guard).
* :meth:`PersonCurator.merge` — manual same-as for cases the automated
  chain missed (different names, no shared email/slack id).

Identifier parsing is centralised in :func:`parse_identifier` so the CLI
and MCP layers accept the same shapes.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Iterable

from neo4j import Driver

from worker.curation.audit import AuditLog

logger = logging.getLogger(__name__)


class CurationError(Exception):
    """Raised when an identifier cannot be resolved or a guard rejects an op."""


# ---------------------------------------------------------------------------
# Identifier parsing
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PersonRef:
    """A parsed reference to a Person/Entity node.

    Exactly one of ``email``, ``slack`` (``(team_id, user_id)``), or
    ``name`` is populated — selected by :func:`parse_identifier`.
    """

    raw: str
    email: str | None = None
    slack: tuple[str, str] | None = None
    name: str | None = None

    @property
    def kind(self) -> str:
        if self.email is not None:
            return "email"
        if self.slack is not None:
            return "slack"
        return "name"


_SLACK_RE = re.compile(r"^slack(?:-user)?:([^/]+)/(.+)$")


def parse_identifier(raw: str) -> PersonRef:
    """Classify ``raw`` as an email, slack id, or name.

    Recognised shapes:

    ``alice@example.com``
        Email — anything containing ``@``.

    ``slack:T123/U456`` or ``slack-user:T123/U456``
        Slack identity.  Both prefixes are accepted; ``slack-user:`` is
        the legacy form used inside parser-emitted source ids.

    Anything else is treated as a literal Person/Entity ``name``.

    Raises :class:`CurationError` for an empty input.
    """
    if not raw or not raw.strip():
        raise CurationError("identifier must be a non-empty string")
    raw = raw.strip()

    if "@" in raw:
        return PersonRef(raw=raw, email=raw.lower())

    if raw.startswith(("slack:", "slack-user:")):
        m = _SLACK_RE.match(raw)
        if m is None:
            raise CurationError(
                f"invalid slack identifier: {raw!r} (expected slack:<team>/<user>)"
            )
        return PersonRef(raw=raw, slack=(m.group(1), m.group(2)))

    return PersonRef(raw=raw, name=raw)


# ---------------------------------------------------------------------------
# Cypher helpers
# ---------------------------------------------------------------------------


def _match_person_clause(var: str, ref: PersonRef) -> tuple[str, dict[str, Any]]:
    """Build a Cypher MATCH clause matching ``ref`` to a Person/Entity node.

    Returns ``(clause, params)`` where ``clause`` looks like
    ``MATCH (var) WHERE ...`` and ``params`` carries the Cypher parameters
    (named ``{var}_email`` / ``{var}_team`` / ``{var}_uid`` / ``{var}_name``).
    """
    if ref.email is not None:
        clause = (
            f"MATCH ({var}:Person) "
            f"WHERE toLower({var}.email) = ${var}_email"
        )
        return clause, {f"{var}_email": ref.email}

    if ref.slack is not None:
        team, uid = ref.slack
        clause = (
            f"MATCH ({var}:Person) "
            f"WHERE {var}.team_id = ${var}_team "
            f"AND {var}.slack_user_id = ${var}_uid"
        )
        return clause, {f"{var}_team": team, f"{var}_uid": uid}

    # Name match — Person or Entity, case-insensitive.
    clause = (
        f"MATCH ({var}) "
        f"WHERE ({var}:Person OR {var}:Entity) "
        f"AND toLower({var}.name) = toLower(${var}_name)"
    )
    return clause, {f"{var}_name": ref.name}


# ---------------------------------------------------------------------------
# Result shapes
# ---------------------------------------------------------------------------


@dataclass
class SameAsEdge:
    other_label: str  # "Person" or "Entity"
    other_name: str | None
    other_email: str | None
    other_slack_user_id: str | None
    other_team_id: str | None
    match_type: str | None
    confidence: float | None
    cross_source: bool | None
    direction: str  # "out" or "in" relative to the focal node


@dataclass
class FocalNode:
    label: str  # "Person" or "Entity"
    name: str | None
    email: str | None
    slack_user_id: str | None
    team_id: str | None
    is_self: bool


@dataclass
class InspectResult:
    focal: FocalNode | None
    same_as: list[SameAsEdge] = field(default_factory=list)
    never_same_as: list[SameAsEdge] = field(default_factory=list)


@dataclass
class CurationActionResult:
    action: str
    detail: dict[str, Any]


# ---------------------------------------------------------------------------
# PersonCurator
# ---------------------------------------------------------------------------


class PersonCurator:
    """Run curation operations against a Neo4j graph.

    The driver is owned by the caller — the curator does not open or close
    it.  Optional :class:`AuditLog` records every mutating action.
    """

    def __init__(
        self,
        driver: Driver,
        *,
        audit: AuditLog | None = None,
        actor: str = "user",
    ) -> None:
        self._driver = driver
        self._audit = audit
        self._actor = actor

    # -- inspect -----------------------------------------------------------

    def inspect(self, identifier: str) -> InspectResult:
        """Return the focal node, all incident SAME_AS, and NEVER_SAME_AS edges.

        The cluster a Person belongs to is the connected component reachable
        through ``SAME_AS``; ``inspect`` returns the *direct* edges only,
        not the full closure.  Operators can chain ``inspect`` calls on
        neighbours to walk the cluster.
        """
        ref = parse_identifier(identifier)
        match_clause, params = _match_person_clause("p", ref)

        with self._driver.session() as session:
            focal_record = session.run(
                f"""
                {match_clause}
                RETURN labels(p) AS labels,
                       p.name AS name,
                       p.email AS email,
                       p.slack_user_id AS slack_user_id,
                       p.team_id AS team_id,
                       coalesce(p.is_self, false) AS is_self
                LIMIT 1
                """,
                **params,
            ).single()

            if focal_record is None:
                raise CurationError(
                    f"no Person/Entity matches identifier {identifier!r}"
                )

            focal = FocalNode(
                label=_primary_label(focal_record["labels"]),
                name=focal_record["name"],
                email=focal_record["email"],
                slack_user_id=focal_record["slack_user_id"],
                team_id=focal_record["team_id"],
                is_self=bool(focal_record["is_self"]),
            )

            edge_records = session.run(
                f"""
                {match_clause}
                MATCH (p)-[r:SAME_AS|NEVER_SAME_AS]-(o)
                RETURN type(r) AS rtype,
                       startNode(r) = p AS outgoing,
                       labels(o) AS olabels,
                       o.name AS oname,
                       o.email AS oemail,
                       o.slack_user_id AS oslack,
                       o.team_id AS oteam,
                       r.match_type AS match_type,
                       r.confidence AS confidence,
                       r.cross_source AS cross_source
                """,
                **params,
            )
            same_as: list[SameAsEdge] = []
            never: list[SameAsEdge] = []
            for rec in edge_records:
                edge = SameAsEdge(
                    other_label=_primary_label(rec["olabels"]),
                    other_name=rec["oname"],
                    other_email=rec["oemail"],
                    other_slack_user_id=rec["oslack"],
                    other_team_id=rec["oteam"],
                    match_type=rec["match_type"],
                    confidence=(
                        float(rec["confidence"])
                        if rec["confidence"] is not None
                        else None
                    ),
                    cross_source=(
                        bool(rec["cross_source"])
                        if rec["cross_source"] is not None
                        else None
                    ),
                    direction="out" if rec["outgoing"] else "in",
                )
                if rec["rtype"] == "SAME_AS":
                    same_as.append(edge)
                else:
                    never.append(edge)

        return InspectResult(focal=focal, same_as=same_as, never_same_as=never)

    # -- split -------------------------------------------------------------

    def split(self, identifier: str, member: str) -> CurationActionResult:
        """Break the SAME_AS edge between ``identifier`` and ``member``.

        Adds a ``NEVER_SAME_AS`` block edge so future reconcile passes do
        not recreate the merge.  Both endpoints are matched against the
        Person/Entity graph.

        Raises :class:`CurationError` if either endpoint is missing.
        Returns the count of removed SAME_AS edges (0 if there was no
        direct edge, e.g. transitive cluster membership).
        """
        ref_a = parse_identifier(identifier)
        ref_b = parse_identifier(member)
        clause_a, params_a = _match_person_clause("a", ref_a)
        clause_b, params_b = _match_person_clause("b", ref_b)
        params = {**params_a, **params_b}

        with self._driver.session() as session:
            # Verify both endpoints exist (better error than silent no-op).
            for var, clause, p in (
                ("a", clause_a, params_a),
                ("b", clause_b, params_b),
            ):
                rec = session.run(
                    f"{clause} RETURN count({var}) AS cnt",
                    **p,
                ).single()
                if rec is None or rec["cnt"] == 0:
                    raw = identifier if var == "a" else member
                    raise CurationError(
                        f"no Person/Entity matches identifier {raw!r}"
                    )

            removed_rec = session.run(
                f"""
                {clause_a}
                {clause_b}
                MATCH (a)-[r:SAME_AS]-(b)
                DELETE r
                RETURN count(r) AS removed
                """,
                **params,
            ).single()
            removed = int(removed_rec["removed"]) if removed_rec else 0

            block_rec = session.run(
                f"""
                {clause_a}
                {clause_b}
                MERGE (a)-[r:NEVER_SAME_AS]->(b)
                ON CREATE SET r.created_at = timestamp(),
                              r.actor = $actor
                RETURN r.created_at AS created_at
                """,
                actor=self._actor,
                **params,
            ).single()

        detail: dict[str, Any] = {
            "identifier": identifier,
            "member": member,
            "same_as_removed": removed,
            "never_same_as_created_at": (
                int(block_rec["created_at"]) if block_rec else None
            ),
        }
        self._log("split", args={"identifier": identifier, "member": member}, result=detail)
        return CurationActionResult(action="split", detail=detail)

    # -- confirm -----------------------------------------------------------

    def confirm(self, a: str, b: str) -> CurationActionResult:
        """Lock the SAME_AS edge between ``a`` and ``b`` as user-confirmed.

        Writes (or updates) a ``SAME_AS`` edge with ``match_type =
        'user_confirmed'`` and ``confidence = 1.0``.  All reconcile steps
        already use ``MERGE`` with a ``NOT (a)-[:SAME_AS]-(b)`` guard, so
        an existing edge is never overwritten — confirming an edge is the
        one place we deliberately *do* mutate edge metadata.
        """
        return self._upsert_user_confirmed(a, b, action="confirm")

    # -- merge -------------------------------------------------------------

    def merge(self, a: str, b: str) -> CurationActionResult:
        """Manually create a ``user_confirmed`` SAME_AS edge between ``a`` and ``b``.

        Functionally identical to :meth:`confirm` — kept as a separate
        verb for the cases the issue distinguishes (confirming an
        automation's guess vs. asserting a merge the chain missed).
        """
        return self._upsert_user_confirmed(a, b, action="merge")

    def _upsert_user_confirmed(
        self, a: str, b: str, *, action: str
    ) -> CurationActionResult:
        ref_a = parse_identifier(a)
        ref_b = parse_identifier(b)
        clause_a, params_a = _match_person_clause("a", ref_a)
        clause_b, params_b = _match_person_clause("b", ref_b)
        params = {**params_a, **params_b}

        with self._driver.session() as session:
            for var, clause, p, raw in (
                ("a", clause_a, params_a, a),
                ("b", clause_b, params_b, b),
            ):
                rec = session.run(
                    f"{clause} RETURN count({var}) AS cnt",
                    **p,
                ).single()
                if rec is None or rec["cnt"] == 0:
                    raise CurationError(
                        f"no Person/Entity matches identifier {raw!r}"
                    )

            blocked_rec = session.run(
                f"""
                {clause_a}
                {clause_b}
                RETURN exists((a)-[:NEVER_SAME_AS]-(b)) AS blocked,
                       id(a) = id(b) AS same_node
                """,
                **params,
            ).single()
            if blocked_rec and blocked_rec["same_node"]:
                raise CurationError(
                    f"cannot {action} a node with itself: {a!r}"
                )
            if blocked_rec and blocked_rec["blocked"]:
                raise CurationError(
                    f"cannot {action}: {a!r} and {b!r} are blocked by NEVER_SAME_AS — "
                    f"remove the block first"
                )

            edge_rec = session.run(
                f"""
                {clause_a}
                {clause_b}
                MERGE (a)-[r:SAME_AS]->(b)
                SET r.match_type = 'user_confirmed',
                    r.confidence = 1.0,
                    r.cross_source = true,
                    r.confirmed_at = timestamp(),
                    r.actor = $actor
                RETURN r.confirmed_at AS confirmed_at
                """,
                actor=self._actor,
                **params,
            ).single()

        detail: dict[str, Any] = {
            "a": a,
            "b": b,
            "match_type": "user_confirmed",
            "confidence": 1.0,
            "confirmed_at": (
                int(edge_rec["confirmed_at"]) if edge_rec else None
            ),
        }
        self._log(action, args={"a": a, "b": b}, result=detail)
        return CurationActionResult(action=action, detail=detail)

    # -- internal ----------------------------------------------------------

    def _log(self, action: str, *, args: dict[str, Any], result: dict[str, Any]) -> None:
        if self._audit is None:
            return
        self._audit.append(action, args=args, result=result, actor=self._actor)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _primary_label(labels: Iterable[str]) -> str:
    """Pick the most informative label for display.

    A Neo4j node may carry multiple labels; we prefer ``Person`` and fall
    back to ``Entity``, then to whatever the first label is.
    """
    labels_list = list(labels)
    for preferred in ("Person", "Entity"):
        if preferred in labels_list:
            return preferred
    return labels_list[0] if labels_list else "Node"
