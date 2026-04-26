# Entity Resolution: Person Reconciliation

Fieldnotes ingests Person nodes from many source types (Gmail, Calendar,
Obsidian, Slack, Git). The reconciliation chain runs after every batch and
collapses duplicates so a single human is one node in the graph.

The chain runs in this order — each step builds on the work of the
previous one:

1. **Email-based merge** — `Writer.reconcile_persons()`
   Groups Person nodes by `toLower(trim(p.email))`, sets the longest name
   variant as canonical, and creates `SAME_AS` edges between every node in
   each group. All parsers normalise emails through
   `parsers.base.canonicalize_email()` before writing, so e.g.
   `@googlemail.com` collapses with `@gmail.com`.

2. **Slack identity merge** — `Writer.reconcile_persons_by_slack_user()`
   Groups Person nodes by `(team_id, slack_user_id)` and creates
   `SAME_AS` edges (`match_type = 'slack_user_id'`) between members of
   each group with size > 1.

   This step closes a gap that the email-based merge cannot fix: a Slack
   user's profile email may be unknown when their first message is
   ingested. The parser then keys the Person on
   `slack-user:{team}/{user_id}` (no email property). When a later
   message exposes the user's email, the parser emits a separate,
   email-keyed Person — which now also carries `slack_user_id` and
   `team_id` so this step can link the two nodes.

3. **Fuzzy name merge** — `Writer.reconcile_persons_by_name()`
   RapidFuzz `token_sort_ratio` over Person names with a conservative
   length-dependent threshold (≥ 95 for short names, 88 for longer).

4. **Entity → Person bridge** — `Writer.bridge_entity_persons()`
   Bridges LLM-extracted `Entity {type: 'Person'}` nodes to structured
   Person nodes via fuzzy name match.

5. **Cross-source resolution + transitive closure** —
   `Writer.resolve_cross_source_entities()` followed by
   `Writer.close_same_as_transitive()`.

6. **Self-identity** — `Writer.reconcile_self_person(cfg.me)`
   Runs only when the top-level `[me]` block is configured. For each
   email in `cfg.me.emails` (already canonicalised by the config loader)
   it `MERGE`s a Person keyed on `email`, sets `is_self = true`, and
   propagates a survivor `display_name`. It then creates `SAME_AS` edges
   between every pair of self-Persons (`match_type = 'self_identity'`,
   `confidence = 1.0`, `cross_source = true`).

   The display name is `cfg.me.name` when set, otherwise the longest
   existing `name` among the matched Persons. A single email is a no-op
   for `SAME_AS` edges (the lone Person is still flagged `is_self`).

   This step runs last, *after* the email-based merge has already
   unified email-keyed Persons across sources. The `[me]` step then
   groups *your* aliases — e.g. `me@personal.com` and `me@work.com` —
   that the prior steps cannot link because they have different emails.
   `is_self` is queryable directly: `MATCH (p:Person {is_self: true})`
   resolves to your self-Persons (or follow `SAME_AS` from any of them).

Steps 1, 2, 5, and 6 are deterministic and source-agnostic. Steps 3 and 4
are fuzzy and tuned conservatively to avoid false-positive merges.

## Why a separate Slack-identity step?

Email is the canonical merge key across sources. Slack is unique in
having a stable internal identifier (`slack_user_id`) that is *also*
authoritative within the workspace, but is not portable to other
sources. Treating Slack identity as a second deterministic merge key —
narrowly scoped to Person nodes that already carry it — keeps the
default email path unchanged while bridging the no-email-fallback case.

The step writes `SAME_AS` edges; it does not destructively merge node
properties. Downstream traversal queries that follow `SAME_AS` will see
the slack-user-keyed and email-keyed nodes as the same entity.
