# Cross-Source Entity Resolution & Deduplication

This document describes the current state of entity resolution in Fieldnotes and concrete techniques for improvement.

## Current State

### What Works Today

| Mechanism | Where | How |
|---|---|---|
| **Person merge by email** | `reconcile_persons()` in `writer.py` | Exact email match → longest name kept, `SAME_AS` edges created |
| **Email canonicalization** | `canonicalize_email()` in `parsers/base.py` | `strip().lower()` + `@googlemail.com` → `@gmail.com` across Gmail, Calendar, Obsidian, Git |
| **Entity fuzzy dedup** | `resolver.py` | 3-strategy cascade: exact → RapidFuzz `token_sort_ratio` → embedding cosine similarity |
| **Cross-source entity resolution** | `resolve_cross_source()` in `resolver.py` | Exact name → email-based Person match → fuzzy name match |
| **Graph hint merging** | `_write_graph_hint()` in `writer.py` | `MERGE` on custom merge keys (`email` for Person, `thread_id` for Thread) |

### Gaps

| Gap | Impact |
|---|---|
| **No fuzzy name reconciliation for Person nodes** | "M. Smith" and "Markus Smith" from different sources stay as separate Person nodes even when they're the same person |
| **Entity(Person) ↔ Person node disconnect** | LLM-extracted `Person` entities (Entity label, name only) never bridge to structured `Person` nodes (email, merge key) |
| **OmniFocus has no Person extraction** | Task owners, assignees, and people mentioned in task titles/notes are ignored |
| **No email extraction from text bodies** | Emails mentioned inline in documents don't create linkable Person nodes |
| **No transitive SAME_AS closure** | If A ↔ B and B ↔ C via `SAME_AS`, A ↔ C is never inferred |

> **All gaps above have been addressed** — see Improvement Techniques below.

## Improvement Techniques

### 1. Fuzzy Name Reconciliation for Person Nodes

**Status: Implemented**

Apply the resolver's existing fuzzy matching (RapidFuzz `token_sort_ratio` with length-aware thresholds) to Person nodes, not just Entity nodes. A `reconcile_persons_by_name()` pass runs after the email-based reconciliation and catches name variants like:

- "M. Smith" ↔ "Markus Smith"
- "Bob Jones" ↔ "Robert Jones" (when confidence is high enough)
- "alice" ↔ "Alice Smith" (blocked by length-aware threshold — short names require near-exact match)

Uses a high threshold (~95 for names ≥ 6 chars) to minimise false positives. Only creates `SAME_AS` edges — does not destructively merge nodes.

### 2. Git Author ↔ Gmail/Calendar Person Bridging

**Status: Implemented**

Git commit authors create Person nodes with `subject_merge_key="email"`. When the git email matches a Gmail/Calendar email, they merge automatically. The git parser now uses `canonicalize_email()` (same as Gmail, Calendar, and Obsidian parsers) so `@googlemail.com` addresses normalise to `@gmail.com` for consistent cross-source bridging. The remaining gap (different emails for the same person across git and Gmail) is addressed by technique #1 (fuzzy name matching).

### 3. LLM-Extracted Entity(Person) → Structured Person Bridging

**Status: Implemented**

The LLM extractor creates `Entity` nodes with `type=Person` and a `name` property. These are disconnected from the structured `Person` nodes created by parsers (which have `email` properties). A post-reconciliation pass queries Entity nodes where `type='Person'` and attempts fuzzy name matching against existing Person nodes. Matches create `SAME_AS` edges with confidence scores.

This bridges: "Alice Smith" extracted from a meeting transcript → `alice.smith@example.com` Person node from Gmail.

### 4. OmniFocus People Extraction

**Status: Implemented**

The OmniFocus parser now extracts Person nodes from three sources within task data:

1. **Email addresses** in task name/note → `Task -[MENTIONS]-> Person` with `email` merge key (confidence 1.0)
2. **@mentions** like `@Alice` or `@Bob Smith` in task name/note → `Task -[MENTIONS]-> Person` with name (confidence 0.9)
3. **Tags under `People/`** (e.g. `People/Alice`) → `Task -[MENTIONS]-> Person` with name from tag leaf (confidence 0.95)

@mentions require an uppercase first letter to avoid false positives (e.g. `@home`). Duplicate Person hints across all three extraction methods are deduplicated within each task.

### 5. Email Extraction from Text Bodies

**Status: Implemented**

A shared `extract_email_person_hints()` utility in `parsers/base.py` scans document text for email addresses using a regex pattern. The pipeline calls it automatically for every document with text content (in `_process_text()`, before chunking), creating `MENTIONS` Person hints with `email` merge key.

This creates linkable Person nodes from emails mentioned inline — in meeting notes, README contributor lists, code comments, etc. — without needing LLM inference. Emails are canonicalized via `canonicalize_email()` for consistent cross-source bridging.

### 6. Transitive SAME_AS Closure

**Status: Implemented**

A post-reconciliation step traverses variable-length `SAME_AS` paths (2–4 hops) to find node pairs that are transitively linked but lack a direct edge:

```cypher
MATCH (a)-[:SAME_AS*2..4]-(b)
WHERE id(a) < id(b)
  AND NOT (a)-[:SAME_AS]-(b)
WITH DISTINCT a, b
MERGE (a)-[r:SAME_AS]->(b)
SET r.match_type = 'transitive_closure',
    r.cross_source = true
RETURN count(r) AS cnt
```

This closes gaps where A(git) `SAME_AS` B(entity) and B(entity) `SAME_AS` C(gmail) but A and C aren't linked. Runs as the final step after all other reconciliation passes, bounded to 4 hops to keep query cost manageable.

## Architecture Notes

### Merge Keys by Node Type

| Node Type | Merge Key | ID Format |
|---|---|---|
| Person | `email` | `person:{email}` |
| Entity | `source_id` (default) | Various |
| Thread | `thread_id` | `gmail-thread:{id}` |
| Tag | `source_id` | `tag:{name}` or `omnifocus-tag:{path}` |
| File, Email, Commit, etc. | `source_id` | Source-specific |

### Confidence Thresholds

| Context | Threshold | Rationale |
|---|---|---|
| Fuzzy name match (short names, < 4 chars) | 100 (exact only) | Prevents AWS/AMS false positives |
| Fuzzy name match (4–5 chars) | 95 | Stricter for short names |
| Fuzzy name match (≥ 6 chars) | 88 | Standard fuzzy threshold |
| Embedding similarity fallback | 0.92 cosine | High bar — SAME_AS only, no destructive merge |
| Person name reconciliation | 95 | Conservative — false positives are worse than missed merges |
| Entity→Person bridging | 93 | Slightly higher than standard entity resolution |
| Cross-source confidence floor | 0.8 | Configurable via resolver |

### Resolution Order

1. `reconcile_persons()` — email-based Person dedup (exact)
2. `reconcile_persons_by_name()` — fuzzy Person name dedup (technique #1)
3. `bridge_entity_persons()` — Entity(Person) → Person linking (technique #3)
4. `resolve_cross_source_entities()` — general entity resolution across sources
5. `close_same_as_transitive()` — transitive closure over SAME_AS edges (technique #6)
