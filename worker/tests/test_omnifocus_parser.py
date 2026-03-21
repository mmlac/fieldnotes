"""Tests for the OmniFocus parser."""

from __future__ import annotations

from typing import Any


from worker.parsers.omnifocus import OmniFocusParser


# ── helpers ────────────────────────────────────────────────────────


def _make_event(
    task_id: str = "task-1",
    name: str = "Buy groceries",
    note: str = "",
    status: str = "active",
    flagged: bool = False,
    tags: list[str] | None = None,
    project: str = "",
    parent_task: str = "",
    parent_task_id: str = "",
    operation: str = "created",
    **kwargs: Any,
) -> dict[str, Any]:
    return {
        "id": "evt-uuid",
        "source_type": "omnifocus",
        "source_id": f"omnifocus://{task_id}",
        "operation": operation,
        "mime_type": "text/plain",
        "meta": {
            "id": task_id,
            "name": name,
            "note": note,
            "status": status,
            "flagged": flagged,
            "tags": tags or [],
            "project": project,
            "parent_task": parent_task,
            "parent_task_id": parent_task_id,
            "creation_date": kwargs.get("creation_date"),
            "modification_date": kwargs.get("modification_date"),
            "completion_date": kwargs.get("completion_date"),
            "due_date": kwargs.get("due_date"),
            "defer_date": kwargs.get("defer_date"),
        },
    }


# ── basic parsing ─────────────────────────────────────────────────


class TestOmniFocusParser:
    def setup_method(self) -> None:
        self.parser = OmniFocusParser()

    def test_source_type(self) -> None:
        assert self.parser.source_type == "omnifocus"

    def test_basic_task(self) -> None:
        event = _make_event(name="Test task", note="Some notes")
        docs = self.parser.parse(event)
        assert len(docs) == 1
        doc = docs[0]
        assert doc.source_type == "omnifocus"
        assert doc.source_id == "omnifocus://task-1"
        assert doc.node_label == "Task"
        assert doc.node_props["name"] == "Test task"
        assert doc.node_props["status"] == "active"
        assert "Test task" in doc.text
        assert "Some notes" in doc.text

    def test_deleted_operation(self) -> None:
        event = _make_event(operation="deleted")
        docs = self.parser.parse(event)
        assert len(docs) == 1
        assert docs[0].operation == "deleted"
        assert docs[0].text == ""

    def test_status_in_node_props(self) -> None:
        for status in ("active", "completed", "deferred", "dropped", "blocked"):
            event = _make_event(status=status)
            docs = self.parser.parse(event)
            assert docs[0].node_props["status"] == status

    def test_flagged_in_node_props(self) -> None:
        event = _make_event(flagged=True)
        docs = self.parser.parse(event)
        assert docs[0].node_props["flagged"] is True

    def test_dates_in_node_props(self) -> None:
        event = _make_event(
            due_date="2026-04-01T12:00:00Z",
            defer_date="2026-03-15T08:00:00Z",
            completion_date="2026-04-02T09:00:00Z",
            creation_date="2026-03-01T10:00:00Z",
            modification_date="2026-04-02T09:00:00Z",
        )
        docs = self.parser.parse(event)
        props = docs[0].node_props
        assert props["due_date"] == "2026-04-01T12:00:00Z"
        assert props["defer_date"] == "2026-03-15T08:00:00Z"
        assert props["completion_date"] == "2026-04-02T09:00:00Z"
        assert props["creation_date"] == "2026-03-01T10:00:00Z"
        assert props["modification_date"] == "2026-04-02T09:00:00Z"


# ── tag graph hints ──────────────────────────────────────────────


class TestTagGraphHints:
    def setup_method(self) -> None:
        self.parser = OmniFocusParser()

    def test_single_tag(self) -> None:
        event = _make_event(tags=["Work"])
        docs = self.parser.parse(event)
        hints = docs[0].graph_hints
        tag_hints = [h for h in hints if h.predicate == "TAGGED"]
        assert len(tag_hints) == 1
        assert tag_hints[0].object_id == "omnifocus-tag:Work"
        assert tag_hints[0].object_label == "Tag"
        assert tag_hints[0].object_props["name"] == "Work"
        assert tag_hints[0].confidence == 1.0

    def test_hierarchical_tag(self) -> None:
        """Tags like 'People/Boss' should be stored as-is."""
        event = _make_event(tags=["People/Boss"])
        docs = self.parser.parse(event)
        hints = docs[0].graph_hints
        tag_hints = [h for h in hints if h.predicate == "TAGGED"]
        assert len(tag_hints) == 1
        assert tag_hints[0].object_id == "omnifocus-tag:People/Boss"
        assert tag_hints[0].object_props["name"] == "People/Boss"

    def test_multiple_tags(self) -> None:
        event = _make_event(tags=["Work", "People/Boss", "Urgent"])
        docs = self.parser.parse(event)
        hints = docs[0].graph_hints
        tag_hints = [h for h in hints if h.predicate == "TAGGED"]
        assert len(tag_hints) == 3
        tag_ids = {h.object_id for h in tag_hints}
        assert tag_ids == {
            "omnifocus-tag:Work",
            "omnifocus-tag:People/Boss",
            "omnifocus-tag:Urgent",
        }

    def test_no_tags(self) -> None:
        event = _make_event(tags=[])
        docs = self.parser.parse(event)
        hints = docs[0].graph_hints
        tag_hints = [h for h in hints if h.predicate == "TAGGED"]
        assert len(tag_hints) == 0

    def test_empty_tag_skipped(self) -> None:
        event = _make_event(tags=["Work", "", "  "])
        docs = self.parser.parse(event)
        hints = docs[0].graph_hints
        tag_hints = [h for h in hints if h.predicate == "TAGGED"]
        assert len(tag_hints) == 1


# ── project graph hints ─────────────────────────────────────────


class TestProjectGraphHints:
    def setup_method(self) -> None:
        self.parser = OmniFocusParser()

    def test_project_edge(self) -> None:
        event = _make_event(project="Home Renovation")
        docs = self.parser.parse(event)
        hints = docs[0].graph_hints
        proj_hints = [h for h in hints if h.predicate == "IN_PROJECT"]
        assert len(proj_hints) == 1
        assert proj_hints[0].object_id == "omnifocus-project:Home Renovation"
        assert proj_hints[0].object_label == "Project"
        assert proj_hints[0].object_props["name"] == "Home Renovation"
        assert proj_hints[0].confidence == 1.0

    def test_no_project(self) -> None:
        event = _make_event(project="")
        docs = self.parser.parse(event)
        hints = docs[0].graph_hints
        proj_hints = [h for h in hints if h.predicate == "IN_PROJECT"]
        assert len(proj_hints) == 0


# ── parent task graph hints ──────────────────────────────────────


class TestParentTaskGraphHints:
    def setup_method(self) -> None:
        self.parser = OmniFocusParser()

    def test_subtask_edge(self) -> None:
        event = _make_event(parent_task="Plan grocery list", parent_task_id="parent-42")
        docs = self.parser.parse(event)
        hints = docs[0].graph_hints
        parent_hints = [h for h in hints if h.predicate == "SUBTASK_OF"]
        assert len(parent_hints) == 1
        assert parent_hints[0].object_id == "omnifocus://parent-42"
        assert parent_hints[0].object_label == "Task"
        assert parent_hints[0].object_props["name"] == "Plan grocery list"
        assert parent_hints[0].confidence == 0.95

    def test_subtask_edge_no_id_skipped(self) -> None:
        """Without parent_task_id we cannot produce a valid merge key — skip the hint."""
        event = _make_event(parent_task="Plan grocery list", parent_task_id="")
        docs = self.parser.parse(event)
        hints = docs[0].graph_hints
        parent_hints = [h for h in hints if h.predicate == "SUBTASK_OF"]
        assert len(parent_hints) == 0

    def test_no_parent(self) -> None:
        event = _make_event(parent_task="", parent_task_id="")
        docs = self.parser.parse(event)
        hints = docs[0].graph_hints
        parent_hints = [h for h in hints if h.predicate == "SUBTASK_OF"]
        assert len(parent_hints) == 0


# ── source metadata ──────────────────────────────────────────────


class TestSourceMetadata:
    def setup_method(self) -> None:
        self.parser = OmniFocusParser()

    def test_metadata_includes_task_id(self) -> None:
        event = _make_event(task_id="abc-123")
        docs = self.parser.parse(event)
        assert docs[0].source_metadata["task_id"] == "abc-123"

    def test_metadata_includes_project(self) -> None:
        event = _make_event(project="Work")
        docs = self.parser.parse(event)
        assert docs[0].source_metadata["project"] == "Work"

    def test_metadata_includes_tags(self) -> None:
        event = _make_event(tags=["a", "b"])
        docs = self.parser.parse(event)
        assert docs[0].source_metadata["tags"] == ["a", "b"]


# ── text content ─────────────────────────────────────────────────


class TestTextContent:
    def setup_method(self) -> None:
        self.parser = OmniFocusParser()

    def test_includes_name(self) -> None:
        event = _make_event(name="Important task")
        docs = self.parser.parse(event)
        assert "Important task" in docs[0].text

    def test_includes_note(self) -> None:
        event = _make_event(note="Detailed description")
        docs = self.parser.parse(event)
        assert "Detailed description" in docs[0].text

    def test_includes_project_in_text(self) -> None:
        event = _make_event(project="My Project")
        docs = self.parser.parse(event)
        assert "Project: My Project" in docs[0].text

    def test_includes_tags_in_text(self) -> None:
        event = _make_event(tags=["Work", "People/Boss"])
        docs = self.parser.parse(event)
        assert "Tags: Work, People/Boss" in docs[0].text

    def test_non_active_status_in_text(self) -> None:
        event = _make_event(status="deferred")
        docs = self.parser.parse(event)
        assert "Status: deferred" in docs[0].text

    def test_active_status_not_in_text(self) -> None:
        event = _make_event(status="active")
        docs = self.parser.parse(event)
        assert "Status:" not in docs[0].text


# ── parser registration ──────────────────────────────────────────


class TestParserRegistration:
    def test_registered_in_registry(self) -> None:
        from worker.parsers.registry import get

        parser = get("omnifocus")
        assert parser.source_type == "omnifocus"


# ── Person extraction (emails, @mentions, People/ tags) ──────────


class TestPersonExtraction:
    def setup_method(self) -> None:
        self.parser = OmniFocusParser()

    def _mentions(self, docs):
        return [h for h in docs[0].graph_hints if h.predicate == "MENTIONS"]

    # -- email extraction --

    def test_email_in_note(self) -> None:
        event = _make_event(note="Contact alice@example.com about this")
        hints = self._mentions(self.parser.parse(event))
        assert len(hints) == 1
        assert hints[0].object_label == "Person"
        assert hints[0].object_props["email"] == "alice@example.com"
        assert hints[0].object_merge_key == "email"
        assert hints[0].object_id == "person:alice@example.com"
        assert hints[0].confidence == 1.0

    def test_multiple_emails_in_note(self) -> None:
        event = _make_event(note="CC alice@a.com and bob@b.com")
        hints = self._mentions(self.parser.parse(event))
        emails = {h.object_props["email"] for h in hints}
        assert emails == {"alice@a.com", "bob@b.com"}

    def test_email_canonicalized(self) -> None:
        event = _make_event(note="user@googlemail.com")
        hints = self._mentions(self.parser.parse(event))
        assert hints[0].object_props["email"] == "user@gmail.com"

    def test_no_email_no_mention_hints(self) -> None:
        event = _make_event(name="Plain task", note="Nothing special")
        hints = self._mentions(self.parser.parse(event))
        assert len(hints) == 0

    # -- @mention extraction --

    def test_at_mention_in_name(self) -> None:
        event = _make_event(name="Review PR with @Alice")
        hints = self._mentions(self.parser.parse(event))
        assert len(hints) == 1
        assert hints[0].object_label == "Person"
        assert hints[0].object_props["name"] == "Alice"
        assert hints[0].confidence == 0.9

    def test_at_mention_two_words(self) -> None:
        event = _make_event(name="Call @Bob Smith tomorrow")
        hints = self._mentions(self.parser.parse(event))
        assert any(h.object_props.get("name") == "Bob Smith" for h in hints)

    def test_at_mention_in_note(self) -> None:
        event = _make_event(note="Ask @Carol about the deadline")
        hints = self._mentions(self.parser.parse(event))
        assert len(hints) == 1
        assert hints[0].object_props["name"] == "Carol"

    def test_at_mention_lowercase_ignored(self) -> None:
        """@mentions must start with uppercase to avoid false positives like @home."""
        event = _make_event(note="working @home today")
        hints = self._mentions(self.parser.parse(event))
        assert len(hints) == 0

    # -- People/ tag extraction --

    def test_people_tag_creates_person_hint(self) -> None:
        event = _make_event(tags=["People/Alice"])
        hints = self._mentions(self.parser.parse(event))
        assert len(hints) == 1
        assert hints[0].object_label == "Person"
        assert hints[0].object_props["name"] == "Alice"
        assert hints[0].confidence == 0.95

    def test_people_tag_nested(self) -> None:
        """Deeper hierarchy like People/Work/Boss keeps everything after People/."""
        event = _make_event(tags=["People/Work/Boss"])
        hints = self._mentions(self.parser.parse(event))
        assert len(hints) == 1
        assert hints[0].object_props["name"] == "Work/Boss"

    def test_non_people_tag_no_person(self) -> None:
        event = _make_event(tags=["Work", "Urgent"])
        hints = self._mentions(self.parser.parse(event))
        assert len(hints) == 0

    def test_people_tag_still_creates_tag_hint(self) -> None:
        """People/ tags should still create TAGGED hints alongside MENTIONS."""
        event = _make_event(tags=["People/Alice"])
        docs = self.parser.parse(event)
        tag_hints = [h for h in docs[0].graph_hints if h.predicate == "TAGGED"]
        mention_hints = [h for h in docs[0].graph_hints if h.predicate == "MENTIONS"]
        assert len(tag_hints) == 1
        assert len(mention_hints) == 1

    # -- deduplication --

    def test_email_and_at_mention_same_name_deduped(self) -> None:
        """If an email and @mention produce the same key, don't duplicate."""
        event = _make_event(note="@Alice alice@example.com")
        hints = self._mentions(self.parser.parse(event))
        # Email hint uses email key, @mention uses name key — both should appear
        # since they track different merge keys
        assert len(hints) == 2

    def test_people_tag_deduped_with_at_mention(self) -> None:
        """If @Alice appears and tag People/Alice exists, produce only one Person."""
        event = _make_event(name="Talk to @Alice", tags=["People/Alice"])
        hints = self._mentions(self.parser.parse(event))
        names = [h.object_props.get("name") for h in hints]
        assert names.count("Alice") == 1

    def test_deleted_task_no_person_hints(self) -> None:
        event = _make_event(operation="deleted")
        docs = self.parser.parse(event)
        assert docs[0].graph_hints == []
