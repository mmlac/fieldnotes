"""Gmail end-to-end integration test.

Exercises the full Gmail flow: mock Gmail API → GmailSource emits events →
GmailParser produces ParsedDocuments → Pipeline processes them → verify
Person/Email/Thread nodes via Writer.

Stubs external services (Gmail API, Ollama, Neo4j, Qdrant) but exercises
the real parsing, chunking, extraction-result handling, and graph hint logic.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from worker.config import Config, ModelConfig, ProviderConfig, RolesConfig
from worker.models.resolver import ModelRegistry
from worker.parsers.gmail import GmailParser
from worker.pipeline import Pipeline
from worker.pipeline.extractor import ExtractionResult
from worker.pipeline.resolver import ResolvedEntity, ResolutionResult
from worker.pipeline.writer import WriteUnit, Writer
from worker.sources.gmail import GmailSource, _build_ingest_event

# Ensure provider decorators run
import worker.models.providers  # noqa: F401


def _make_config() -> Config:
    """Build a minimal Config for the pipeline."""
    cfg = Config()
    cfg.providers["local"] = ProviderConfig(
        name="local",
        type="ollama",
        settings={},
    )
    cfg.models["llm"] = ModelConfig(alias="llm", provider="local", model="qwen3.5:27b")
    cfg.models["embedder"] = ModelConfig(
        alias="embedder",
        provider="local",
        model="nomic-embed-text",
    )
    cfg.roles = RolesConfig(
        mapping={
            "extraction": "llm",
            "embedding": "embedder",
        }
    )
    return cfg


def _make_gmail_message(
    msg_id: str = "msg-001",
    thread_id: str = "thread-100",
    subject: str = "Project Update",
    sender: str = "Alice <alice@example.com>",
    to: str = "bob@example.com, carol@example.com",
    date: str = "Tue, 11 Mar 2026 14:30:00 +0000",
    internal_date: str = "1773506400000",
) -> dict[str, Any]:
    """Build a fake Gmail API message resource."""
    return {
        "id": msg_id,
        "threadId": thread_id,
        "internalDate": internal_date,
        "historyId": "12345",
        "payload": {
            "headers": [
                {"name": "Subject", "value": subject},
                {"name": "From", "value": sender},
                {"name": "To", "value": to},
                {"name": "Date", "value": date},
            ],
        },
    }


class TestGmailEndToEnd:
    """Full Gmail pipeline: source event → parse → pipeline → writer."""

    @patch("worker.pipeline.resolve_entities_from_registry")
    @patch("worker.pipeline.extract_chunks")
    @patch("worker.pipeline.embed_chunks")
    def test_gmail_full_pipeline(
        self,
        mock_embed,
        mock_extract,
        mock_resolve,
    ) -> None:
        """Mock Gmail API message → parser → pipeline → verify WriteUnit
        has Email node, Person/Thread graph hints, and correct structure."""
        # --- Setup ---
        registry = ModelRegistry(_make_config())
        writer = MagicMock(spec=Writer)
        writer.fetch_existing_entities.return_value = []
        pipeline = Pipeline(registry=registry, writer=writer)

        # 1. Build an IngestEvent from a mock Gmail API message
        msg = _make_gmail_message()
        event = _build_ingest_event(msg)

        # Inject email body text (source would fetch this separately)
        event["text"] = (
            "Hi team, here's the project update. We shipped the new API "
            "endpoint and updated the documentation. " * 10
        )

        # 2. Parse with GmailParser
        parser = GmailParser()
        docs = parser.parse(event)
        assert len(docs) == 1
        doc = docs[0]

        # Verify parser output
        assert doc.source_type == "gmail"
        assert doc.source_id == "gmail:msg-001"
        assert doc.operation == "created"
        assert doc.node_label == "Email"
        assert doc.node_props["message_id"] == "msg-001"
        assert doc.node_props["subject"] == "Project Update"

        # Verify graph hints
        sent_hints = [h for h in doc.graph_hints if h.predicate == "SENT"]
        to_hints = [h for h in doc.graph_hints if h.predicate == "TO"]
        part_of_hints = [h for h in doc.graph_hints if h.predicate == "PART_OF"]

        assert len(sent_hints) == 1
        assert sent_hints[0].subject_id == "person:alice@example.com"
        assert sent_hints[0].subject_label == "Person"
        assert sent_hints[0].subject_merge_key == "email"

        assert len(to_hints) == 2
        recipient_ids = {h.object_id for h in to_hints}
        assert "person:bob@example.com" in recipient_ids
        assert "person:carol@example.com" in recipient_ids

        assert len(part_of_hints) == 1
        assert part_of_hints[0].object_id == "gmail-thread:thread-100"
        assert part_of_hints[0].object_label == "Thread"
        assert part_of_hints[0].object_merge_key == "thread_id"

        # 3. Mock pipeline stages
        mock_embed.side_effect = lambda texts, reg: [(t, [0.1] * 768) for t in texts]
        mock_extract.side_effect = lambda chunks, reg, **kw: [
            ExtractionResult(
                entities=[{"name": "API", "type": "Technology", "confidence": 0.9}],
                triples=[],
            )
            for _ in chunks
        ]
        mock_resolve.return_value = ResolutionResult(
            entities=[
                ResolvedEntity(
                    name="API",
                    type="Technology",
                    confidence=0.9,
                    merged_into=None,
                ),
            ],
        )

        # --- Execute pipeline ---
        pipeline.process(doc)

        # --- Verify writer was called ---
        writer.write.assert_called_once()
        unit: WriteUnit = writer.write.call_args[0][0]

        # Email node
        assert unit.doc.source_id == "gmail:msg-001"
        assert unit.doc.node_label == "Email"
        assert unit.doc.source_type == "gmail"

        # Chunks and vectors
        assert len(unit.chunks) >= 1
        assert len(unit.vectors) == len(unit.chunks)
        assert all(len(v) == 768 for v in unit.vectors)

        # Entities from extraction
        assert any(e["name"] == "API" for e in unit.entities)

        # Graph hints preserved through pipeline
        assert len(unit.doc.graph_hints) == 4  # 1 SENT + 2 TO + 1 PART_OF
        hint_predicates = {h.predicate for h in unit.doc.graph_hints}
        assert hint_predicates == {"SENT", "TO", "PART_OF"}

    def test_gmail_html_body_stripped(self) -> None:
        """HTML email bodies are stripped to plain text before pipeline."""
        msg = _make_gmail_message()
        event = _build_ingest_event(msg)
        event["text"] = (
            "<html><body>"
            "<p>Hi team,</p>"
            "<p>The <b>deployment</b> is complete.</p>"
            "<style>.header{color:blue}</style>"
            "</body></html>"
        )
        event["mime_type"] = "text/html"

        parser = GmailParser()
        docs = parser.parse(event)
        doc = docs[0]

        assert "<p>" not in doc.text
        assert "<b>" not in doc.text
        assert "<style>" not in doc.text
        assert "Hi team," in doc.text
        assert "deployment" in doc.text

    def test_gmail_deleted_operation(self) -> None:
        """Deleted emails go straight to writer without chunking."""
        registry = ModelRegistry(_make_config())
        writer = MagicMock(spec=Writer)
        pipeline = Pipeline(registry=registry, writer=writer)

        msg = _make_gmail_message()
        event = _build_ingest_event(msg)
        event["operation"] = "deleted"

        parser = GmailParser()
        docs = parser.parse(event)
        assert len(docs) == 1
        assert docs[0].operation == "deleted"
        assert docs[0].graph_hints == []

        pipeline.process(docs[0])

        writer.write.assert_called_once()
        unit: WriteUnit = writer.write.call_args[0][0]
        assert unit.doc.operation == "deleted"
        assert unit.chunks == []
        assert unit.vectors == []

    @patch("worker.pipeline.resolve_entities_from_registry")
    @patch("worker.pipeline.extract_chunks")
    @patch("worker.pipeline.embed_chunks")
    def test_gmail_multiple_messages_same_thread(
        self,
        mock_embed,
        mock_extract,
        mock_resolve,
    ) -> None:
        """Multiple emails in the same thread share the same Thread node
        via PART_OF hints with matching thread_id merge keys."""
        registry = ModelRegistry(_make_config())
        writer = MagicMock(spec=Writer)
        writer.fetch_existing_entities.return_value = []
        pipeline = Pipeline(registry=registry, writer=writer)

        mock_embed.side_effect = lambda texts, reg: [(t, [0.1] * 768) for t in texts]
        mock_extract.side_effect = lambda chunks, reg, **kw: [
            ExtractionResult(entities=[], triples=[]) for _ in chunks
        ]
        mock_resolve.return_value = ResolutionResult(entities=[])

        parser = GmailParser()

        # Two messages in the same thread
        for msg_id in ("msg-001", "msg-002"):
            msg = _make_gmail_message(
                msg_id=msg_id,
                thread_id="thread-shared",
                subject="Discussion",
            )
            event = _build_ingest_event(msg)
            event["text"] = f"Message body for {msg_id}. " * 20

            docs = parser.parse(event)
            pipeline.process(docs[0])

        assert writer.write.call_count == 2

        # Both WriteUnits should have PART_OF hints pointing to same thread
        for call in writer.write.call_args_list:
            unit: WriteUnit = call[0][0]
            part_of = [h for h in unit.doc.graph_hints if h.predicate == "PART_OF"]
            assert len(part_of) == 1
            assert part_of[0].object_id == "gmail-thread:thread-shared"
            assert part_of[0].object_merge_key == "thread_id"
            assert part_of[0].object_props["thread_id"] == "thread-shared"

    def test_gmail_source_build_ingest_event(self) -> None:
        """Verify _build_ingest_event produces well-formed events."""
        msg = _make_gmail_message(
            msg_id="msg-xyz",
            thread_id="thread-abc",
            subject="Test Email",
            sender="Dan <dan@company.com>",
            to="eve@company.com",
        )
        event = _build_ingest_event(msg)

        assert event["source_type"] == "gmail"
        assert event["source_id"] == "gmail:msg-xyz"
        assert event["operation"] == "created"
        assert event["mime_type"] == "message/rfc822"
        assert event["meta"]["message_id"] == "msg-xyz"
        assert event["meta"]["thread_id"] == "thread-abc"
        assert event["meta"]["subject"] == "Test Email"
        assert event["meta"]["sender_email"] == "Dan <dan@company.com>"
        assert "eve@company.com" in event["meta"]["recipients"]

    def test_gmail_source_configure(self, tmp_path) -> None:
        """GmailSource.configure() reads config fields correctly."""
        secrets = tmp_path / "creds.json"
        secrets.write_text("{}")

        source = GmailSource()
        source.configure(
            {
                "account": "default",
                "client_secrets_path": str(secrets),
                "poll_interval_seconds": 60,
                "max_initial_threads": 100,
                "label_filter": "IMPORTANT",
            }
        )

        assert source.name() == "gmail"
        assert source._account == "default"
        assert source._poll_interval == 60
        assert source._max_initial_threads == 100
        assert source._label_filter == "IMPORTANT"
        assert source._client_secrets_path == secrets.resolve()

    def test_gmail_source_configure_requires_secrets(self) -> None:
        """GmailSource.configure() raises if client_secrets_path missing."""
        source = GmailSource()
        with pytest.raises(ValueError, match="client_secrets_path"):
            source.configure({"account": "default"})

    @patch("worker.pipeline.resolve_entities_from_registry")
    @patch("worker.pipeline.extract_chunks")
    @patch("worker.pipeline.embed_chunks")
    def test_gmail_person_merge_keys(
        self,
        mock_embed,
        mock_extract,
        mock_resolve,
    ) -> None:
        """Person nodes use email as merge key for deduplication across emails."""
        registry = ModelRegistry(_make_config())
        writer = MagicMock(spec=Writer)
        writer.fetch_existing_entities.return_value = []
        pipeline = Pipeline(registry=registry, writer=writer)

        mock_embed.side_effect = lambda texts, reg: [(t, [0.1] * 768) for t in texts]
        mock_extract.side_effect = lambda chunks, reg, **kw: [
            ExtractionResult(entities=[], triples=[]) for _ in chunks
        ]
        mock_resolve.return_value = ResolutionResult(entities=[])

        parser = GmailParser()

        # Same sender appears in two different emails
        for msg_id, subject in [("msg-a", "First"), ("msg-b", "Second")]:
            msg = _make_gmail_message(
                msg_id=msg_id,
                thread_id=f"thread-{msg_id}",
                subject=subject,
                sender="Alice <alice@example.com>",
                to="bob@example.com",
            )
            event = _build_ingest_event(msg)
            event["text"] = f"Content of {subject}. " * 20
            docs = parser.parse(event)
            pipeline.process(docs[0])

        assert writer.write.call_count == 2

        # Both emails have SENT hints from the same Person merge key
        for call in writer.write.call_args_list:
            unit: WriteUnit = call[0][0]
            sent = [h for h in unit.doc.graph_hints if h.predicate == "SENT"]
            assert len(sent) == 1
            assert sent[0].subject_id == "person:alice@example.com"
            assert sent[0].subject_merge_key == "email"
            assert sent[0].subject_props == {"email": "alice@example.com"}

    @patch("worker.pipeline.resolve_entities_from_registry")
    @patch("worker.pipeline.extract_chunks")
    @patch("worker.pipeline.embed_chunks")
    def test_gmail_batch_processing(
        self,
        mock_embed,
        mock_extract,
        mock_resolve,
    ) -> None:
        """process_batch isolates failures between gmail documents."""
        registry = ModelRegistry(_make_config())
        writer = MagicMock(spec=Writer)
        writer.fetch_existing_entities.return_value = []
        pipeline = Pipeline(registry=registry, writer=writer)

        # First embed call fails, second succeeds
        call_count = [0]

        def embed_side_effect(texts, reg):
            call_count[0] += 1
            if call_count[0] == 1:
                raise RuntimeError("embed failed")
            return [(t, [0.1] * 768) for t in texts]

        mock_embed.side_effect = embed_side_effect
        mock_extract.side_effect = lambda chunks, reg, **kw: [
            ExtractionResult(entities=[], triples=[]) for _ in chunks
        ]
        mock_resolve.return_value = ResolutionResult(entities=[])

        parser = GmailParser()
        docs = []
        for msg_id in ("msg-fail", "msg-ok"):
            msg = _make_gmail_message(msg_id=msg_id)
            event = _build_ingest_event(msg)
            event["text"] = f"Email body for {msg_id}. " * 20
            docs.extend(parser.parse(event))

        failed = pipeline.process_batch(docs)

        # First doc failed, second succeeded
        assert len(failed) == 1
        assert failed[0].source_id == "gmail:msg-fail"
        assert any(
            call[0][0].doc.source_id == "gmail:msg-ok"
            for call in writer.write.call_args_list
        )
