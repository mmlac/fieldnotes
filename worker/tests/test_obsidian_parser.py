"""Tests for the Obsidian parser."""

from worker.parsers.base import GraphHint, ParsedDocument
from worker.parsers.obsidian import ObsidianParser


def _make_event(text: str, **overrides) -> dict:
    base = {
        "source_type": "obsidian",
        "source_id": "notes/test.md",
        "operation": "created",
        "text": text,
        "meta": {},
    }
    base.update(overrides)
    return base


class TestObsidianParser:
    def setup_method(self):
        self.parser = ObsidianParser()

    def test_source_type(self):
        assert self.parser.source_type == "obsidian"

    def test_basic_frontmatter(self):
        note = "---\ntitle: My Note\ntags:\n  - foo\n  - bar\n---\nHello world"
        docs = self.parser.parse(_make_event(note))
        assert len(docs) == 1
        doc = docs[0]
        assert doc.node_props["title"] == "My Note"
        assert doc.source_metadata["frontmatter"]["title"] == "My Note"

    def test_wikilinks_produce_graph_hints(self):
        note = "---\ntitle: Test\n---\nSee [[Other Note]] and [[Folder/Deep]]."
        docs = self.parser.parse(_make_event(note))
        doc = docs[0]
        link_hints = [h for h in doc.graph_hints if h.predicate == "LINKS_TO"]
        assert len(link_hints) == 2
        targets = {h.object_id for h in link_hints}
        assert targets == {"Other Note", "Folder/Deep"}
        for h in link_hints:
            assert h.confidence == 0.95
            assert h.subject_id == "notes/test.md"

    def test_wikilink_alias_ignored(self):
        note = "---\n---\nSee [[Target|display text]]."
        docs = self.parser.parse(_make_event(note))
        link_hints = [h for h in docs[0].graph_hints if h.predicate == "LINKS_TO"]
        assert len(link_hints) == 1
        assert link_hints[0].object_id == "Target"

    def test_tags_in_body(self):
        note = "---\n---\nSome text #project/alpha and #beta here."
        docs = self.parser.parse(_make_event(note))
        tag_hints = [h for h in docs[0].graph_hints if h.predicate == "TAGGED_BY_USER"]
        tag_ids = {h.object_id for h in tag_hints}
        assert "tag:project/alpha" in tag_ids
        assert "tag:beta" in tag_ids
        for h in tag_hints:
            assert h.confidence == 1.0
            assert h.object_props == {"source": "user"}

    def test_tags_in_frontmatter(self):
        note = "---\ntags:\n  - fromfm\n---\nBody text."
        docs = self.parser.parse(_make_event(note))
        tag_hints = [h for h in docs[0].graph_hints if h.predicate == "TAGGED_BY_USER"]
        tag_ids = {h.object_id for h in tag_hints}
        assert "tag:fromfm" in tag_ids

    def test_web_clip_detection(self):
        note = "---\nurl: https://example.com\n---\nClipped content."
        docs = self.parser.parse(_make_event(note))
        doc = docs[0]
        assert doc.node_props["web_clip"] is True
        assert doc.node_props["source_url"] == "https://example.com"
        assert doc.source_metadata["web_clip"] is True

    def test_image_embeds(self):
        note = "---\n---\nSome text\n![[photo.png]]\nMore text\n![[deep/img.jpg]]"
        docs = self.parser.parse(_make_event(note))
        assert len(docs) == 3  # 1 text + 2 images
        img_docs = [d for d in docs if d.mime_type != "text/plain"]
        assert len(img_docs) == 2
        img_ids = {d.source_id for d in img_docs}
        assert "photo.png" in img_ids
        assert "deep/img.jpg" in img_ids
        for d in img_docs:
            assert d.text == ""
            assert d.node_props["embedded_in"] == "notes/test.md"

    def test_image_embeds_with_vault_root(self):
        note = "---\n---\n![[pic.png]]"
        docs = self.parser.parse(_make_event(note, meta={"vault_root": "vault"}))
        img = [d for d in docs if d.mime_type != "text/plain"][0]
        assert img.source_id == "vault/pic.png"

    def test_embed_not_wikilink(self):
        """![[image.png]] should NOT produce a LINKS_TO hint."""
        note = "---\n---\n![[photo.png]] and [[Real Link]]"
        docs = self.parser.parse(_make_event(note))
        link_hints = [h for h in docs[0].graph_hints if h.predicate == "LINKS_TO"]
        targets = {h.object_id for h in link_hints}
        assert "photo.png" not in targets
        assert "Real Link" in targets

    def test_deleted_operation(self):
        docs = self.parser.parse(_make_event("", operation="deleted"))
        assert len(docs) == 1
        assert docs[0].operation == "deleted"
        assert docs[0].text == ""

    def test_tags_inside_inline_code_ignored(self):
        """#tags inside backtick inline code should not produce hints."""
        note = "---\n---\nSome `# this is a comment` and real #valid tag."
        docs = self.parser.parse(_make_event(note))
        tag_hints = [h for h in docs[0].graph_hints if h.predicate == "TAGGED_BY_USER"]
        tag_ids = {h.object_id for h in tag_hints}
        assert "tag:valid" in tag_ids
        assert "tag:this" not in tag_ids

    def test_tags_inside_fenced_code_ignored(self):
        """#tags inside fenced code blocks should not produce hints."""
        note = "---\n---\nBefore\n```python\n# this is a comment\nx = #notag\n```\nAfter #real"
        docs = self.parser.parse(_make_event(note))
        tag_hints = [h for h in docs[0].graph_hints if h.predicate == "TAGGED_BY_USER"]
        tag_ids = {h.object_id for h in tag_hints}
        assert "tag:real" in tag_ids
        assert "tag:this" not in tag_ids
        assert "tag:notag" not in tag_ids

    def test_tags_inside_tilde_fenced_code_ignored(self):
        """#tags inside ~~~ fenced code blocks should not produce hints."""
        note = "---\n---\n~~~\n#inside\n~~~\n#outside here"
        docs = self.parser.parse(_make_event(note))
        tag_hints = [h for h in docs[0].graph_hints if h.predicate == "TAGGED_BY_USER"]
        tag_ids = {h.object_id for h in tag_hints}
        assert "tag:outside" in tag_ids
        assert "tag:inside" not in tag_ids

    def test_registry_registration(self):
        from worker.parsers.registry import get

        parser = get("obsidian")
        assert isinstance(parser, ObsidianParser)
