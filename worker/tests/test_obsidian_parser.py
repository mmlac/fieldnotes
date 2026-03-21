"""Tests for the Obsidian parser."""

from pathlib import Path

from worker.parsers.base import canonicalize_email
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


class TestCanonicalizeEmail:
    def test_lowercase(self):
        assert canonicalize_email("Alice@Example.COM") == "alice@example.com"

    def test_strip(self):
        assert canonicalize_email("  bob@x.com  ") == "bob@x.com"

    def test_googlemail_to_gmail(self):
        assert canonicalize_email("user@googlemail.com") == "user@gmail.com"

    def test_gmail_unchanged(self):
        assert canonicalize_email("user@gmail.com") == "user@gmail.com"

    def test_other_domains_unchanged(self):
        assert canonicalize_email("user@outlook.com") == "user@outlook.com"

    def test_empty_string(self):
        assert canonicalize_email("") == ""

    def test_no_at_sign(self):
        assert canonicalize_email("notanemail") == "notanemail"

    def test_googlemail_case_insensitive(self):
        assert canonicalize_email("User@GOOGLEMAIL.COM") == "user@gmail.com"


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

    def test_image_embeds_with_vault_name_fallback(self):
        """vault_name from the source should be used when vault_root is absent."""
        note = "---\n---\n![[pic.png]]"
        docs = self.parser.parse(_make_event(note, meta={"vault_name": "MyVault"}))
        img = [d for d in docs if d.mime_type != "text/plain"][0]
        assert img.source_id == "MyVault/pic.png"

    def test_image_embeds_vault_root_takes_precedence_over_vault_name(self):
        """Explicit vault_root overrides vault_name."""
        note = "---\n---\n![[pic.png]]"
        docs = self.parser.parse(
            _make_event(note, meta={"vault_root": "root", "vault_name": "name"})
        )
        img = [d for d in docs if d.mime_type != "text/plain"][0]
        assert img.source_id == "root/pic.png"

    def test_image_bytes_loaded_from_disk(self, tmp_path: Path):
        """When vault_path is set, image_bytes should be read from disk."""
        img_data = b"\x89PNG\r\n\x1a\nfake"
        (tmp_path / "photo.png").write_bytes(img_data)

        note = "---\n---\n![[photo.png]]"
        event = _make_event(note, meta={"vault_path": str(tmp_path)})
        docs = self.parser.parse(event)
        img = [d for d in docs if d.mime_type != "text/plain"][0]
        assert img.image_bytes == img_data

    def test_image_bytes_none_when_file_missing(self):
        """Missing image file should result in image_bytes=None, not an error."""
        note = "---\n---\n![[missing.png]]"
        event = _make_event(note, meta={"vault_path": "/nonexistent/vault"})
        docs = self.parser.parse(event)
        img = [d for d in docs if d.mime_type != "text/plain"][0]
        assert img.image_bytes is None

    def test_image_bytes_none_without_vault_path(self):
        """Without vault_path in meta, image_bytes should remain None."""
        note = "---\n---\n![[photo.png]]"
        docs = self.parser.parse(_make_event(note))
        img = [d for d in docs if d.mime_type != "text/plain"][0]
        assert img.image_bytes is None

    def test_image_bytes_nested_path(self, tmp_path: Path):
        """Embedded images in subdirectories should be read correctly."""
        (tmp_path / "attachments").mkdir()
        img_data = b"\xff\xd8\xff\xe0fake-jpeg"
        (tmp_path / "attachments" / "pic.jpg").write_bytes(img_data)

        note = "---\n---\n![[attachments/pic.jpg]]"
        event = _make_event(note, meta={"vault_path": str(tmp_path)})
        docs = self.parser.parse(event)
        img = [d for d in docs if d.mime_type != "text/plain"][0]
        assert img.image_bytes == img_data

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

    def test_wikilink_heading_anchor_stripped(self):
        """[[note#heading]] should produce object_id='note', not 'note#heading'."""
        note = "---\n---\nSee [[Other Note#section]] and [[Plain Link]]."
        docs = self.parser.parse(_make_event(note))
        link_hints = [h for h in docs[0].graph_hints if h.predicate == "LINKS_TO"]
        targets = {h.object_id for h in link_hints}
        assert targets == {"Other Note", "Plain Link"}

    def test_wikilink_heading_only_anchor_skipped(self):
        """[[#heading]] (self-link to heading) should be skipped as empty target."""
        note = "---\n---\nSee [[#local-heading]]."
        docs = self.parser.parse(_make_event(note))
        link_hints = [h for h in docs[0].graph_hints if h.predicate == "LINKS_TO"]
        assert len(link_hints) == 0

    def test_path_traversal_embed_rejected(self, tmp_path: Path):
        """Embed paths that escape the vault directory must be skipped."""
        secret = tmp_path / "outside" / "secret.png"
        secret.parent.mkdir()
        secret.write_bytes(b"SECRET")

        vault = tmp_path / "vault"
        vault.mkdir()

        note = "---\n---\n![[../outside/secret.png]]"
        event = _make_event(note, meta={"vault_path": str(vault)})
        docs = self.parser.parse(event)
        # Only the text document should be returned; the traversal embed is skipped
        assert len(docs) == 1
        assert docs[0].mime_type == "text/plain"

    def test_category_frontmatter(self):
        note = "---\ncategories:\n  - books\n---\nSome content."
        docs = self.parser.parse(_make_event(note))
        assert docs[0].node_props["categories"] == ["books"]

    def test_category_absent_when_not_set(self):
        note = "---\ntitle: No Category\n---\nContent."
        docs = self.parser.parse(_make_event(note))
        assert "categories" not in docs[0].node_props

    def test_category_frontmatter_extracted(self):
        note = "---\ncategories: [ProjectX]\n---\nSome content."
        docs = self.parser.parse(_make_event(note))
        assert docs[0].node_props["categories"] == ["ProjectX"]

    def test_category_string_normalised_to_list(self):
        note = "---\ncategories: ProjectX\n---\nSome content."
        docs = self.parser.parse(_make_event(note))
        assert docs[0].node_props["categories"] == ["ProjectX"]

    def test_category_name_synthesized_tag_targets_omnifocus_id(self):
        note = "---\ncategories: [ProjectX]\ntitle: AlphaTask\n---\nContent."
        docs = self.parser.parse(
            _make_event(note, meta={"relative_path": "AlphaTask.md"})
        )
        tag_hints = [h for h in docs[0].graph_hints if h.predicate == "TAGGED_BY_USER"]
        omnifocus_hints = [
            h for h in tag_hints if h.object_id.startswith("omnifocus-tag:")
        ]
        assert len(omnifocus_hints) == 1
        hint = omnifocus_hints[0]
        assert hint.object_id == "omnifocus-tag:ProjectX/AlphaTask"
        assert hint.object_props["name"] == "ProjectX/AlphaTask"
        assert hint.object_props["source"] == "omnifocus"

    def test_category_name_from_filename(self):
        note = "---\ncategories: [ProjectX]\n---\nContent."
        docs = self.parser.parse(
            _make_event(note, meta={"relative_path": "BetaTask.md"})
        )
        tag_hints = [h for h in docs[0].graph_hints if h.predicate == "TAGGED_BY_USER"]
        omnifocus_hints = [
            h for h in tag_hints if h.object_id.startswith("omnifocus-tag:")
        ]
        assert len(omnifocus_hints) == 1
        assert omnifocus_hints[0].object_id == "omnifocus-tag:ProjectX/BetaTask"

    def test_multiple_categories_emit_multiple_hints(self):
        note = "---\ncategories: [Work, Personal]\n---\nContent."
        docs = self.parser.parse(
            _make_event(note, meta={"relative_path": "Deploy.md"})
        )
        tag_hints = [h for h in docs[0].graph_hints if h.predicate == "TAGGED_BY_USER"]
        omnifocus_hints = [
            h for h in tag_hints if h.object_id.startswith("omnifocus-tag:")
        ]
        assert len(omnifocus_hints) == 2
        ids = {h.object_id for h in omnifocus_hints}
        assert ids == {"omnifocus-tag:Work/Deploy", "omnifocus-tag:Personal/Deploy"}

    def test_custom_categories_key(self):
        note = "---\ntopic: Research\n---\nContent."
        docs = self.parser.parse(
            _make_event(note, meta={"relative_path": "LLMs.md", "categories_key": "topic"})
        )
        tag_hints = [h for h in docs[0].graph_hints if h.predicate == "TAGGED_BY_USER"]
        omnifocus_hints = [
            h for h in tag_hints if h.object_id.startswith("omnifocus-tag:")
        ]
        assert len(omnifocus_hints) == 1
        assert omnifocus_hints[0].object_id == "omnifocus-tag:Research/LLMs"

    def test_no_category_no_synthesized_hint(self):
        note = "---\ntitle: AlphaTask\n---\nContent."
        docs = self.parser.parse(_make_event(note))
        tag_hints = [h for h in docs[0].graph_hints if h.predicate == "TAGGED_BY_USER"]
        omnifocus_hints = [
            h for h in tag_hints if h.object_id.startswith("omnifocus-tag:")
        ]
        assert len(omnifocus_hints) == 0

    def test_category_without_name_no_hint(self):
        note = "---\ncategories: [ProjectX]\n---\nContent."
        docs = self.parser.parse(_make_event(note))
        tag_hints = [h for h in docs[0].graph_hints if h.predicate == "TAGGED_BY_USER"]
        omnifocus_hints = [
            h for h in tag_hints if h.object_id.startswith("omnifocus-tag:")
        ]
        assert len(omnifocus_hints) == 0

    def test_registry_registration(self):
        from worker.parsers.registry import get

        parser = get("obsidian")
        assert isinstance(parser, ObsidianParser)

    # --- emails frontmatter → MENTIONS Person hints --------------------------

    def test_emails_string_comma_separated(self):
        note = "---\nemails: alice@example.com, bob@example.com\n---\nContent."
        docs = self.parser.parse(_make_event(note))
        person_hints = [h for h in docs[0].graph_hints if h.predicate == "MENTIONS"]
        assert len(person_hints) == 2
        emails = {h.object_props["email"] for h in person_hints}
        assert emails == {"alice@example.com", "bob@example.com"}

    def test_emails_list_format(self):
        note = "---\nemails:\n  - alice@example.com\n  - bob@example.com\n---\nContent."
        docs = self.parser.parse(_make_event(note))
        person_hints = [h for h in docs[0].graph_hints if h.predicate == "MENTIONS"]
        assert len(person_hints) == 2

    def test_emails_person_node_properties(self):
        note = "---\nemails: alice@example.com\n---\nContent."
        docs = self.parser.parse(_make_event(note))
        hint = [h for h in docs[0].graph_hints if h.predicate == "MENTIONS"][0]
        assert hint.subject_id == "notes/test.md"
        assert hint.subject_label == "File"
        assert hint.object_id == "person:alice@example.com"
        assert hint.object_label == "Person"
        assert hint.object_merge_key == "email"
        assert hint.object_props == {"email": "alice@example.com"}

    def test_emails_googlemail_canonicalized(self):
        note = "---\nemails: alice@googlemail.com\n---\nContent."
        docs = self.parser.parse(_make_event(note))
        hint = [h for h in docs[0].graph_hints if h.predicate == "MENTIONS"][0]
        assert hint.object_id == "person:alice@gmail.com"
        assert hint.object_props["email"] == "alice@gmail.com"

    def test_emails_case_normalized(self):
        note = "---\nemails: Alice@Example.COM\n---\nContent."
        docs = self.parser.parse(_make_event(note))
        hint = [h for h in docs[0].graph_hints if h.predicate == "MENTIONS"][0]
        assert hint.object_props["email"] == "alice@example.com"

    def test_emails_whitespace_trimmed(self):
        note = "---\nemails: \"  alice@example.com  ,  bob@example.com  \"\n---\nContent."
        docs = self.parser.parse(_make_event(note))
        person_hints = [h for h in docs[0].graph_hints if h.predicate == "MENTIONS"]
        assert len(person_hints) == 2
        emails = {h.object_props["email"] for h in person_hints}
        assert emails == {"alice@example.com", "bob@example.com"}

    def test_emails_empty_no_hints(self):
        note = "---\nemails: \"\"\n---\nContent."
        docs = self.parser.parse(_make_event(note))
        person_hints = [h for h in docs[0].graph_hints if h.predicate == "MENTIONS"]
        assert len(person_hints) == 0

    def test_emails_invalid_skipped(self):
        note = "---\nemails: notanemail, valid@example.com\n---\nContent."
        docs = self.parser.parse(_make_event(note))
        person_hints = [h for h in docs[0].graph_hints if h.predicate == "MENTIONS"]
        assert len(person_hints) == 1
        assert person_hints[0].object_props["email"] == "valid@example.com"

    def test_emails_cross_source_with_gmail(self):
        """Person ID format matches Gmail parser for cross-source linking."""
        from worker.parsers.gmail import GmailParser

        # Obsidian note with emails
        note = "---\nemails: alice@example.com\n---\nContent."
        obs_docs = self.parser.parse(_make_event(note))
        obs_hint = [h for h in obs_docs[0].graph_hints if h.predicate == "MENTIONS"][0]

        # Gmail email from the same person
        gmail_event = {
            "source_type": "gmail",
            "source_id": "gmail:msg-1",
            "operation": "created",
            "text": "Hello",
            "meta": {"sender_email": "alice@example.com", "recipients": []},
        }
        gmail_docs = GmailParser().parse(gmail_event)
        gmail_hint = [h for h in gmail_docs[0].graph_hints if h.predicate == "SENT"][0]

        # Both should resolve to the same Person node
        assert obs_hint.object_id == gmail_hint.subject_id
        assert obs_hint.object_merge_key == gmail_hint.subject_merge_key

    def test_no_emails_frontmatter_no_hints(self):
        note = "---\ntitle: No Emails\n---\nContent."
        docs = self.parser.parse(_make_event(note))
        person_hints = [h for h in docs[0].graph_hints if h.predicate == "MENTIONS"]
        assert len(person_hints) == 0
