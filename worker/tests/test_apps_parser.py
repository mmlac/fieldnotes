"""Tests for the macOS apps and Homebrew parsers."""

from worker.parsers.apps import HomebrewParser, MacOSAppsParser


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _app_event(
    bundle_id: str = "com.example.App",
    name: str = "Example App",
    version: str = "1.2.3",
    path: str = "/Applications/Example App.app",
    category: str = "",
    operation: str = "created",
) -> dict:
    return {
        "source_type": "macos_apps",
        "source_id": f"app://{bundle_id}",
        "operation": operation,
        "mime_type": "application/x-apple-app",
        "meta": {
            "name": name,
            "bundle_id": bundle_id,
            "version": version,
            "path": path,
            "category": category,
        },
    }


def _formula_event(
    name: str = "ripgrep",
    version: str = "14.1.0",
    description: str = "Search tool like grep",
    tap: str = "homebrew/core",
    homepage: str = "https://github.com/BurntSushi/ripgrep",
    binaries: list | None = None,
    operation: str = "created",
) -> dict:
    meta: dict = {
        "package_name": name,
        "package_kind": "formula",
        "version": version,
        "tap": tap,
        "homepage": homepage,
    }
    if binaries is not None:
        meta["binaries"] = binaries
    text_parts = [f"{name} (formula)"]
    if description:
        text_parts.append(description)
    if version:
        text_parts.append(f"Version: {version}")
    return {
        "source_type": "homebrew",
        "source_id": f"brew://formula/{name}",
        "operation": operation,
        "mime_type": "text/plain",
        "text": "\n".join(text_parts),
        "meta": meta,
    }


def _cask_event(
    token: str = "firefox",
    name: str = "Firefox",
    version: str = "124.0",
    description: str = "Web browser",
    tap: str = "homebrew/cask",
    homepage: str = "https://www.mozilla.org/firefox/",
    bundle_id: str = "org.mozilla.firefox",
    operation: str = "created",
) -> dict:
    meta: dict = {
        "package_name": name,
        "package_kind": "cask",
        "version": version,
        "tap": tap,
        "homepage": homepage,
    }
    if bundle_id:
        meta["bundle_id"] = bundle_id
        meta["installed_via_brew"] = True
    text_parts = [f"{name} (cask)"]
    if description:
        text_parts.append(description)
    if version:
        text_parts.append(f"Version: {version}")
    return {
        "source_type": "homebrew",
        "source_id": f"brew://cask/{token}",
        "operation": operation,
        "mime_type": "text/plain",
        "text": "\n".join(text_parts),
        "meta": meta,
    }


# ---------------------------------------------------------------------------
# MacOSAppsParser
# ---------------------------------------------------------------------------


class TestMacOSAppsParser:
    def setup_method(self):
        self.parser = MacOSAppsParser()

    def test_source_type(self):
        assert self.parser.source_type == "macos_apps"

    def test_basic_parse(self):
        docs = self.parser.parse(_app_event())
        assert len(docs) == 1
        doc = docs[0]
        assert doc.source_type == "macos_apps"
        assert doc.source_id == "app://com.example.App"
        assert doc.operation == "created"
        assert doc.node_label == "Application"
        assert "Example App" in doc.text
        assert "v1.2.3" in doc.text

    def test_node_props(self):
        docs = self.parser.parse(_app_event())
        props = docs[0].node_props
        assert props["name"] == "Example App"
        assert props["bundle_id"] == "com.example.App"
        assert props["version"] == "1.2.3"
        assert props["path"] == "/Applications/Example App.app"

    def test_source_metadata(self):
        docs = self.parser.parse(_app_event())
        meta = docs[0].source_metadata
        assert meta["bundle_id"] == "com.example.App"
        assert meta["path"] == "/Applications/Example App.app"

    def test_category_edge(self):
        docs = self.parser.parse(
            _app_event(category="public.app-category.developer-tools")
        )
        cat_hints = [h for h in docs[0].graph_hints if h.predicate == "CATEGORIZED_AS"]
        assert len(cat_hints) == 1
        h = cat_hints[0]
        assert h.subject_id == "app://com.example.App"
        assert h.subject_label == "Application"
        assert h.object_id == "category:public.app-category.developer-tools"
        assert h.object_label == "Category"
        assert h.object_props["name"] == "public.app-category.developer-tools"
        assert h.object_merge_key == "name"

    def test_no_category_no_edge(self):
        docs = self.parser.parse(_app_event(category=""))
        cat_hints = [h for h in docs[0].graph_hints if h.predicate == "CATEGORIZED_AS"]
        assert len(cat_hints) == 0

    def test_deleted_operation(self):
        docs = self.parser.parse(_app_event(operation="deleted"))
        assert len(docs) == 1
        assert docs[0].operation == "deleted"
        assert docs[0].text == ""
        assert docs[0].graph_hints == []

    def test_missing_meta_fields(self):
        event = {
            "source_type": "macos_apps",
            "source_id": "app://unknown",
            "operation": "created",
            "meta": {},
        }
        docs = self.parser.parse(event)
        assert len(docs) == 1
        assert docs[0].node_props["name"] == ""
        assert docs[0].node_props["bundle_id"] == ""

    def test_description_from_meta(self):
        """Parser includes description in node_props and text when provided."""
        event = _app_event()
        event["meta"]["description"] = "A powerful code editor."
        docs = self.parser.parse(event)
        doc = docs[0]
        assert doc.node_props["description"] == "A powerful code editor."
        assert "A powerful code editor." in doc.text

    def test_unknown_description_excluded_from_text(self):
        """'Unknown application' descriptions are stored but not in text."""
        event = _app_event()
        event["meta"]["description"] = "Unknown application"
        docs = self.parser.parse(event)
        doc = docs[0]
        assert doc.node_props["description"] == "Unknown application"
        assert "Unknown application" not in doc.text

    def test_no_description_no_prop(self):
        """Without description in meta, node_props has no description key."""
        docs = self.parser.parse(_app_event())
        assert "description" not in docs[0].node_props

    def test_registry_registration(self):
        from worker.parsers.registry import get

        parser = get("macos_apps")
        assert isinstance(parser, MacOSAppsParser)


# ---------------------------------------------------------------------------
# HomebrewParser — Formula
# ---------------------------------------------------------------------------


class TestHomebrewFormulaParser:
    def setup_method(self):
        self.parser = HomebrewParser()

    def test_source_type(self):
        assert self.parser.source_type == "homebrew"

    def test_basic_formula_parse(self):
        docs = self.parser.parse(_formula_event())
        assert len(docs) == 1
        doc = docs[0]
        assert doc.source_type == "homebrew"
        assert doc.source_id == "brew://formula/ripgrep"
        assert doc.operation == "created"
        assert doc.node_label == "Tool"
        assert "ripgrep" in doc.text
        assert "v14.1.0" in doc.text

    def test_formula_node_props(self):
        docs = self.parser.parse(_formula_event())
        props = docs[0].node_props
        assert props["name"] == "ripgrep"
        assert props["version"] == "14.1.0"
        assert props["tap"] == "homebrew/core"
        assert props["homepage"] == "https://github.com/BurntSushi/ripgrep"

    def test_formula_provides_binaries(self):
        docs = self.parser.parse(_formula_event(binaries=["rg", "ripgrep"]))
        provides = [h for h in docs[0].graph_hints if h.predicate == "PROVIDES"]
        assert len(provides) == 2
        binary_names = {h.object_props["name"] for h in provides}
        assert binary_names == {"rg", "ripgrep"}
        for h in provides:
            assert h.subject_label == "Tool"
            assert h.object_label == "Command"
            assert h.object_merge_key == "name"

    def test_formula_no_binaries_no_provides(self):
        docs = self.parser.parse(_formula_event(binaries=[]))
        provides = [h for h in docs[0].graph_hints if h.predicate == "PROVIDES"]
        assert len(provides) == 0

    def test_formula_installed_via_hint(self):
        docs = self.parser.parse(_formula_event())
        installed = [h for h in docs[0].graph_hints if h.predicate == "INSTALLED_VIA"]
        assert len(installed) == 1
        h = installed[0]
        assert h.subject_id == "brew://formula/ripgrep"
        assert h.subject_label == "Tool"
        assert h.object_id == "source:brew_formula"
        assert h.object_label == "Source"
        assert h.object_props["name"] == "brew_formula"

    def test_formula_source_metadata(self):
        docs = self.parser.parse(_formula_event())
        meta = docs[0].source_metadata
        assert meta["package_kind"] == "formula"
        assert meta["tap"] == "homebrew/core"

    def test_formula_deleted(self):
        docs = self.parser.parse(_formula_event(operation="deleted"))
        assert len(docs) == 1
        assert docs[0].operation == "deleted"
        assert docs[0].text == ""
        assert docs[0].graph_hints == []

    def test_registry_registration(self):
        from worker.parsers.registry import get

        parser = get("homebrew")
        assert isinstance(parser, HomebrewParser)


# ---------------------------------------------------------------------------
# HomebrewParser — Cask
# ---------------------------------------------------------------------------


class TestHomebrewCaskParser:
    def setup_method(self):
        self.parser = HomebrewParser()

    def test_basic_cask_parse(self):
        docs = self.parser.parse(_cask_event())
        assert len(docs) == 1
        doc = docs[0]
        assert doc.source_type == "homebrew"
        assert doc.source_id == "brew://cask/firefox"
        assert doc.operation == "created"
        assert doc.node_label == "Application"
        assert "Firefox" in doc.text

    def test_cask_node_props(self):
        docs = self.parser.parse(_cask_event())
        props = docs[0].node_props
        assert props["name"] == "Firefox"
        assert props["version"] == "124.0"
        assert props["installed_via"] == "brew_cask"
        assert props["bundle_id"] == "org.mozilla.firefox"

    def test_cask_installed_via_hint(self):
        docs = self.parser.parse(_cask_event())
        installed = [h for h in docs[0].graph_hints if h.predicate == "INSTALLED_VIA"]
        assert len(installed) == 1
        h = installed[0]
        assert h.object_id == "source:brew_cask"
        assert h.object_label == "Source"

    def test_cask_same_as_app_hint(self):
        docs = self.parser.parse(_cask_event(bundle_id="org.mozilla.firefox"))
        same_as = [h for h in docs[0].graph_hints if h.predicate == "SAME_AS"]
        assert len(same_as) == 1
        h = same_as[0]
        assert h.subject_id == "brew://cask/firefox"
        assert h.subject_label == "Application"
        assert h.object_id == "app://org.mozilla.firefox"
        assert h.object_label == "Application"
        assert h.object_props["bundle_id"] == "org.mozilla.firefox"

    def test_cask_no_bundle_id_no_same_as(self):
        docs = self.parser.parse(_cask_event(bundle_id=""))
        same_as = [h for h in docs[0].graph_hints if h.predicate == "SAME_AS"]
        assert len(same_as) == 0

    def test_cask_deleted(self):
        docs = self.parser.parse(_cask_event(operation="deleted"))
        assert len(docs) == 1
        assert docs[0].operation == "deleted"
        assert docs[0].text == ""

    def test_cask_source_metadata(self):
        docs = self.parser.parse(_cask_event())
        meta = docs[0].source_metadata
        assert meta["package_kind"] == "cask"
        assert meta["bundle_id"] == "org.mozilla.firefox"
