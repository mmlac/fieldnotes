"""Tests for the app description generator and cache."""

import json
from pathlib import Path
from unittest.mock import MagicMock


from worker.models.base import CompletionResponse
from worker.pipeline.app_describer import (
    AppDescriptionCache,
    AppInfo,
    BATCH_SIZE,
    _major_version,
    _parse_response,
    describe_apps,
)


# ---------------------------------------------------------------------------
# AppDescriptionCache
# ---------------------------------------------------------------------------


class TestAppDescriptionCache:
    def test_cache_miss_returns_none(self, tmp_path: Path):
        cache = AppDescriptionCache(tmp_path / "cache.json")
        assert cache.get("com.example.App", "App", "1.0") is None

    def test_put_and_get(self, tmp_path: Path):
        cache = AppDescriptionCache(tmp_path / "cache.json")
        cache.put("com.example.App", "App", "1.0", "A test app.")
        assert cache.get("com.example.App", "App", "1.0") == "A test app."

    def test_persist_and_reload(self, tmp_path: Path):
        path = tmp_path / "cache.json"
        cache = AppDescriptionCache(path)
        cache.put("com.example.App", "App", "1.0", "A test app.")
        cache.save()

        cache2 = AppDescriptionCache(path)
        assert cache2.get("com.example.App", "App", "1.0") == "A test app."

    def test_major_version_bump_invalidates(self, tmp_path: Path):
        cache = AppDescriptionCache(tmp_path / "cache.json")
        cache.put("com.example.App", "App", "1.0", "A test app.")
        # Same major version — cache hit
        assert cache.get("com.example.App", "App", "1.5") == "A test app."
        # Major version bump — cache miss
        assert cache.get("com.example.App", "App", "2.0") is None

    def test_unknown_app_regenerates_on_name_change(self, tmp_path: Path):
        cache = AppDescriptionCache(tmp_path / "cache.json")
        cache.put("com.example.App", "Old Name", "1.0", "Unknown application")
        # Same name — return cached "Unknown application"
        assert cache.get("com.example.App", "Old Name", "1.0") == "Unknown application"
        # Name changed — regenerate
        assert cache.get("com.example.App", "New Name", "1.0") is None

    def test_corrupt_cache_starts_fresh(self, tmp_path: Path):
        path = tmp_path / "cache.json"
        path.write_text("{invalid json")
        cache = AppDescriptionCache(path)
        assert cache.get("com.example.App", "App", "1.0") is None

    def test_creates_parent_dirs(self, tmp_path: Path):
        path = tmp_path / "deep" / "nested" / "cache.json"
        cache = AppDescriptionCache(path)
        cache.put("com.example.App", "App", "1.0", "A test app.")
        cache.save()
        assert path.exists()


# ---------------------------------------------------------------------------
# _major_version
# ---------------------------------------------------------------------------


class TestMajorVersion:
    def test_normal_version(self):
        assert _major_version("1.2.3") == "1"

    def test_empty_version(self):
        assert _major_version("") == ""

    def test_single_component(self):
        assert _major_version("42") == "42"


# ---------------------------------------------------------------------------
# _parse_response
# ---------------------------------------------------------------------------


class TestParseResponse:
    def _make_apps(self):
        return [
            AppInfo("com.docker.Docker", "Docker", "", "4.0"),
            AppInfo("com.apple.Safari", "Safari", "", "17.0"),
        ]

    def test_parse_tool_call_response(self):
        resp = CompletionResponse(
            text="",
            tool_calls=[
                {
                    "function": {
                        "name": "submit_descriptions",
                        "arguments": {
                            "descriptions": [
                                {
                                    "bundle_id": "com.docker.Docker",
                                    "description": "Container platform.",
                                },
                                {
                                    "bundle_id": "com.apple.Safari",
                                    "description": "Apple's web browser.",
                                },
                            ],
                        },
                    },
                }
            ],
            input_tokens=100,
            output_tokens=50,
            cached_tokens=0,
        )
        result = _parse_response(resp, self._make_apps())
        assert result["com.docker.Docker"] == "Container platform."
        assert result["com.apple.Safari"] == "Apple's web browser."

    def test_parse_tool_call_string_args(self):
        resp = CompletionResponse(
            text="",
            tool_calls=[
                {
                    "function": {
                        "name": "submit_descriptions",
                        "arguments": json.dumps(
                            {
                                "descriptions": [
                                    {
                                        "bundle_id": "com.docker.Docker",
                                        "description": "Container platform.",
                                    },
                                ],
                            }
                        ),
                    },
                }
            ],
            input_tokens=100,
            output_tokens=50,
            cached_tokens=0,
        )
        result = _parse_response(resp, self._make_apps())
        assert result["com.docker.Docker"] == "Container platform."

    def test_parse_text_fallback(self):
        resp = CompletionResponse(
            text=json.dumps(
                {
                    "descriptions": [
                        {
                            "bundle_id": "com.docker.Docker",
                            "description": "Container platform.",
                        },
                    ],
                }
            ),
            tool_calls=None,
            input_tokens=100,
            output_tokens=50,
            cached_tokens=0,
        )
        result = _parse_response(resp, self._make_apps())
        assert result["com.docker.Docker"] == "Container platform."

    def test_ignores_unknown_bundle_ids(self):
        resp = CompletionResponse(
            text="",
            tool_calls=[
                {
                    "function": {
                        "name": "submit_descriptions",
                        "arguments": {
                            "descriptions": [
                                {
                                    "bundle_id": "com.unknown.Fake",
                                    "description": "Injected.",
                                },
                            ],
                        },
                    },
                }
            ],
            input_tokens=100,
            output_tokens=50,
            cached_tokens=0,
        )
        result = _parse_response(resp, self._make_apps())
        assert result == {}

    def test_unparseable_response(self):
        resp = CompletionResponse(
            text="I don't know",
            tool_calls=None,
            input_tokens=100,
            output_tokens=50,
            cached_tokens=0,
        )
        result = _parse_response(resp, self._make_apps())
        assert result == {}


# ---------------------------------------------------------------------------
# describe_apps (integration with mock LLM)
# ---------------------------------------------------------------------------


class TestDescribeApps:
    def _make_registry(self, descriptions: dict[str, str]):
        """Create a mock registry that returns descriptions via tool call."""
        model = MagicMock()
        model.complete.return_value = CompletionResponse(
            text="",
            tool_calls=[
                {
                    "function": {
                        "name": "submit_descriptions",
                        "arguments": {
                            "descriptions": [
                                {"bundle_id": bid, "description": desc}
                                for bid, desc in descriptions.items()
                            ],
                        },
                    },
                }
            ],
            input_tokens=100,
            output_tokens=50,
            cached_tokens=0,
        )
        registry = MagicMock()
        registry.for_role.return_value = model
        return registry

    def test_all_cached(self, tmp_path: Path):
        cache = AppDescriptionCache(tmp_path / "cache.json")
        cache.put("com.docker.Docker", "Docker", "4.0", "Container platform.")

        registry = MagicMock()
        apps = [AppInfo("com.docker.Docker", "Docker", "", "4.0")]
        result = describe_apps(apps, registry, cache)

        assert result["com.docker.Docker"] == "Container platform."
        registry.for_role.assert_not_called()

    def test_llm_called_for_cache_miss(self, tmp_path: Path):
        cache = AppDescriptionCache(tmp_path / "cache.json")
        registry = self._make_registry({"com.docker.Docker": "Container platform."})

        apps = [AppInfo("com.docker.Docker", "Docker", "", "4.0")]
        result = describe_apps(apps, registry, cache)

        assert result["com.docker.Docker"] == "Container platform."
        registry.for_role.assert_called()

    def test_results_cached_after_llm(self, tmp_path: Path):
        cache = AppDescriptionCache(tmp_path / "cache.json")
        registry = self._make_registry({"com.docker.Docker": "Container platform."})

        apps = [AppInfo("com.docker.Docker", "Docker", "", "4.0")]
        describe_apps(apps, registry, cache)

        # Verify cached
        assert cache.get("com.docker.Docker", "Docker", "4.0") == "Container platform."

    def test_mixed_cached_and_new(self, tmp_path: Path):
        cache = AppDescriptionCache(tmp_path / "cache.json")
        cache.put("com.cached.App", "Cached", "1.0", "Already cached.")

        registry = self._make_registry({"com.new.App": "New app description."})

        apps = [
            AppInfo("com.cached.App", "Cached", "", "1.0"),
            AppInfo("com.new.App", "New", "", "1.0"),
        ]
        result = describe_apps(apps, registry, cache)

        assert result["com.cached.App"] == "Already cached."
        assert result["com.new.App"] == "New app description."

    def test_llm_failure_skips_app(self, tmp_path: Path):
        cache = AppDescriptionCache(tmp_path / "cache.json")
        registry = MagicMock()
        model = MagicMock()
        model.complete.side_effect = RuntimeError("LLM down")
        registry.for_role.return_value = model

        apps = [AppInfo("com.docker.Docker", "Docker", "", "4.0")]
        result = describe_apps(apps, registry, cache)

        # LLM failure should NOT produce a result or cache "Unknown application"
        assert "com.docker.Docker" not in result
        assert cache.get("com.docker.Docker", "Docker", "4.0") is None

    def test_no_model_configured(self, tmp_path: Path):
        cache = AppDescriptionCache(tmp_path / "cache.json")
        registry = MagicMock()
        registry.for_role.side_effect = KeyError("no model")

        apps = [AppInfo("com.docker.Docker", "Docker", "", "4.0")]
        result = describe_apps(apps, registry, cache)

        # No model means no descriptions — app should be absent from results
        assert "com.docker.Docker" not in result

    def test_batching(self, tmp_path: Path):
        cache = AppDescriptionCache(tmp_path / "cache.json")

        # Create more apps than BATCH_SIZE
        apps = [
            AppInfo(f"com.test.App{i}", f"App{i}", "", "1.0")
            for i in range(BATCH_SIZE + 3)
        ]

        descs = {app.bundle_id: f"Description for {app.display_name}" for app in apps}
        registry = self._make_registry(descs)

        result = describe_apps(apps, registry, cache)

        # Should have called LLM twice (BATCH_SIZE + 3 remaining)
        model = registry.for_role.return_value
        assert model.complete.call_count == 2
        assert len(result) == len(apps)
