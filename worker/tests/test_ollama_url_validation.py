"""Tests for Ollama base_url SSRF validation."""

import pytest

from worker.models.providers.ollama import _validate_ollama_url, OllamaProvider


class TestValidateOllamaUrl:
    def test_accepts_localhost_http(self) -> None:
        assert _validate_ollama_url("http://localhost:11434") == "http://localhost:11434"

    def test_accepts_localhost_https(self) -> None:
        assert _validate_ollama_url("https://localhost:11434") == "https://localhost:11434"

    def test_accepts_127_0_0_1(self) -> None:
        assert _validate_ollama_url("http://127.0.0.1:11434") == "http://127.0.0.1:11434"

    def test_rejects_ftp_scheme(self) -> None:
        with pytest.raises(ValueError, match="http or https"):
            _validate_ollama_url("ftp://localhost:11434")

    def test_rejects_file_scheme(self) -> None:
        with pytest.raises(ValueError, match="http or https"):
            _validate_ollama_url("file:///etc/passwd")

    def test_rejects_no_scheme(self) -> None:
        with pytest.raises(ValueError, match="http or https"):
            _validate_ollama_url("localhost:11434")

    def test_rejects_empty_hostname(self) -> None:
        with pytest.raises(ValueError, match="no hostname"):
            _validate_ollama_url("http://")

    def test_rejects_metadata_ip(self) -> None:
        with pytest.raises(ValueError, match="blocked address"):
            _validate_ollama_url("http://169.254.169.254")

    def test_rejects_link_local(self) -> None:
        with pytest.raises(ValueError, match="blocked address"):
            _validate_ollama_url("http://169.254.0.1")

    def test_rejects_unresolvable_host(self) -> None:
        with pytest.raises(ValueError, match="Cannot resolve"):
            _validate_ollama_url("http://this-host-does-not-exist-xyzzy.invalid")


class TestOllamaProviderConfigure:
    def test_rejects_bad_scheme_via_configure(self) -> None:
        provider = OllamaProvider()
        with pytest.raises(ValueError, match="http or https"):
            provider.configure({"base_url": "gopher://evil.example.com"})

    def test_rejects_metadata_via_configure(self) -> None:
        provider = OllamaProvider()
        with pytest.raises(ValueError, match="blocked address"):
            provider.configure({"base_url": "http://169.254.169.254/latest/meta-data"})

    def test_default_url_passes_validation(self) -> None:
        provider = OllamaProvider()
        provider.configure({})
        assert provider._base_url == "http://localhost:11434"
