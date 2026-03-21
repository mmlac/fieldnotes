"""Tests for Ollama base_url SSRF validation."""

import pytest

from worker.models.providers.ollama import _validate_ollama_url, OllamaProvider


class TestValidateOllamaUrl:
    """Validate that user-supplied URLs are checked against all private ranges."""

    def test_accepts_public_ip(self) -> None:
        assert (
            _validate_ollama_url("http://203.0.113.1:11434")
            == "http://203.0.113.1:11434"
        )

    def test_accepts_https(self) -> None:
        assert (
            _validate_ollama_url("https://203.0.113.1:11434")
            == "https://203.0.113.1:11434"
        )

    # --- scheme checks ---

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

    def test_rejects_unresolvable_host(self) -> None:
        with pytest.raises(ValueError, match="Cannot resolve"):
            _validate_ollama_url("http://this-host-does-not-exist-xyzzy.invalid")

    # --- RFC 1918 ---

    def test_rejects_10_network(self) -> None:
        with pytest.raises(ValueError, match="blocked address"):
            _validate_ollama_url("http://10.0.0.1:11434")

    def test_rejects_172_16_network(self) -> None:
        with pytest.raises(ValueError, match="blocked address"):
            _validate_ollama_url("http://172.16.0.1:11434")

    def test_rejects_192_168_network(self) -> None:
        with pytest.raises(ValueError, match="blocked address"):
            _validate_ollama_url("http://192.168.1.1:11434")

    # --- loopback ---

    def test_rejects_loopback(self) -> None:
        with pytest.raises(ValueError, match="blocked address"):
            _validate_ollama_url("http://127.0.0.1:11434")

    def test_rejects_loopback_alt(self) -> None:
        with pytest.raises(ValueError, match="blocked address"):
            _validate_ollama_url("http://127.0.0.2:8080")

    # --- link-local / metadata ---

    def test_rejects_metadata_ip(self) -> None:
        with pytest.raises(ValueError, match="blocked address"):
            _validate_ollama_url("http://169.254.169.254")

    def test_rejects_link_local(self) -> None:
        with pytest.raises(ValueError, match="blocked address"):
            _validate_ollama_url("http://169.254.0.1")

    # --- IPv4-mapped IPv6 ---

    def test_rejects_ipv4_mapped_loopback(self) -> None:
        with pytest.raises(ValueError, match="blocked address"):
            _validate_ollama_url("http://[::ffff:127.0.0.1]:11434")

    def test_rejects_ipv4_mapped_metadata(self) -> None:
        with pytest.raises(ValueError, match="blocked address"):
            _validate_ollama_url("http://[::ffff:169.254.169.254]")

    def test_rejects_ipv4_mapped_rfc1918(self) -> None:
        with pytest.raises(ValueError, match="blocked address"):
            _validate_ollama_url("http://[::ffff:10.0.0.1]:11434")

    # --- CGNAT ---

    def test_rejects_cgnat(self) -> None:
        with pytest.raises(ValueError, match="blocked address"):
            _validate_ollama_url("http://100.64.0.1:11434")

    def test_rejects_cgnat_upper(self) -> None:
        with pytest.raises(ValueError, match="blocked address"):
            _validate_ollama_url("http://100.127.255.254:11434")


class TestOllamaProviderConfigure:
    """Test configure() applies validation only to user-supplied URLs."""

    def test_rejects_bad_scheme_via_configure(self) -> None:
        provider = OllamaProvider()
        with pytest.raises(ValueError, match="http or https"):
            provider.configure({"base_url": "gopher://evil.example.com"})

    def test_rejects_metadata_via_configure(self) -> None:
        provider = OllamaProvider()
        with pytest.raises(ValueError, match="blocked address"):
            provider.configure({"base_url": "http://169.254.169.254/latest/meta-data"})

    def test_rejects_rfc1918_via_configure(self) -> None:
        provider = OllamaProvider()
        with pytest.raises(ValueError, match="blocked address"):
            provider.configure({"base_url": "http://10.0.0.1:11434"})

    def test_default_url_passes_without_validation(self) -> None:
        """Built-in default (localhost) is trusted and bypasses validation."""
        provider = OllamaProvider()
        provider.configure({})
        assert provider._base_url == "http://localhost:11434"

    def test_explicit_default_url_passes(self) -> None:
        """Explicitly supplying the default localhost URL must not be rejected."""
        provider = OllamaProvider()
        provider.configure({"base_url": "http://localhost:11434"})
        assert provider._base_url == "http://localhost:11434"

    def test_explicit_default_url_with_trailing_slash_passes(self) -> None:
        """Trailing slash on the default URL is stripped and accepted."""
        provider = OllamaProvider()
        provider.configure({"base_url": "http://localhost:11434/"})
        assert provider._base_url == "http://localhost:11434"
