import Mcp
import main
import memory
import pytest
from langchain_core.messages import HumanMessage


def test_extract_text_handles_str_and_parts():
    assert memory.extract_text("hello") == "hello"
    assert memory.extract_text([{"text": "a"}, {"text": "b"}, "c"]) == "abc"


def test_message_text():
    assert memory.message_text(HumanMessage(content="a question")) == "a question"


@pytest.mark.parametrize(
    "text,expected",
    [("hi", False), ("thanks", False), ("ok", False), ("I like tea", True),
     ("my name is Sam", True)],
)
def test_is_worth_saving(text, expected):
    assert memory.is_worth_saving(text) is expected


def test_auth_tokens_and_user_lookup(monkeypatch):
    monkeypatch.setenv("AUTH_TOKENS", '{"abc": "alice", "xyz": "bob"}')
    tokens = main.load_auth_tokens()
    assert main.user_for_token("abc", tokens) == "alice"
    assert main.user_for_token("nope", tokens) is None


def test_malformed_auth_tokens(monkeypatch):
    monkeypatch.setenv("AUTH_TOKENS", "not-json")
    with pytest.raises(RuntimeError):
        main.load_auth_tokens()


def test_read_file_requires_allowlist(monkeypatch, tmp_path):
    monkeypatch.delenv("MCP_ALLOWED_DIR", raising=False)
    with pytest.raises(PermissionError):
        Mcp.safe_local_path("anything.txt")

    monkeypatch.setenv("MCP_ALLOWED_DIR", str(tmp_path))
    (tmp_path / "ok.txt").write_text("data")
    assert Mcp.safe_local_path("ok.txt") == (tmp_path / "ok.txt").resolve()
    with pytest.raises(PermissionError):
        Mcp.safe_local_path("../outside.txt")


class FakeResponse:
    def __init__(self, status_code=200, text="body"):
        self.status_code = status_code
        self.text = text

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError("http error")


def test_read_url_host_allowlist(monkeypatch):
    def unexpected_get(*args, **kwargs):
        raise AssertionError("request must not be made")

    monkeypatch.setattr(Mcp.requests, "get", unexpected_get)
    monkeypatch.setenv("MCP_ALLOWED_HOSTS", "example.com")
    with pytest.raises(PermissionError):
        Mcp.read_url("https://evil.example.org/steal")


def test_read_url_allowed_host_and_redirect(monkeypatch):
    monkeypatch.setenv("MCP_ALLOWED_HOSTS", "example.com")
    monkeypatch.setattr(Mcp.requests, "get", lambda *a, **k: FakeResponse(text="page"))
    assert Mcp.read_url("https://example.com/page") == "page"

    monkeypatch.setattr(Mcp.requests, "get", lambda *a, **k: FakeResponse(302))
    with pytest.raises(PermissionError):
        Mcp.read_url("https://example.com/page")
