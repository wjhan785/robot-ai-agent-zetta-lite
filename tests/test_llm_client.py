import pytest

from llm.client import DeepSeekClient, FakeLLMClient


def test_fake_client_records_calls_and_returns_response():
    client = FakeLLMClient(["hello"])
    out = client.complete(system="sys", user="usr")
    assert out == "hello"
    assert client.calls == [{"system": "sys", "user": "usr"}]


def test_fake_client_cycles_through_responses_then_repeats_last():
    client = FakeLLMClient(["first", "second"])
    assert client.complete("s", "u1") == "first"
    assert client.complete("s", "u2") == "second"
    assert client.complete("s", "u3") == "second"  # repeats last once exhausted


def test_fake_client_requires_at_least_one_response():
    with pytest.raises(ValueError):
        FakeLLMClient([])


def test_deepseek_client_requires_api_key(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        DeepSeekClient()


def test_deepseek_client_accepts_explicit_key(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    client = DeepSeekClient(api_key="sk-test")
    assert client.api_key == "sk-test"
