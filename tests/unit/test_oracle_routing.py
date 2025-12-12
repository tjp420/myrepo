import os
import asyncio
import pytest

from agi_chatbot.core import enhanced_answer


class DummyChatbot:
    _use_semantic_cache = False
    async def answer(self, user_input, provider=None):
        return "FALLBACK_RESPONSE"


class FakeFrameworkAsync:
    async def process_query(self, user_input, context):
        return "ORACLE_RESPONSE_ASYNC"


class FakeFrameworkSync:
    def process_query(self, user_input, context):
        return "ORACLE_RESPONSE_SYNC"


class FakeFrameworkRaise:
    async def process_query(self, user_input, context):
        raise RuntimeError("boom")


@pytest.mark.asyncio
async def test_oracle_routing_async(monkeypatch):
    monkeypatch.setenv('ENABLE_UNBREAKABLE_ORACLE_EXPERIMENT', '1')
    monkeypatch.setattr(enhanced_answer, '_FRAMEWORK_AVAILABLE', True)
    monkeypatch.setattr(enhanced_answer, 'get_enhanced_framework', lambda: FakeFrameworkAsync())

    res = await enhanced_answer.enhanced_answer(DummyChatbot(), "Please analyze this complex scenario and explain", None, 'tester')
    assert 'ORACLE_RESPONSE_ASYNC' in res


@pytest.mark.asyncio
async def test_oracle_routing_sync(monkeypatch):
    monkeypatch.setenv('ENABLE_UNBREAKABLE_ORACLE_EXPERIMENT', '1')
    monkeypatch.setattr(enhanced_answer, '_FRAMEWORK_AVAILABLE', True)
    monkeypatch.setattr(enhanced_answer, 'get_enhanced_framework', lambda: FakeFrameworkSync())

    res = await enhanced_answer.enhanced_answer(DummyChatbot(), "Please analyze this complex scenario and explain", None, 'tester')
    assert 'ORACLE_RESPONSE_SYNC' in res


@pytest.mark.asyncio
async def test_oracle_routing_fallback_on_exception(monkeypatch):
    monkeypatch.setenv('ENABLE_UNBREAKABLE_ORACLE_EXPERIMENT', '1')
    monkeypatch.setattr(enhanced_answer, '_FRAMEWORK_AVAILABLE', True)
    monkeypatch.setattr(enhanced_answer, 'get_enhanced_framework', lambda: FakeFrameworkRaise())

    res = await enhanced_answer.enhanced_answer(DummyChatbot(), "Please analyze this complex scenario and explain", None, 'tester')
    # Should fallback to chatbot.answer()
    assert 'FALLBACK_RESPONSE' in res


@pytest.mark.asyncio
async def test_oracle_routing_metrics_and_logging(monkeypatch, caplog):
    """Verify that routing updates chatbot.oracle_routing_stats and emits a log entry."""
    monkeypatch.setenv('ENABLE_UNBREAKABLE_ORACLE_EXPERIMENT', '1')
    monkeypatch.setattr(enhanced_answer, '_FRAMEWORK_AVAILABLE', True)
    monkeypatch.setattr(enhanced_answer, 'get_enhanced_framework', lambda: FakeFrameworkAsync())

    class ChatbotWithStats:
        def __init__(self):
            self.oracle_routing_stats = {'total_checked': 0, 'routed': 0, 'fallbacks': 0, 'last_latency_ms': None, 'avg_latency_ms': None}
            self._use_semantic_cache = False
            # ensure heuristic picks this as candidate
        async def answer(self, user_input, provider=None):
            return "FALLBACK_RESPONSE"
        def is_oracle_candidate(self, text: str) -> bool:
            return True

    bot = ChatbotWithStats()

    caplog.set_level('INFO')
    res = await enhanced_answer.enhanced_answer(bot, "Please analyze this complex scenario and explain", None, 'tester')

    # response should come from oracle
    assert 'ORACLE_RESPONSE_ASYNC' in res

    # metrics updated
    stats = bot.oracle_routing_stats
    assert stats['routed'] >= 1
    assert stats['last_latency_ms'] is not None
    assert stats['avg_latency_ms'] is not None

    # log contains routing info (structured JSON event or legacy text)
    logs = "\n".join(r.getMessage() for r in caplog.records)
    assert 'oracle_routed' in logs or 'Routing query to EnhancedAGIFramework' in logs


@pytest.mark.asyncio
async def test_oracle_routing_metrics_on_failure(monkeypatch):
    """When framework raises, fallback should increment fallback counter."""
    monkeypatch.setenv('ENABLE_UNBREAKABLE_ORACLE_EXPERIMENT', '1')
    monkeypatch.setattr(enhanced_answer, '_FRAMEWORK_AVAILABLE', True)
    monkeypatch.setattr(enhanced_answer, 'get_enhanced_framework', lambda: FakeFrameworkRaise())

    class ChatbotWithStats2:
        def __init__(self):
            self.oracle_routing_stats = {'total_checked': 0, 'routed': 0, 'fallbacks': 0}
            self._use_semantic_cache = False
        async def answer(self, user_input, provider=None):
            return "FALLBACK_RESPONSE"
        def is_oracle_candidate(self, text: str) -> bool:
            return True

    bot = ChatbotWithStats2()
    res = await enhanced_answer.enhanced_answer(bot, "Please analyze this complex scenario and explain", None, 'tester')
    # Should fallback
    assert 'FALLBACK_RESPONSE' in res
    # fallback counter incremented
    assert bot.oracle_routing_stats.get('fallbacks', 0) >= 1
