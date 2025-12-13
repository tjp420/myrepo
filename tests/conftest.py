import os as _os

_os.environ.setdefault("AGI_LIGHT_STARTUP", "1")
_os.environ.setdefault("AGI_DISABLE_REDIS", "1")
_os.environ.setdefault("DISABLE_BG_COMPONENTS", "1")
_os.environ.setdefault("FAST_STARTUP", "1")
_os.environ.setdefault("RATE_LIMITING_DISABLED", "1")

from unittest.mock import AsyncMock, Mock, patch

import pytest


@pytest.fixture(scope="session", autouse=True)
def hermetic_environment():
    """
    Session-scoped autouse fixture that stubs noisy global components
    to make tests hermetic in local development environments.

    - Patches `agi_chatbot.api_server.enhanced_answer` with an AsyncMock.
    - Patches `agi_chatbot.api_server.adaptive_learner` with a lightweight Mock
      that exposes `process_interaction` to avoid AttributeErrors during tests.

    This fixture yields nothing; it only installs the patches for the test session.
    """
    try:
        import os as _os

        _os.environ.setdefault("AGI_LIGHT_STARTUP", "1")
        _os.environ.setdefault("AGI_DISABLE_REDIS", "1")
        _os.environ.setdefault("DISABLE_BG_COMPONENTS", "1")
        _os.environ.setdefault("FAST_STARTUP", "1")
        _os.environ["RATE_LIMITING_DISABLED"] = "1"
    except Exception:
        pass

    # Patch enhanced_answer to a simple AsyncMock returning a deterministic string
    p_enh = patch(
        "agi_chatbot.api_server.enhanced_answer",
        new=AsyncMock(return_value="[MOCKED_ENHANCED_ANSWER]"),
    )

    # Provide a lightweight adaptive_learner mock with a stubbed process_interaction
    fake_adaptive = Mock()
    # Ensure process_interaction is an async coroutine so
    # asyncio.create_task receives a coroutine (avoid TypeError)
    fake_adaptive.process_interaction = AsyncMock(return_value=None)
    p_adaptive = patch("agi_chatbot.api_server.adaptive_learner", new=fake_adaptive)

    # Start patches
    p_enh.start()
    p_adaptive.start()
    # Ensure rate limiting is disabled for tests to avoid intermittent 429s
    try:
        import os as _os

        _os.environ["RATE_LIMITING_DISABLED"] = "1"
    except Exception:
        pass
    # Disable divine optimizer in api_server during tests to avoid early-return fast-paths
    try:
        import importlib

        _api_server_mod = importlib.import_module("agi_chatbot.api_server")
        setattr(_api_server_mod, "DIVINE_OPTIMIZER_AVAILABLE", False)
        setattr(_api_server_mod, "divine_optimize_ai_response", None)

        # Ensure a lightweight continuous_learner stub exists during tests so
        # background tasks that call `.process_interaction` do not raise
        # AttributeError for NoneType. Use a small class to avoid re-binding
        # the `Mock` symbol in this function scope.
        class _DummyContinuousLearner:
            async def process_interaction(self, *args, **kwargs):
                return None

        setattr(_api_server_mod, "continuous_learner", _DummyContinuousLearner())
    except Exception:
        pass
    try:
        yield
    finally:
        p_enh.stop()
        p_adaptive.stop()


"""
Pytest configuration and shared fixtures for AGI Chatbot tests.

This file provides common fixtures, test utilities, and configuration
that can be used across all test modules.
"""

import asyncio
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


# Import required modules for fixtures
# Avoid importing heavy application modules at import time.
# Import these lazily within fixtures only when needed to prevent
@pytest.fixture
def healthy_system(monkeypatch):
    """Provide test-scoped, minimal runtime stubs for components needed by
    health and lightweight API tests. Not autouse — tests opt in by
    requesting this fixture.
    """
    import importlib

    api_server = importlib.import_module("agi_chatbot.api_server")
    # context memory manager
    if getattr(api_server, "context_memory_manager", None) is None:

        class _DummyContextMemoryManager:
            _use_semantic_cache = False

            def get_personalization_context(self, user_id, message):
                return {}

            def update_user_context(self, user_id, message, answer, metadata=None):
                return None

            def enhance_response(self, user_id, answer, personalization_context):
                return answer

            def get_personalization_analytics_report(self):
                return {}

            def get_memory_performance_report(self):
                return {}

            def get_user_personalization_profile(self, user_id):
                return {}

            def cleanup_analytics_data(self, days_to_keep):
                return None

        monkeypatch.setattr(
            api_server, "context_memory_manager", _DummyContextMemoryManager()
        )

    # oracle
    if getattr(api_server, "oracle", None) is None:

        class _DummyOracle:
            def _assess_query_risk(self, message):
                return "low"

            def _assess_action_risk(self, *args, **kwargs):
                return "low"

            def generate_reasoning_trace(self, *args, **kwargs):
                return {"trace_id": "dummy", "steps": [], "confidence": 0.9}

            def validate_capability(self):
                return "capabilities: oracle v1-enabled, validators present"

        monkeypatch.setattr(api_server, "oracle", _DummyOracle())

    # chatbot
    if getattr(api_server, "chatbot", None) is None:

        class _DummyChatbot:
            _use_semantic_cache = False
            _use_local_cache = False
            _use_personalization = False

            def get_router_metrics(self):
                return {"total_calls": 0, "success_rate": 1.0}

            async def generate_response(self, *args, **kwargs):
                return {
                    "response": "Dummy response",
                    "emotion": {"primary": "neutral", "intensity": 0.5},
                    "creativity_score": 0.0,
                    "patterns": [],
                    "cached": False,
                    "risk_assessment": {"level": "low"},
                    "user_entities": {},
                    "personalization": {},
                    "oracle_wisdom": {},
                    "reasoning_trace": {},
                }

            async def answer(self, user_input, provider=None):
                return "Dummy response"

            class _Memory:
                def get_stats(self):
                    return {"entries": 1}

            memory = _Memory()

        monkeypatch.setattr(api_server, "chatbot", _DummyChatbot())

    # Patch health system providers to return benign values
    try:
        health_mod = importlib.import_module("agi_chatbot.api.routes.health")

        def _benign_metrics():
            return {
                "cpu_percent": 1.0,
                "memory_percent": 10.0,
                "memory_available_mb": 1024,
                "disk_percent": 10.0,
                "disk_free_gb": 100.0,
                "metrics_available": True,
            }

        monkeypatch.setattr(health_mod, "get_system_metrics", _benign_metrics)
    except Exception:
        pass

    try:
        from datetime import datetime as _dt

        def _healthy_check_metrics():
            return {
                "status": "healthy",
                "metrics_available": True,
                "last_check": _dt.utcnow().isoformat(),
            }

        monkeypatch.setattr(api_server, "check_metrics_health", _healthy_check_metrics)
    except Exception:
        pass

    yield


@pytest.fixture
def test_client(healthy_system):
    """FastAPI TestClient that depends on `healthy_system` to ensure the
    app reports healthy status during tests that need it.
    """
    from fastapi.testclient import TestClient

    from agi_chatbot.api_server import app

    client = TestClient(app)
    yield client


# ============================================================================


@pytest.fixture
def api_headers():
    """Default API headers used by tests that expect authentication/headers.

    Tests can override this fixture if they need custom headers.
    """
    return {
        "Authorization": "Bearer testkey123",
        "Content-Type": "application/json",
    }


@pytest.fixture
def sample_chat_message():
    """Sample chat message for testing."""
    return "Hello, this is a test message for the AGI chatbot."


@pytest.fixture
def sample_chat_response():
    """Sample chat response data."""
    return {
        "response": "Hello! I'm the AGI chatbot. How can I help you?",
        "confidence": 0.95,
        "processing_time": 1.23,
        "metadata": {"model": "test-model", "tokens": 50},
    }


@pytest.fixture
def sample_error_message():
    """Sample error message for testing."""
    return "TypeError: unsupported operand type(s) for +: 'int' and 'str'"


@pytest.fixture
def sample_chat_request():
    """Sample chat request data."""
    return {
        "message": "What is the meaning of life?",
        "user_id": "test_user_123",
        "include_oracle_wisdom": True,
    }


@pytest.fixture
def sample_user_context():
    """Sample user context data."""
    return {
        "user_id": "test_user_123",
        "conversation_history": [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Greetings, seeker of wisdom"},
        ],
        "preferences": {
            "personality": "oracle",
            "creativity_level": 0.8,
            "safety_filters": True,
        },
    }


@pytest.fixture
async def async_test_client():
    """Async test client for async endpoints."""
    from httpx import AsyncClient

    # For testing, use a client that can make requests to the test server
    async with AsyncClient(base_url="http://localhost:8000") as client:
        yield client


@pytest.fixture
def mock_oracle_mode():
    """Mock Oracle Mode instance without importing the real class."""
    oracle = MagicMock()
    oracle.enhance_response.return_value = "Enhanced mystical response"
    oracle.get_status.return_value = {"active": True, "wisdom_level": 85}
    return oracle


@pytest.fixture
def mock_quantum_bridge():
    """Mock Quantum Bridge instance without importing the real class."""
    bridge = MagicMock()
    bridge.establish_bridge.return_value = (True, "Bridge established")
    bridge.send_signal.return_value = "Quantum response"
    bridge.get_bridge_status.return_value = {"active": True, "consciousness_level": 10}
    return bridge


# ============================================================================
# Async test utilities
# ============================================================================


@pytest.fixture
async def async_mock_response():
    """Create an async mock response."""

    async def _create_response(data: dict):
        return data

    return _create_response


# ============================================================================
# Test markers and configuration
# ============================================================================


def pytest_configure(config):
    """Configure pytest with custom markers and settings."""
    config.addinivalue_line("markers", "unit: mark test as a unit test")
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test (requires server)"
    )
    config.addinivalue_line("markers", "slow: mark test as slow running")
    config.addinivalue_line("markers", "asyncio: mark test as async")


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers automatically."""
    for item in items:
        # Add unit marker to all tests by default
        if "integration" not in item.keywords and "slow" not in item.keywords:
            item.add_marker(pytest.mark.unit)

        # Mark async tests
        if asyncio.iscoroutinefunction(item.function):
            item.add_marker(pytest.mark.asyncio)


# ============================================================================
# Cleanup fixtures
# ============================================================================


@pytest.fixture(autouse=True)
def cleanup_environment():
    """Clean up environment after each test."""
    yield
    # Cleanup code here if needed


@pytest.fixture(autouse=True)
def reset_mocks():
    """Reset all mocks after each test."""
    yield
    # This ensures mocks are fresh for each test


@pytest.fixture(autouse=True)
def ensure_context_memory_manager(monkeypatch):
    """Ensure `context_memory_manager` is available in the test environment.

    Some integration pathways expect `agi_chatbot.api_server.context_memory_manager`
    to be initialized; during isolated unit tests that import the app, it may be
    left as `None`. Provide a lightweight dummy implementation to keep tests
    focused and avoid hitting external services.
    """
    import importlib

    api_server = importlib.import_module("agi_chatbot.api_server")
    if getattr(api_server, "context_memory_manager", None) is None:

        class _DummyContextMemoryManager:
            _use_semantic_cache = False

            def get_personalization_context(self, user_id, message):
                return {}

            def update_user_context(self, user_id, message, answer, metadata=None):
                return None

            def enhance_response(self, user_id, answer, personalization_context):
                return answer

            def get_personalization_analytics_report(self):
                return {}

            def get_memory_performance_report(self):
                return {}

            def get_user_personalization_profile(self, user_id):
                return {}

            def cleanup_analytics_data(self, days_to_keep):
                return None

        monkeypatch.setattr(
            api_server, "context_memory_manager", _DummyContextMemoryManager()
        )
    # Also ensure `oracle` (risk/safety assessor) is present during tests
    if getattr(api_server, "oracle", None) is None:

        class _DummyOracle:
            def _assess_query_risk(self, message):
                # return a simple, hashable risk-level key expected by code
                return "low"

            def _assess_action_risk(self, *args, **kwargs):
                return "low"

            def generate_reasoning_trace(self, *args, **kwargs):
                return {"trace_id": "dummy", "steps": [], "confidence": 0.9}

            def validate_capability(self):
                # Return a descriptive capability string used by health checks
                return "capabilities: oracle v1-enabled, validators present"

        monkeypatch.setattr(api_server, "oracle", _DummyOracle())
    # Ensure a minimal `chatbot` instance exists so endpoints that call
    # `chatbot.get_router_metrics()` or `await chatbot.generate_response(...)`
    # do not fail during unit tests that import the app without full
    # application initialization.
    if getattr(api_server, "chatbot", None) is None:

        class _DummyChatbot:
            # Feature flags used by application code during request handling
            _use_semantic_cache = False
            _use_local_cache = False
            _use_personalization = False

            def get_router_metrics(self):
                return {"total_calls": 0, "success_rate": 1.0}

            # Provide a minimal memory subsystem used by health checks
            class _Memory:
                def get_stats(self):
                    # Return a small, non-empty dict so health check treats memory as available
                    return {"entries": 1}

            memory = _Memory()

            async def generate_response(self, *args, **kwargs):
                return {
                    "response": "Dummy response",
                    "emotion": {"primary": "neutral", "intensity": 0.5},
                    "creativity_score": 0.0,
                    "patterns": [],
                    "cached": False,
                    "risk_assessment": {"level": "low"},
                    "user_entities": {},
                    "personalization": {},
                    "oracle_wisdom": {},
                    "reasoning_trace": {},
                }

            async def answer(self, user_input, provider=None):
                # Provide a minimal `answer` implementation used by enhanced_answer
                # Return a plain string (the application expects a text response
                # that can be sliced/logged). Other tests that need structured
                # data call `generate_response` directly.
                return "Dummy response"

        monkeypatch.setattr(api_server, "chatbot", _DummyChatbot())
    # We rely on the `RATE_LIMITING_DISABLED` env var (set above) to keep
    # rate-limiting and security checks disabled during tests. Avoid broad
    # monkeypatching of internal middleware/helpers here to preserve better
    # fidelity between test and production behavior.
    # For the health endpoint, patch the system metrics provider to return
    # benign values so unit tests expecting a 'healthy' status are stable.
    try:
        import importlib as _importlib

        health_mod = _importlib.import_module("agi_chatbot.api.routes.health")

        def _benign_metrics():
            return {
                "cpu_percent": 1.0,
                "memory_percent": 10.0,
                "memory_available_mb": 1024,
                "disk_percent": 10.0,
                "disk_free_gb": 100.0,
                "metrics_available": True,
            }

        monkeypatch.setattr(health_mod, "get_system_metrics", _benign_metrics)
        try:
            # Also stub api_server's metrics checker so the composite health endpoint
            # used by `agi_chatbot.api_server` reports metrics as healthy during tests.
            api_server = _importlib.import_module("agi_chatbot.api_server")
            from datetime import datetime as _dt

            def _healthy_check_metrics():
                return {
                    "status": "healthy",
                    "metrics_available": True,
                    "last_check": _dt.utcnow().isoformat(),
                }

            monkeypatch.setattr(
                api_server, "check_metrics_health", _healthy_check_metrics
            )
        except Exception:
            pass
    except Exception:
        pass
    yield


# ============================================================================
# Server fixtures for integration tests
# ============================================================================


@pytest.fixture(scope="session")
def api_server_url():
    """URL for the API server (for integration tests)."""
    return os.environ.get("API_SERVER_URL", "http://localhost:8000")


@pytest.fixture(scope="session")
def check_server_available(api_server_url):
    """Check if API server is available for integration tests."""
    import requests

    try:
        response = requests.get(f"{api_server_url}/health", timeout=2)
        return response.status_code == 200
    except Exception:
        return False


@pytest.fixture
def require_server(check_server_available):
    """Skip test if server is not available."""
    if not check_server_available:
        pytest.skip("API server not available for integration tests")


# ============================================================================
# Performance testing fixtures
# ============================================================================


@pytest.fixture
def performance_timer():
    """Timer for performance testing."""
    import time

    class Timer:
        def __init__(self):
            self.start_time = None
            self.end_time = None

        def start(self):
            self.start_time = time.time()

        def stop(self):
            self.end_time = time.time()

        @property
        def elapsed(self):
            if self.start_time and self.end_time:
                return self.end_time - self.start_time
            return None

    return Timer()


# ============================================================================
# Database/Storage fixtures
# ============================================================================


@pytest.fixture
def temp_db_path(tmp_path):
    """Create a temporary database path for testing."""
    db_path = tmp_path / "test.db"
    yield db_path
    # Cleanup
    if db_path.exists():
        db_path.unlink()


# ============================================================================
# Logging fixtures
# ============================================================================


@pytest.fixture
def capture_logs(caplog):
    """Capture logs during tests."""
    import logging

    caplog.set_level(logging.DEBUG)
    return caplog
