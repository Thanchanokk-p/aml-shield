"""
Minimal smoke test — checks that the API module loads without errors
and that core functions exist. Runs in CI before deployment.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_api_module_imports():
    """API module must import without raising exceptions."""
    from src import api
    assert api.app is not None


def test_predict_endpoint_exists():
    """The /predict route must be registered on the FastAPI app."""
    from src import api
    routes = [route.path for route in api.app.routes]
    assert "/predict" in routes


def test_health_endpoint_exists():
    """The /health route must be registered on the FastAPI app."""
    from src import api
    routes = [route.path for route in api.app.routes]
    assert "/health" in routes
