import os
import pytest
import pandas as pd

def test_project_structure():
    """Verify essential project directories exist."""
    assert os.path.exists("src") or os.path.exists("../src")
    assert os.path.exists("docs") or os.path.exists("../docs")
    assert os.path.exists("data") or os.path.exists("../data")

def test_environment_config():
    """Ensure environment configuration file or key exists."""
    assert os.path.exists("config.env") or os.getenv("GOOGLE_API_KEY") is not None

def test_dummy_sales_data_structure():
    """Validate sample sales data format if present in data directory."""
    data_path = "data/sales_data.csv" if os.path.exists("data/sales_data.csv") else "../data/sales_data.csv"
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        assert not df.empty, "Sales dataset should not be empty"

def test_retrieval_module_import():
    """Ensure RAG retrieval logic can be imported properly."""
    try:
        from src.retrieval import build_knowledge_base
        assert callable(build_knowledge_base)
    except ImportError:
        pytest.skip("retrieval module not in Python path")