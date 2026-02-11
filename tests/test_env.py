import pytest

def test_env():
    try:
        import hierarchicalcausalmodels
    except ImportError:
        pytest.fail("hierarchicalcausalmodels not found")
    else:
        assert True

if __name__ == "__main__":
    pytest.main()