"""Tests for backward compatibility with the crabpath name."""
import warnings


def test_import_crabpath_emits_warning():
    """Test warning on legacy import."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        import crabpath  # noqa: F401
        dep_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
        assert len(dep_warnings) >= 1
        assert "openclawbrain" in str(dep_warnings[0].message)


def test_crabpath_exports_match():
    """Test that legacy exports match public API."""
    import openclawbrain

    import crabpath  # noqa: F811

    for name in openclawbrain.__all__:
        assert hasattr(crabpath, name), f"crabpath shim missing: {name}"
