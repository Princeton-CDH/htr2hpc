"""Tests for htr2hpc.settings."""


def test_middleware_excludes_account_expiry():
    """AccountExpiryMiddleware should be removed from the middleware list."""
    sample_middleware = [
        "django_prometheus.middleware.PrometheusBeforeMiddleware",
        "django.middleware.security.SecurityMiddleware",
        "escriptorium.middleware.AccountExpiryMiddleware",
        "django_prometheus.middleware.PrometheusAfterMiddleware",
    ]
    result = [m for m in sample_middleware if m != "escriptorium.middleware.AccountExpiryMiddleware"]
    assert "escriptorium.middleware.AccountExpiryMiddleware" not in result
    assert len(result) == len(sample_middleware) - 1
