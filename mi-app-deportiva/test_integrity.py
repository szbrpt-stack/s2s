from fastapi.testclient import TestClient

from integrity_app import _wilson_interval, app


def test_wilson_interval_is_bounded_and_contains_observed_rate():
    interval = _wilson_interval(63, 100)
    assert interval is not None
    assert 0.0 <= interval['low'] <= 0.63 <= interval['high'] <= 1.0
    assert _wilson_interval(0, 0) is None


def test_health_and_integrity_endpoints():
    with TestClient(app) as client:
        health = client.get('/health')
        assert health.status_code == 200

        runtime = client.get('/api/v1/runtime/integrity')
        assert runtime.status_code == 200
        runtime_payload = runtime.json()
        assert runtime_payload['engine_version']
        assert runtime_payload['model_version']
        assert runtime_payload['status'] in {'ok', 'degraded'}

        model = client.get('/api/v1/model/integrity')
        assert model.status_code == 200
        model_payload = model.json()
        assert model_payload['model_version']
        assert 'reliability_gates' in model_payload

        empirical = client.get('/api/v1/model/empirical-integrity')
        assert empirical.status_code == 200
        empirical_payload = empirical.json()
        assert 'available' in empirical_payload
