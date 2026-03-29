import httpx
import pytest
import pytest_asyncio

from app import app, processed_data


@pytest_asyncio.fixture
async def client():
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        yield client


@pytest.fixture
def valid_zone():
    """First available zone in processed data."""
    return processed_data.columns[0]


@pytest.fixture
def valid_datetime():
    """A datetime with enough lookback history (index row 12 guarantees 12 prior hours)."""
    return processed_data.index[12].isoformat()


class TestHealthEndpoint:
    @pytest.mark.asyncio
    async def test_health_returns_200(self, client):
        response = await client.get("/health")
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_health_response_body(self, client):
        response = await client.get("/health")
        data = response.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True
        assert data["zones_available"] > 0


class TestZonesEndpoint:
    @pytest.mark.asyncio
    async def test_zones_returns_200(self, client):
        response = await client.get("/zones")
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_zones_returns_list(self, client):
        response = await client.get("/zones")
        data = response.json()
        assert "zones" in data
        assert isinstance(data["zones"], list)
        assert len(data["zones"]) > 0


class TestPredictEndpoint:
    @pytest.mark.asyncio
    async def test_predict_returns_200(self, client, valid_zone, valid_datetime):
        payload = {"zone": valid_zone, "datetime": valid_datetime}
        response = await client.post("/predict", json=payload)
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_predict_returns_float_in_range(self, client, valid_zone, valid_datetime):
        payload = {"zone": valid_zone, "datetime": valid_datetime}
        response = await client.post("/predict", json=payload)
        data = response.json()
        assert isinstance(data["predicted_occupancy"], float)
        assert 0.0 <= data["predicted_occupancy"] <= 1.0

    @pytest.mark.asyncio
    async def test_predict_response_includes_zone_and_datetime(self, client, valid_zone, valid_datetime):
        payload = {"zone": valid_zone, "datetime": valid_datetime}
        response = await client.post("/predict", json=payload)
        data = response.json()
        assert data["zone"] == valid_zone
        assert data["datetime"] == valid_datetime

    @pytest.mark.asyncio
    async def test_predict_invalid_zone_returns_404(self, client, valid_datetime):
        payload = {"zone": "INVALID_ZONE_999", "datetime": valid_datetime}
        response = await client.post("/predict", json=payload)
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_predict_missing_fields_returns_422(self, client):
        response = await client.post("/predict", json={})
        assert response.status_code == 422


class TestMetricsEndpoint:
    @pytest.mark.asyncio
    async def test_metrics_returns_200(self, client):
        response = await client.get("/metrics")
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_metrics_structure(self, client):
        response = await client.get("/metrics")
        data = response.json()
        assert "prediction_count" in data
        assert "avg_latency_ms" in data
        assert "model_version" in data
