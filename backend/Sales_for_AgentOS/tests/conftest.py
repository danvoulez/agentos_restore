import pytest, asyncio, uuid
import mongomock_motor
from httpx import AsyncClient
from app.main import app
from app.db.mongo_client import get_database

@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()

@pytest.fixture(autouse=True)
async def mongo_mock(monkeypatch):
    mock_client = mongomock_motor.AsyncMongoMockClient()
    test_db = mock_client[f"test_{uuid.uuid4().hex}"]

    monkeypatch.setattr("app.db.mongo_client._client", mock_client)
    monkeypatch.setattr("app.db.mongo_client.get_database", lambda: test_db)
    yield