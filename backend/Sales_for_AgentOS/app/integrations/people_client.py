from app.integrations.base_client import BaseIntegrationClient
from app.core.config import get_settings

class PeopleClient(BaseIntegrationClient):
    def __init__(self):
        super().__init__("PeopleService", get_settings().PEOPLE_SERVICE_URL)
        self.token = get_settings().PEOPLE_SERVICE_S2S_TOKEN

    def _get_auth_headers(self):
        return {"Authorization": f"Bearer {self.token}"} if self.token else {}