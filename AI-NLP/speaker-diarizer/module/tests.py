from django.conf import settings
from django.test import TestCase


class InputEnpointViewTest(TestCase):
    def test_invalid_request_id(self):
        response = self.client.get("/module/speakerdiarization/invalid/")
        self.assertEqual(response.status_code, 404)

    def test_sentiment_classifier(self):
        response = {}

        with open(settings.BASE_DIR / "input.json", "rb") as fp:
            response = self.client.post("/api/upload/", {"file": fp})
            self.assertEqual(response.status_code, 201)

        req_id = response.json()["masked_request_id"]
        response = self.client.get(f"/module/speakerdiarization/{req_id}/")
        self.assertEqual(response.status_code, 200)
