import json

from django.conf import settings
from django.test import TestCase


class FileViewSetTest(TestCase):
    def test_file_upload(self):
        response = self.client.post("/api/teams/", {"name": "hive"})
        self.assertEqual(response.status_code, 201, response.json())
        with open(settings.BASE_DIR / "input.json", "rb") as fp:
            response = self.client.post("/api/uploads/", {"file": fp, "team_name": 1})
            self.assertEqual(response.status_code, 201, response.json())

    def test_file_list_view(self):
        response = self.client.get("/api/uploads/")
        self.assertEqual(response.status_code, 200, response.json())

    def test_cancel_extraction_view(self):
        response = {}
        response = self.client.post("/api/teams/", {"name": "hive"})
        self.assertEqual(response.status_code, 201, response.json())
        with open(settings.BASE_DIR / "input.json", "rb") as fp:
            response = self.client.post("/api/uploads/", {"file": fp, "team_name": 1})
            self.assertEqual(response.status_code, 201, response.json())

        response = self.client.patch(
            f"/api/uploads/{response.json()['id']}/cancel_extraction/"
        )
        self.assertEqual(response.status_code, 200, response.json())


class TeamListCreateAPIViewTest(TestCase):
    def test_create_team_name(self):
        response = self.client.post("/api/teams/", {"name": "hive"})
        self.assertEqual(response.status_code, 201, response.json())

    def test_get_team_name_list(self):
        response = self.client.get("/api/teams/")
        self.assertEqual(response.status_code, 200, response.json())


class MeetingMetadataRetrieveUpdateAPIViewTest(TestCase):
    def setUp(self):
        response = self.client.post("/api/teams/", {"name": "hive"})
        self.assertEqual(response.status_code, 201, response.json())
        with open(settings.BASE_DIR / "input.json", "rb") as fp:
            response = self.client.post("/api/uploads/", {"file": fp, "team_name": 1})
            self.assertEqual(response.status_code, 201, response.json())
        self.req_id = response.json()["masked_request_id"]

    def test_get_meeting_metadata(self):
        response = self.client.get(f"/api/meeting-metadata/{self.req_id}/")
        self.assertEqual(response.status_code, 200, response.json())

    def test_patch_meeting_metadata(self):
        response = self.client.patch(
            f"/api/meeting-metadata/{self.req_id}/",
            data={"organiser": "Faisal"},
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 200, response.json())
        self.assertEqual(response.json()["organiser"], "Faisal")
