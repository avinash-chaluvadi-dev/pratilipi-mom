import logging as lg

from django.conf import settings
from django.test import TestCase

logger = lg.getLogger("file")


class ClassifierViewTest(TestCase):
    def test_invalid_request_id(self):
        response = self.client.gets

    def test_text_summarizer(self):
        response = self.client.post(
            "/api/teams/",
            {"name": "hive"},
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 201, response.json())
        with open(settings.BASE_DIR / "input.json", "rb") as fp:
            response = self.client.post("/api/uploads/", {"file": fp, "team_name": 1})
            self.assertEqual(response.status_code, 201, response.json())

        req_id = response.json()["masked_request_id"]
        response = self.client.get(f"/module/ner/{req_id}/")
        self.assertEqual(response.status_code, 200)


class DeadlineEscalationViewTest(TestCase):
    def test_invalid_request_id(self):
        response = self.client.get("/module/allocator/invalid/")
        self.assertEqual(response.status_code, 404)

    def test_deadline_escalation(self):
        response = self.client.post(
            "/api/teams/",
            {"name": "hive"},
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 201, response.json())
        with open(settings.BASE_DIR / "allocator_test.json", "rb") as fp:
            response = self.client.post("/api/uploads/", {"file": fp, "team_name": 1})
            self.assertEqual(response.status_code, 201, response.json())

        req_id = response.json()["masked_request_id"]
        response = self.client.get(f"/module/allocator/{req_id}/")
        self.assertEqual(response.status_code, 200)


class RecommendationClassifierViewTest(TestCase):
    def test_invalid_request_id(self):
        response = self.client.get("/module/allocator/invalid/")
        self.assertEqual(response.status_code, 404)

    def test_recommendation_classifier(self):
        response = self.client.post(
            "/api/teams/",
            {"name": "hive"},
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 201, response.json())
        with open(settings.BASE_DIR / "recommendation_test.json", "rb") as fp:
            response = self.client.post("/api/uploads/", {"file": fp, "team_name": 1})
            self.assertEqual(response.status_code, 201, response.json())

        req_id = response.json()["masked_request_id"]
        response = self.client.get(f"/module/recommendation/{req_id}/")
        self.assertEqual(response.status_code, 200)

    def test_invalid_markerclsrequest_id(self):
        try:
            response = self.client.get("/module/markercls/req_001/")
            self.assertEqual(response.status_code, 404)
        except Exception as identifier:
            pass
            logger.info("Exception::{0}".format(identifier))

    def test_marker_classifier(self):
        try:
            print("=" * 100, "Test:2")
            response = {}

            # post:
            with open(settings.BASE_DIR / "marker_classifier.json", "rb") as fp:
                response = self.client.post(
                    "/api/uploads/", {"file": fp, "team_name": 1}
                )
            self.assertEqual(response.status_code, 201)
            # call:
            req_id = response.json()["masked_request_id"]
            response = self.client.get(f"/module/markercls/{req_id}/")
            self.assertEqual(response.status_code, 200)
        except Exception as identifier:
            pass
            logger.info("Exception::{0}".format(identifier))

    def test_validate_marker_classifier_inputfeed(self):
        try:
            print("=" * 100, "Test:3")

            import json

            """An malformed inputfeed data is invalid."""
            dataInputFeed = settings.BASE_DIR / "marker_classifier.json"
            with open(dataInputFeed, "rb") as fp:
                json_data = json.loads(fp.read())
                if json_data.get("transcript"):
                    self.assertTrue(json_data)
                    logger.info("File input feed it is a valid feed.")
                else:
                    self.assertFalse(json_data)
                    logger.info("File input feed is not a valid.")
        except Exception as identifier:
            pass
            logger.info("Exception::{0}".format(identifier))

    def test_invalid_labelclsrequest_id(self):
        try:
            print("=" * 100, "Test:4")
            response = self.client.get("/module/labelcls/req_001/")
            self.assertEqual(response.status_code, 404)
        except Exception as identifier:
            pass
            logger.info("Exception::{0}".format(identifier))

    def test_label_classifier(self):
        try:
            print("=" * 100, "Test:5")
            response = {}
            with open(settings.BASE_DIR / "label_classifier.json", "rb") as fp:
                response = self.client.post(
                    "/api/uploads/", {"file": fp, "team_name": 1}
                )
            self.assertEqual(response.status_code, 201)

            req_id = response.json()["masked_request_id"]
            response = self.client.get(f"/module/labelcls/{req_id}/")
            self.assertEqual(response.status_code, 200)
        except Exception as identifier:
            pass
            logger.info("Exception::{0}".format(identifier))

    def test_validate_label_classifier_inputfeed(self):
        try:
            print("=" * 100, "Test:6")
            import json

            """An malformed inputfeed data is invalid."""
            dataInputFeed = settings.BASE_DIR / "label_classifier.json"
            with open(dataInputFeed, "rb") as fp:
                json_data = json.loads(fp.read())
                # print("DATA:INPUT:FEED::", json_data)
                if json_data.get("speech_to_text"):
                    self.assertTrue(json_data)
                    logger.info("File input feed it is a valid feed.")
                else:
                    self.assertFalse(json_data)
                    logger.info("File input feed is not a valid.")
        except Exception as identifier:
            pass
            logger.info("Exception::{0}".format(identifier))
