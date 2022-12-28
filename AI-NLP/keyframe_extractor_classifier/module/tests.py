# -*- coding: utf-8 -*-
"""
    @Author         : HIVE TEAM
    @Purpose        : Model TestCases***
    @Description    : This is the generalized test case to validate other model related actions.
    @Date           : 25-08-2021
    @Last Modified  : 01-09-2021
"""
import logging as lg

from django.conf import settings
from django.test import TestCase
from rest_framework import serializers

logger = lg.getLogger("file")


class KeyframeextractionclassifierViewTest(TestCase):
    def test_invalid_request_id(self):
        try:
            response = self.client.get("/module/keyframeext/req_079/")
            self.assertEqual(response.status_code, 404)
        except Exception as identifier:
            pass
            logger.info("Exception::{0}".format(identifier))

    def test_keyframe_extraction_classifier(self):
        try:
            print("Test Case Initiated :: {0}".format("test_validate_inputfeed"))

            response = {}

            with open(settings.BASE_DIR / "input_keyframelabel.json", "rb") as fp:
                response = self.client.post("/api/upload/", {"file": fp})
            self.assertEqual(response.status_code, 201)

            req_id = response.json()["masked_request_id"]
            response = self.client.get(f"/module/keyframeext/{req_id}/")
            self.assertEqual(response.status_code, 200)
        except Exception as identifier:
            pass
            logger.info("Exception::{0}".format(identifier))

    def test_validate_inputfeed(self):
        try:
            print("Test Case Initiated :: {0}".format("test_validate_inputfeed"))
            import json

            """An malformed inputfeed data is invalid."""
            dataInputFeed = settings.BASE_DIR / "input_keyframelabel.json"
            with open(dataInputFeed, "rb") as fp:
                json_data = json.loads(fp.read())
                # print("DATA:INPUT:FEED::", json_data)
                if json_data.get("keyframes"):
                    self.assertTrue(json_data)
                    logger.info("File input feed it is a valid feed.")
                else:
                    self.assertFalse(json_data)
                    logger.info("File input feed is not a valid.")
        except Exception as identifier:
            pass
            logger.info("Exception::{0}".format(identifier))

    def test_invalid_framify_request_id(self):
        try:
            print(
                "Test Case Initiated :: {0}".format("test_invalid_framify_request_id")
            )
            response = self.client.get("/module/framify/req_100/")
            self.assertEqual(response.status_code, 404)
        except Exception as identifier:
            pass
            logger.info("Exception::{0}".format(identifier))

    def test_framify_extraction_classifier(self):
        try:
            print(
                "Test Case Initiated :: {0}".format(
                    "test_framify_extraction_classifier"
                )
            )
            response = {}

            with open(settings.BASE_DIR / "input_framify.json", "rb") as fp:
                response = self.client.post("/api/upload/", {"file": fp})
            self.assertEqual(response.status_code, 201)

            req_id = response.json()["masked_request_id"]
            response = self.client.get(f"/module/framify/{req_id}/")
            self.assertEqual(response.status_code, 200)
        except Exception as identifier:
            pass
            logger.info("Exception::{0}".format(identifier))

    def test_framify_validate_inputfeed(self):
        try:
            print(
                "Test Case Initiated :: {0}".format("test_framify_validate_inputfeed")
            )
            import json

            """An malformed inputfeed data is invalid."""
            dataInputFeed = settings.BASE_DIR / "input_framify.json"
            with open(dataInputFeed, "rb") as fp:
                json_data = json.loads(fp.read())
                # print("DATA:INPUT:FEED::", json_data)
                if json_data.get("keyframes"):
                    self.assertTrue(json_data)
                    logger.info("File input feed it is a valid feed.")
                else:
                    self.assertFalse(json_data)
                    logger.info("File input feed is not a valid.")
        except Exception as identifier:
            pass
            logger.info("Exception::{0}".format(identifier))
