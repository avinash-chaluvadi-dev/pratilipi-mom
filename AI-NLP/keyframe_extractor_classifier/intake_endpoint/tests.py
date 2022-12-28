from django.test import TestCase

from intake_endpoint import intake_endpoint


class InputEnpointViewTest(TestCase):
    def test_input_location_is_updated(self):
        payload = {
            "url": "C://__LegaAvi//INFRASTRUCTURE//_HPOD//SOURCE//NEW_INPUTCNG//"
        }

        response = self.client.post("/input/update-input-endpoint/", payload)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(intake_endpoint.get_applicable_config()["url"], payload["url"])
