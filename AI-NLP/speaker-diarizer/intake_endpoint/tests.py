from django.test import TestCase

from intake_endpoint import intake_endpoint


class InputEnpointViewTest(TestCase):
    def test_input_location_is_updated(self):
        payload = {"url": "C://Users//AH41035//Projects//sense_maker/test_folder"}

        response = self.client.post("/input/update-intake-endpoint/", payload)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(intake_endpoint.get_applicable_config()["url"], payload["url"])
