from django.test import TestCase

from output_endpoint import output_endpoint


class OutputEnpointViewTest(TestCase):
    def test_output_location_is_updated(self):
        payload = {"url": "C://__LegaAvi//INFRASTRUCTURE//_HPOD//SOURCE//OUTTCNG//"}

        response = self.client.post("/output/update-output-endpoint/", payload)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(output_endpoint.get_applicable_config()["url"], payload["url"])
