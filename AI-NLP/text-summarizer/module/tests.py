import json

from django.conf import settings
from django.test import TestCase


class MinutesOfMeetingAPIViewTest(TestCase):
    """Test Class for annotation adaptor. Each method tests rule supported by annotation adaptor"""

    req_id = ""
    test_input = {
        "status": "success",
        "request_id": "req_001",
        "concatenated_view": [
            {
                "start_time": "05:30:02",
                "end_time": "05:30:20",
                "speaker_id": 0,
                "chunk_id": 0,
                "path": "C:/Users/AG99023/Downloads//98e52eab-9aae-4438-94fb-95f4766c00a0/speaker_diarization/output/audio_chunk_2508_20331_0.wav",
                "transcript": "[SOT][]Oh,[/]okay. So that[]if[/] people[]don't know.[/][EOT]",
                "summary": "at where it has not joined so when he is back upon lan yor contequativ yes under the name of the that is.",
                "label": ["Update", 60.51340103149414],
            }
        ],
    }

    expected_transcript = "okay. So that people"

    def setUp(self):
        response = self.client.post("/api/teams/", {"name": "hive"})
        self.assertEqual(response.status_code, 201, response.json())
        with open(settings.BASE_DIR / "input.json", "rb") as fp:
            response = self.client.post("/api/uploads/", {"file": fp, "team_name": 1})
            self.assertEqual(response.status_code, 201, response.json())
        self.req_id = response.json()["masked_request_id"]

    def test_r1(self):
        """Rule :  Extra words by machine"""
        response = self.client.patch(
            f"/module/mom/{self.req_id}/",
            data=json.dumps(self.test_input),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 200, response.json())
        cleaned_transcript = response.json()[0]["transcript"]
        self.assertEquals(
            cleaned_transcript,
            self.expected_transcript,
            "cleaned and expcted transcript does not match",
        )

    def test_r2(self):
        """Rule: Words missed by machine"""
        self.test_input = {
            "status": "success",
            "request_id": "req_001",
            "concatenated_view": [
                {
                    "start_time": "05:30:02",
                    "end_time": "05:30:20",
                    "speaker_id": 0,
                    "chunk_id": 0,
                    "path": "C:/Users/AG99023/Downloads//98e52eab-9aae-4438-94fb-95f4766c00a0/speaker_diarization/output/audio_chunk_2508_20331_0.wav",
                    "transcript": "[SOT][]yeah[/] [-]we are left with[/-] a week to complete <>a[#]our[/#]</> sprint []yeah[/][EOT]",
                    "summary": "at where it has not joined so when he is back upon lan yor contequativ yes under the name of the that is.",
                    "label": ["Update", 60.51340103149414],
                }
            ],
        }
        self.expected_transcript = " we are left with a week to complete our sprint "
        response = self.client.patch(
            f"/module/mom/{self.req_id}/",
            data=json.dumps(self.test_input),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 200, response.json())
        cleaned_transcript = response.json()[0]["transcript"]
        self.assertEquals(
            cleaned_transcript,
            self.expected_transcript,
            "cleaned and expcted transcript does not match",
        )

    def test_r3(self):
        """Rule: Speaker separation"""
        self.test_input = {
            "status": "success",
            "request_id": "req_001",
            "concatenated_view": [
                {
                    "start_time": "05:30:02",
                    "end_time": "05:30:20",
                    "speaker_id": 0,
                    "chunk_id": 0,
                    "path": "C:/Users/AG99023/Downloads//98e52eab-9aae-4438-94fb-95f4766c00a0/speaker_diarization/output/audio_chunk_2508_20331_0.wav",
                    "transcript": "[SOT][sp1]This is done[sp1] [sp2] yes [sp2][EOT]",
                    "summary": "at where it has not joined so when he is back upon lan yor contequativ yes under the name of the that is.",
                    "label": ["Update", 60.51340103149414],
                }
            ],
        }
        self.expected_transcript = {"sp1": ["This is done"], "sp2": [" yes "]}
        response = self.client.patch(
            f"/module/mom/{self.req_id}/",
            data=json.dumps(self.test_input),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 200, response.json())
        cleaned_transcript = response.json()[0]["transcript"]
        self.assertEquals(
            cleaned_transcript,
            self.expected_transcript,
            "cleaned and expcted transcript does not match",
        )

    def test_r4(self):
        """Rule: machine totally skipped to record a sentence associated with the audio chunk."""
        self.test_input = {
            "status": "success",
            "request_id": "req_001",
            "concatenated_view": [
                {
                    "start_time": "05:30:02",
                    "end_time": "05:30:20",
                    "speaker_id": 0,
                    "chunk_id": 0,
                    "path": "C:/Users/AG99023/Downloads//98e52eab-9aae-4438-94fb-95f4766c00a0/speaker_diarization/output/audio_chunk_2508_20331_0.wav",
                    "transcript": "[SOT][$] sentiment analysis used right so it is again on anujs name so we have two stories which we need to get an update from anuj I'll touch base.[/$][EOT]",
                    "summary": "at where it has not joined so when he is back upon lan yor contequativ yes under the name of the that is.",
                    "label": ["Update", 60.51340103149414],
                }
            ],
        }
        self.expected_transcript = " sentiment analysis used right so it is again on anujs name so we have two stories which we need to get an update from anuj I'll touch base."
        response = self.client.patch(
            f"/module/mom/{self.req_id}/",
            data=json.dumps(self.test_input),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 200, response.json())
        cleaned_transcript = response.json()[0]["transcript"]
        self.assertEquals(
            cleaned_transcript,
            self.expected_transcript,
            "cleaned and expcted transcript does not match",
        )

    def test_r5(self):
        """Rule: misinterpreted word."""
        self.test_input = {
            "status": "success",
            "request_id": "req_001",
            "concatenated_view": [
                {
                    "start_time": "05:30:02",
                    "end_time": "05:30:20",
                    "speaker_id": 0,
                    "chunk_id": 0,
                    "path": "C:/Users/AG99023/Downloads//98e52eab-9aae-4438-94fb-95f4766c00a0/speaker_diarization/output/audio_chunk_2508_20331_0.wav",
                    "transcript": "[SOT][]yeah[/] [-]we are left with[/-] a week to complete <>a[#]our[/#]</> sprint []yeah[/][EOT]",
                    "summary": "at where it has not joined so when he is back upon lan yor contequativ yes under the name of the that is.",
                    "label": ["Update", 60.51340103149414],
                }
            ],
        }
        self.expected_transcript = " we are left with a week to complete our sprint "
        response = self.client.patch(
            f"/module/mom/{self.req_id}/",
            data=json.dumps(self.test_input),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 200, response.json())
        cleaned_transcript = response.json()[0]["transcript"]
        self.assertEquals(
            cleaned_transcript,
            self.expected_transcript,
            "cleaned and expcted transcript does not match",
        )

    def test_r6(self):
        """Rule: misinterpreted sentence."""
        self.test_input = {
            "status": "success",
            "request_id": "req_001",
            "concatenated_view": [
                {
                    "start_time": "05:30:02",
                    "end_time": "05:30:20",
                    "speaker_id": 0,
                    "chunk_id": 0,
                    "path": "C:/Users/AG99023/Downloads//98e52eab-9aae-4438-94fb-95f4766c00a0/speaker_diarization/output/audio_chunk_2508_20331_0.wav",
                    "transcript": "[SOT][]Yeah.[/] [sp1]Okay. []So[/] [-]sarika[/-] []uh,[/] do you remember, uh, which one you ()would worked have done?[#]worked on?[/#](/) Uh, do you have the time stamp []per day?[/] [-]for that?[/-][sp1][sp2]Yeah.[sp2][EOT]",
                    "summary": "at where it has not joined so when he is back upon lan yor contequativ yes under the name of the that is.",
                    "label": ["Update", 60.51340103149414],
                }
            ],
        }
        self.expected_transcript = {
            "sp1": [
                "Okay.  sarika  do you remember, uh, which one you worked on? Uh, do you have the time stamp  for that?"
            ],
            "sp2": ["Yeah."],
        }
        response = self.client.patch(
            f"/module/mom/{self.req_id}/",
            data=json.dumps(self.test_input),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 200, response.json())
        cleaned_transcript = response.json()[0]["transcript"]
        self.assertEquals(
            cleaned_transcript,
            self.expected_transcript,
            "cleaned and expcted transcript does not match",
        )

    def test_r7(self):
        """Rule: alphabets to numerical mismatch."""
        self.test_input = {
            "status": "success",
            "request_id": "req_001",
            "concatenated_view": [
                {
                    "start_time": "05:30:02",
                    "end_time": "05:30:20",
                    "speaker_id": 0,
                    "chunk_id": 0,
                    "path": "C:/Users/AG99023/Downloads//98e52eab-9aae-4438-94fb-95f4766c00a0/speaker_diarization/output/audio_chunk_2508_20331_0.wav",
                    "transcript": "[SOT][sp1]It's 40% better than the previous output approximately [sp1][sp2]right[sp2][sp1][@]forty [#]40[/#][/@][sp1][EOT]",
                    "summary": "at where it has not joined so when he is back upon lan yor contequativ yes under the name of the that is.",
                    "label": ["Update", 60.51340103149414],
                }
            ],
        }
        self.expected_transcript = {
            "sp1": [
                "It's 40% better than the previous output approximately ",
                "40",
            ],
            "sp2": ["right"],
        }
        response = self.client.patch(
            f"/module/mom/{self.req_id}/",
            data=json.dumps(self.test_input),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 200, response.json())
        cleaned_transcript = response.json()[0]["transcript"]
        self.assertEquals(
            cleaned_transcript,
            self.expected_transcript,
            "cleaned and expcted transcript does not match",
        )

    def test_r8(self):
        """Rule: word recorded wrong with respect to it's spelling."""
        self.test_input = {
            "status": "success",
            "request_id": "req_001",
            "concatenated_view": [
                {
                    "start_time": "05:30:02",
                    "end_time": "05:30:20",
                    "speaker_id": 0,
                    "chunk_id": 0,
                    "path": "C:/Users/AG99023/Downloads//98e52eab-9aae-4438-94fb-95f4766c00a0/speaker_diarization/output/audio_chunk_2508_20331_0.wav",
                    "transcript": "[SOT][sc]bettar[#]better[/#][/sc][EOT]",
                    "summary": "at where it has not joined so when he is back upon lan yor contequativ yes under the name of the that is.",
                    "label": ["Update", 60.51340103149414],
                }
            ],
        }
        self.expected_transcript = "better"
        response = self.client.patch(
            f"/module/mom/{self.req_id}/",
            data=json.dumps(self.test_input),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 200, response.json())
        cleaned_transcript = response.json()[0]["transcript"]
        self.assertEquals(
            cleaned_transcript,
            self.expected_transcript,
            "cleaned and expcted transcript does not match",
        )


class TextSummarizerViewTest(TestCase):
    def test_invalid_request_id(self):
        response = self.client.get("/module/summarizer/invalid/")
        self.assertEqual(response.status_code, 404)

    def test_text_summarizer(self):
        response = self.client.post(
            "/api/teams/",
            {"name": "hive"},
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 201, response.json())
        with open(settings.BASE_DIR / "summarizer_test.json", "rb") as fp:
            response = self.client.post("/api/uploads/", {"file": fp, "team_name": 1})
            self.assertEqual(response.status_code, 201, response.json())

        req_id = response.json()["masked_request_id"]
        response = self.client.get(f"/module/summarizer/{req_id}/")
        self.assertEqual(response.status_code, 200)
