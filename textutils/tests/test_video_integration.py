
import os,json
import tempfile
from pathlib import Path
from unittest.mock import patch

from django.contrib.auth.models import User
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase
from django.urls import reverse
from unittest.mock import Mock

class VideoUploadIntegrationTests(TestCase):

    def setUp(self):
        self.user = User.objects.create_user(
            username="video_test_user",
            password="test-password-123",
        )

        self.client.force_login(self.user)
        session = self.client.session
        session["resense_role"] = "user"
        session.save()

    def test_missing_video_returns_message(self):
        response = self.client.post(reverse("output"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(
            response,
            "Choose a video file before uploading."
        )
    @patch("textutils.views.subprocess.run")
    def test_unsupported_file_extension_is_rejected(self, mock_run):
        video = SimpleUploadedFile(
            "test_video.txt",
            b"not-a-video",
            content_type="text/plain",
        )

        response = self.client.post(
            reverse("output"),
            {"fi": video},
        )

        self.assertEqual(response.status_code, 200)
        mock_run.assert_not_called()

    @patch("textutils.views.subprocess.run")
    def test_oversized_video_upload_is_rejected(self, mock_run):
        from django.test import RequestFactory
        from textutils.views import output

        video = SimpleUploadedFile(
            "large_video.mp4",
            b"small-test-content",
            content_type="video/mp4",
        )

        video.size = 100 * 1024 * 1024 + 1

        request = RequestFactory().post("/output/")
        request.FILES["fi"] = video
        request.user = self.user

        response = output.__wrapped__(request)

        self.assertEqual(response.status_code, 200)
        self.assertContains(
            response,
            "Video exceeds the maximum upload size of 100 MB."
        )
        mock_run.assert_not_called()


    @patch("textutils.views.subprocess.run")
    def test_successful_video_upload_calls_processor(self, mock_run):
        mock_run.return_value = Mock(
            returncode=0,
            stdout=json.dumps({
                "status": "success",
                "total_frames": 20,
                "frames_saved": 20,
                "prediction": {
                    "predicted_class": "ASD",
                    "model_score": 0.72,
                    "selected_frame": "frame_004.jpg",
                    "frames_processed": 20,
                },
            }),
            stderr="",
        )

        video = SimpleUploadedFile(
            "test_video.avi",
            b"synthetic-test-video-content",
            content_type="video/x-msvideo",
        )

        response = self.client.post(
            reverse("output"),
            {"fi": video},
        )

        self.assertEqual(response.status_code, 200)

        self.assertContains(
            response,
            "experimental model inference completed successfully."
        )
        self.assertContains(response, "ASD")
        self.assertContains(response, "0.7200")
        self.assertContains(response, "not a medical diagnosis")

        mock_run.assert_called_once()

        command = mock_run.call_args[0][0]
        self.assertTrue(command[1].endswith("process_video.py"))

        upload_path = Path(command[-1])

        self.assertFalse(
            upload_path.exists(),
            "Temporary uploaded video should be deleted after processing."
        )

    @patch("textutils.views.subprocess.run")
    def test_processor_failure_returns_error_message(self, mock_run):
        from unittest.mock import Mock

        mock_run.return_value = Mock(
            returncode=1,
            stdout="",
            stderr="Invalid video",
        )

        video = SimpleUploadedFile(
            "invalid.avi",
            b"invalid-video-content",
            content_type="video/x-msvideo",
        )

        response = self.client.post(
            reverse("output"),
            {"fi": video},
        )

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Video processing failed.")

    @patch("textutils.views.subprocess.run")
    def test_processor_timeout_returns_message(self, mock_run):
        import subprocess

        mock_run.side_effect = subprocess.TimeoutExpired(
            cmd="main.py",
            timeout=120,
        )

        video = SimpleUploadedFile(
            "timeout.avi",
            b"synthetic-test-video-content",
            content_type="video/x-msvideo",
        )

        response = self.client.post(
            reverse("output"),
            {"fi": video},
        )

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Video processing timed out.")

    def test_unauthenticated_user_cannot_upload(self):
        self.client.logout()

        response = self.client.post(reverse("output"))

        self.assertEqual(response.status_code, 302)
