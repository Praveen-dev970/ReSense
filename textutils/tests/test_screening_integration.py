
import uuid

from django.contrib.auth.models import User
from django.test import TestCase
from django.urls import reverse

from textutils.models import AutismScreeningResult


class ScreeningIntegrationTests(TestCase):

    def setUp(self):
        self.user = User.objects.create_user(
            username="screening_user",
            password="test-password-123",
        )

        self.other_user = User.objects.create_user(
            username="other_user",
            password="test-password-123",
        )

        self.admin = User.objects.create_superuser(
            username="screening_admin",
            email="admin@example.com",
            password="test-password-123",
        )

    def login_as_user(self, user):
        self.client.force_login(user)
        session = self.client.session
        session["resense_role"] = "user"
        session.save()

    def login_as_admin(self):
        self.client.force_login(self.admin)
        session = self.client.session
        session["resense_role"] = "admin"
        session.save()

    def valid_submission(self, token=None):
        data = {
            "submission_token": str(token or uuid.uuid4()),
            "age": "2",
            "age_months": "0",
            "gender": "Male",
            "jaundice": "No",
            "asd_family": "No",
        }

        # Valid responses using the application's documented options.
        for number in range(1, 11):
            data["t{}".format(number)] = "Always"

        return data

    def create_screening(self, user):
        return AutismScreeningResult.objects.create(
            user=user,
            submission_token=uuid.uuid4(),
            q1=0, q2=0, q3=0, q4=0, q5=0,
            q6=0, q7=0, q8=0, q9=0, q10=0,
            age=2,
            age_months=0,
            gender="Male",
            jaundice="No",
            asd_family="No",
            result="At or below threshold",
        )

    def test_valid_submission_is_saved_once(self):
        self.login_as_user(self.user)

        response = self.client.post(
            reverse("analyze"),
            self.valid_submission(),
        )

        self.assertEqual(response.status_code, 302)
        self.assertEqual(
            AutismScreeningResult.objects.filter(
                user=self.user
            ).count(),
            1,
        )

    def test_repeated_submission_token_does_not_create_duplicate(self):
        self.login_as_user(self.user)
        token = uuid.uuid4()
        data = self.valid_submission(token)

        first_response = self.client.post(
            reverse("analyze"), data
        )
        second_response = self.client.post(
            reverse("analyze"), data
        )

        self.assertEqual(first_response.status_code, 302)
        self.assertEqual(second_response.status_code, 302)
        self.assertEqual(
            AutismScreeningResult.objects.filter(
                submission_token=token
            ).count(),
            1,
        )

    def test_invalid_submission_is_not_saved(self):
        self.login_as_user(self.user)
        data = self.valid_submission()
        data["t1"] = "invalid-response"

        response = self.client.post(
            reverse("analyze"), data
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            AutismScreeningResult.objects.filter(
                user=self.user
            ).count(),
            0,
        )

    def test_screening_history_only_shows_own_records(self):
        own_record = self.create_screening(self.user)
        self.create_screening(self.other_user)

        self.login_as_user(self.user)

        response = self.client.get(
            reverse("screening_history")
        )

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, str(own_record.id))
        self.assertEqual(
            list(response.context["records"]),
            [own_record],
        )

    def test_user_cannot_access_another_users_result(self):
        other_record = self.create_screening(self.other_user)
        self.login_as_user(self.user)

        response = self.client.get(
            reverse(
                "screening_result",
                kwargs={"result_id": other_record.id},
            )
        )

        self.assertEqual(response.status_code, 404)

    def test_admin_can_access_all_screening_records(self):
        self.create_screening(self.user)
        self.create_screening(self.other_user)

        self.login_as_admin()

        response = self.client.get(
            reverse("admin_screenings")
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.context["records"].count(),
            2,
        )

    def test_regular_user_cannot_access_admin_screenings(self):
        self.login_as_user(self.user)

        response = self.client.get(
            reverse("admin_screenings")
        )

        self.assertEqual(response.status_code, 302)
