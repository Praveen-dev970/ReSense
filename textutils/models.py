"""
Custom database models for Autism Detection System
"""
from django.db import models
from django.contrib.auth.models import User
import uuid

class VideoFile(models.Model):
    """Model for storing uploaded video files"""
    vpath = models.CharField(max_length=255)
    vdescr = models.TextField(blank=True, null=True)
    uploaded_at = models.DateTimeField(auto_now_add=True)
    uploaded_by = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True)

    class Meta:
        db_table = 'vfiles'

    def __str__(self):
        return f"{self.vpath} - {self.vdescr[:50]}"


class AutismScreeningResult(models.Model):
    """Model for storing screening questionnaire results"""
    user = models.ForeignKey(User,
        on_delete=models.CASCADE,
        null=True, blank=True)
    submission_token = models.UUIDField(
        default=uuid.uuid4,
        editable=False,
        unique=True
    )
    q1 = models.IntegerField()
    q2 = models.IntegerField()
    q3 = models.IntegerField()
    q4 = models.IntegerField()
    q5 = models.IntegerField()
    q6 = models.IntegerField()
    q7 = models.IntegerField()
    q8 = models.IntegerField()
    q9 = models.IntegerField()
    q10 = models.IntegerField()
    age = models.IntegerField()
    age_months = models.IntegerField(default=0)
    gender = models.CharField(max_length=10)
    jaundice = models.CharField(max_length=5)
    asd_family = models.CharField(max_length=5)
    result = models.CharField(max_length=50)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = 'screening_results'

    def __str__(self):
        return f"Screening {self.id} - {self.result}"


class VideoAssessmentResult(models.Model):
    """Store experimental video CNN assessment metadata for each user."""

    CLASS_CHOICES = [
        ("ASD", "ASD"),
        ("TD", "TD"),
    ]

    user = models.ForeignKey(
        User,
        on_delete=models.CASCADE,
        related_name="video_assessment_results"
    )

    predicted_class = models.CharField(
        max_length=3,
        choices=CLASS_CHOICES
    )

    # Raw model output; this is not a clinically validated probability.
    model_score = models.FloatField()

    total_frames = models.PositiveIntegerField()
    frames_saved = models.PositiveIntegerField()
    frames_processed = models.PositiveIntegerField()

    selected_frame = models.CharField(max_length=100)

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "video_assessment_results"
        ordering = ["-created_at"]

    def __str__(self):
        return (
            f"Video assessment {self.id} - "
            f"{self.predicted_class} - {self.user.username}"
        )
