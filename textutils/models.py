"""
Custom database models for Autism Detection System
"""
from django.db import models
from django.contrib.auth.models import User


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
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True)
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
    gender = models.CharField(max_length=10)
    jaundice = models.CharField(max_length=5)
    asd_family = models.CharField(max_length=5)
    result = models.CharField(max_length=50)
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        db_table = 'screening_results'
    
    def __str__(self):
        return f"Screening {self.id} - {self.result}"
