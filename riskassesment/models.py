from django.db import models
from django.contrib.auth.models import User

class RiskAssessment(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True)
    main_control = models.CharField(max_length=5000, default='')
    subcontrol = models.CharField(max_length=5000, default='')
    control_details = models.CharField(max_length=5000, default='')
    status = models.BooleanField(default=True)

class UserAssessment(models.Model):
    risk_assessment = models.ForeignKey(RiskAssessment, on_delete=models.CASCADE)
    likelihood = models.IntegerField(default=0)
    impact = models.IntegerField(default=0)
    exposure = models.IntegerField(default=0)
    risk_level = models.CharField(max_length=5000, default='')
    user = models.ForeignKey(User, on_delete=models.CASCADE)
