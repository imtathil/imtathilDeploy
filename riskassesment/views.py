from django.shortcuts import render
from .models import RiskAssessment, UserAssessment

# Create your views here.

def risk(request):
    return render(request ,'riskassesment/risk.html', {'title':'risk'}) 

