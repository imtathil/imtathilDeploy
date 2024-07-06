from django.urls import path
from . import views

urlpatterns = [
    path('risk/', views.risk , name='risk'),
]