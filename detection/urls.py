from django.urls import path
from . import views

urlpatterns = [
    path('detect/', views.detect_skin_problem, name='detect_skin_problem'),
]
