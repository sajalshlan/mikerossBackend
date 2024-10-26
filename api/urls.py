from django.urls import path
from . import views

urlpatterns = [
    path('health/', views.health, name='health'),
    path('upload_file/', views.upload_file, name='upload_file'),
    path('perform_analysis/', views.perform_analysis, name='perform_analysis'),
    path('perform_conflict_check/', views.perform_conflict_check, name='perform_conflict_check'),
]