from django.urls import path
from . import views
from rest_framework_simplejwt.views import TokenObtainPairView, TokenRefreshView
from .views import CustomTokenObtainPairView

urlpatterns = [
    path('health/', views.health, name='health'),
    path('upload_file/', views.upload_file, name='upload_file'),
    path('perform_analysis/', views.perform_analysis, name='perform_analysis'),
    path('perform_conflict_check/', views.perform_conflict_check, name='perform_conflict_check'),
    path('token/', CustomTokenObtainPairView.as_view(), name='token_obtain_pair'),
    path('token/refresh/', TokenRefreshView.as_view(), name='token_refresh'),
    path('register/', views.register, name='register'),
    path('profile/', views.get_user_profile, name='user_profile'),
]