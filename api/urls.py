from django.urls import path
from . import views
from rest_framework_simplejwt.views import TokenObtainPairView, TokenRefreshView
from .views import CustomTokenObtainPairView, explain_text

urlpatterns = [
    path('upload_file/', views.upload_file, name='upload_file'),
    path('perform_analysis/', views.perform_analysis, name='perform_analysis'),
    path('perform_conflict_check/', views.perform_conflict_check, name='perform_conflict_check'),
    path('token/', CustomTokenObtainPairView.as_view(), name='token_obtain_pair'),
    path('token/refresh/', TokenRefreshView.as_view(), name='token_refresh'),
    path('register/', views.register, name='register'),
    path('profile/', views.get_user_profile, name='user_profile'),
    path('accept_terms/', views.accept_terms, name='accept_terms'),
    path('explain_text/', explain_text, name='explain_text'),
    path('reply_to_comment/', views.reply_to_comment, name='reply_to_comment'),
    path('redraft_comment/', views.redraft_comment, name='redraft_comment'),
    path('analyze_clauses/', views.analyze_clauses, name='analyze_clauses'),
    path('analyze_parties/', views.analyze_parties, name='analyze_parties'),
    path('redraft_text/', views.redraft_text, name='redraft_text'),
    path('brainstorm_chat/', views.brainstorm_chat, name='brainstorm_chat'),
    path('preview_pdf_as_docx/', views.preview_pdf_as_docx, name='preview_pdf_as_docx'),
    path('chat/', views.chat, name='chat'),
]
