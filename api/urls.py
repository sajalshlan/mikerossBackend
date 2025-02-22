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
    path('api_summary/', views.get_api_summary, name='api_summary'),
    path('plugin/explain_text/', views.plugin_explain_text, name='plugin_explain_text'),
    path('plugin/reply_to_comment/', views.plugin_reply_to_comment, name='plugin_reply_to_comment'),
    path('plugin/redraft_comment/', views.plugin_redraft_comment, name='plugin_redraft_comment'),
    path('plugin/analyze_clauses/', views.plugin_analyze_clauses, name='plugin_analyze_clauses'),
    path('plugin/analyze_parties/', views.plugin_analyze_parties, name='plugin_analyze_parties'),
    path('plugin/redraft_text/', views.plugin_redraft_text, name='plugin_redraft_text'),
    path('plugin/brainstorm_chat/', views.plugin_brainstorm_chat, name='plugin_brainstorm_chat'),
]
