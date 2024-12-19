from django.contrib import admin
from django.db.models import Count
from django.utils import timezone
from datetime import datetime, timedelta
from .models import Organization, User
from drf_api_logger.models import APILogsModel
from drf_api_logger.admin import APILogsAdmin
import json

# First unregister the existing admin
admin.site.unregister(APILogsModel)

# Create our custom admin by extending the original APILogsAdmin
class CustomAPILogsAdmin(APILogsAdmin):
    list_display = APILogsAdmin.list_display + ('user', 'organization')

    def user(self, obj):
        try:
            headers = json.loads(obj.headers) if obj.headers else {}
            return headers.get('user', 'Anonymous')
        except json.JSONDecodeError:
            return 'Anonymous'

    def organization(self, obj):
        try:
            headers = json.loads(obj.headers) if obj.headers else {}
            return headers.get('x-organization-id', 'N/A')
        except json.JSONDecodeError:
            return 'N/A'

# Register our custom admin
admin.site.register(APILogsModel, CustomAPILogsAdmin)

@admin.register(Organization)
class OrganizationAdmin(admin.ModelAdmin):
    list_display = ('name', 'created_at', 'get_api_calls_today', 'get_api_calls_month')
    
    def get_api_calls_today(self, obj):
        today = timezone.now().date()
        return APILogsModel.objects.filter(
            headers__contains=f'"x-organization-id":"{obj.id}"',
            added_on__date=today
        ).count()
    get_api_calls_today.short_description = 'API Calls Today'

    def get_api_calls_month(self, obj):
        month_start = timezone.now().date().replace(day=1)
        return APILogsModel.objects.filter(
            headers__contains=f'"x-organization-id":"{obj.id}"',
            added_on__date__gte=month_start
        ).count()
    get_api_calls_month.short_description = 'API Calls This Month'

@admin.register(User)
class UserAdmin(admin.ModelAdmin):
    list_display = ('username', 'email', 'organization', 'is_root', 'get_api_calls_today')
    list_filter = ('organization', 'is_root')
    search_fields = ('username', 'email')

    def get_api_calls_today(self, obj):
        today = timezone.now().date()
        return APILogsModel.objects.filter(
            headers__contains=f'"user":"{obj.username}"',
            added_on__date=today
        ).count()
    get_api_calls_today.short_description = 'API Calls Today'
