from django.contrib import admin
from django.db.models import Count
from django.utils import timezone
from datetime import datetime, timedelta
from .models import Organization, User
from drf_api_logger.models import APILogsModel
from drf_api_logger.admin import APILogsAdmin
import json
from django.contrib.auth.admin import UserAdmin as BaseUserAdmin
from django.contrib.auth.forms import UserCreationForm, UserChangeForm
from django import forms

# First unregister the existing admin
admin.site.unregister(APILogsModel)

# Create our custom admin by extending the original APILogsAdmin
class CustomAPILogsAdmin(APILogsAdmin):
    list_display = ('api', 'method', 'status_code', 'execution_time', 'added_on', 'user', 'organization')
    list_filter = ('method', 'status_code', 'added_on')
    search_fields = ('api', 'headers')

    def user(self, obj):
        try:
            # Try to get user from META headers first
            if isinstance(obj.headers, str):
                headers = json.loads(obj.headers)
            else:
                headers = obj.headers
            return headers.get('USER', headers.get('user', 'Anonymous'))
        except (json.JSONDecodeError, AttributeError):
            return 'Anonymous'
    user.short_description = 'User'

    def organization(self, obj):
        try:
            if isinstance(obj.headers, str):
                headers = json.loads(obj.headers)
            else:
                headers = obj.headers
            org_id = headers.get('X_ORGANIZATION_ID', headers.get('x-organization-id', 'N/A'))
            
            # If we have a valid org_id, get the organization name
            if org_id and org_id != 'N/A':
                try:
                    org = Organization.objects.get(id=org_id)
                    return org.name
                except Organization.DoesNotExist:
                    return f'Org {org_id} (deleted)'
            return 'N/A'
        except (json.JSONDecodeError, AttributeError):
            return 'N/A'
    organization.short_description = 'Organization'

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

class CustomUserCreationForm(UserCreationForm):
    class Meta(UserCreationForm.Meta):
        model = User
        fields = ('username', 'email', 'organization', 'is_root')

class CustomUserChangeForm(UserChangeForm):
    class Meta(UserChangeForm.Meta):
        model = User
        fields = ('username', 'email', 'organization', 'is_root')

@admin.register(User)
class UserAdmin(BaseUserAdmin):
    form = CustomUserChangeForm
    add_form = CustomUserCreationForm
    
    list_display = ('username', 'email', 'organization', 'is_root', 'get_api_calls_today')
    list_filter = ('organization', 'is_root', 'is_staff', 'is_superuser')
    search_fields = ('username', 'email')
    ordering = ('username',)

    fieldsets = (
        (None, {'fields': ('username', 'password')}),
        ('Personal info', {'fields': ('email', 'organization')}),
        ('Permissions', {'fields': ('is_root', 'is_active', 'is_staff', 'is_superuser')}),
    )
    add_fieldsets = (
        (None, {
            'classes': ('wide',),
            'fields': ('username', 'email', 'organization', 'is_root', 'password1', 'password2'),
        }),
    )

    def get_api_calls_today(self, obj):
        today = timezone.now().date()
        return APILogsModel.objects.filter(
            headers__contains=f'"user":"{obj.username}"',
            added_on__date=today
        ).count()
    get_api_calls_today.short_description = 'API Calls Today'
