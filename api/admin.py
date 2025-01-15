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
import pytz
from django.conf import settings

# First unregister the existing admin
admin.site.unregister(APILogsModel)

# Create our custom admin by extending the original APILogsAdmin
class CustomAPILogsAdmin(APILogsAdmin):
    list_display = ('api', 'method', 'status_code', 'execution_time', 'get_date', 'get_time', 'user', 'organization')
    list_filter = ('method', 'status_code', 'added_on')
    search_fields = ('api', 'headers')

    def get_date(self, obj):
        try:
            ist_time = obj.added_on + timedelta(hours=5, minutes=30)
            return ist_time.strftime("%d/%m/%y")
        except Exception as e:
            return str(obj.added_on)
    get_date.short_description = 'Date'
    get_date.admin_order_field = 'added_on'

    def get_time(self, obj):
        try:
            ist_time = obj.added_on + timedelta(hours=5, minutes=30)
            return f"{ist_time.strftime('%H:%M:%S')}"
        except Exception as e:
            return str(obj.added_on)
    get_time.short_description = 'Time (IST)'
    get_time.admin_order_field = 'added_on'

    def user(self, obj):
        try:
            if isinstance(obj.headers, str):
                headers = json.loads(obj.headers)
            else:
                headers = obj.headers

            # Get user info from headers
            username = headers.get('USER', headers.get('user'))
            
            # Return username if it exists and isn't None or empty
            if username and username.strip():
                return username
                
            # Check for root user or other special cases
            user_id = headers.get('X_USER_ID', headers.get('x-user-id'))
            if user_id:
                try:
                    user = User.objects.get(id=user_id)
                    return user.username
                except User.DoesNotExist:
                    pass
                    
            return 'Anonymous'
        except (json.JSONDecodeError, AttributeError):
            return 'Anonymous'
    user.short_description = 'User'

    def organization(self, obj):
        try:
            if isinstance(obj.headers, str):
                headers = json.loads(obj.headers)
            else:
                headers = obj.headers
            
            # Get organization ID from headers
            org_id = headers.get('X_ORGANIZATION_ID', headers.get('x-organization-id'))
            
            # Check if org_id is None, 'N/A', empty string, or invalid
            if not org_id or org_id == 'N/A' or org_id == '':
                return 'No Organization'
            
            try:
                # Convert to int to ensure it's a valid ID
                org_id = int(org_id)
                org = Organization.objects.get(id=org_id)
                return org.name
            except (ValueError, Organization.DoesNotExist):
                return 'Invalid Organization'
                
        except (json.JSONDecodeError, AttributeError):
            return 'No Organization'
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
