from rest_framework import serializers
from django.contrib.auth import get_user_model
from django.contrib.auth.password_validation import validate_password
from .models import User, Organization, Document

User = get_user_model()

class OrganizationSerializer(serializers.ModelSerializer):
    class Meta:
        model = Organization
        fields = ('id', 'name')

class UserSerializer(serializers.ModelSerializer):
    organization_details = OrganizationSerializer(source='organization', read_only=True)

    class Meta:
        model = User
        fields = ('id', 'username', 'email', 'organization', 'organization_details', 'is_root', 'accepted_terms')

class RegisterSerializer(serializers.ModelSerializer):
    password = serializers.CharField(write_only=True, required=True)
    organization = serializers.PrimaryKeyRelatedField(queryset=Organization.objects.all(), required=False)

    class Meta:
        model = User
        fields = ('username', 'password', 'email', 'organization', 'is_root')

    def create(self, validated_data):
        is_root = validated_data.pop('is_root', False)
        user = User.objects.create_user(
            username=validated_data['username'],
            email=validated_data['email'],
            password=validated_data['password'],
            organization=validated_data.get('organization'),
            is_root=is_root,
            is_staff=is_root,  # Give admin panel access to root users
            is_superuser=is_root  # Give full permissions to root users
        )
        return user

class AcceptTermsSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = ['accepted_terms']
