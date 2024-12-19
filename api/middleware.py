from django.contrib.auth import get_user_model
from rest_framework_simplejwt.tokens import AccessToken
from django.conf import settings
import jwt

class APILoggerMiddlewareCustom:
    def __init__(self, get_response):
        self.get_response = get_response

    def get_user_from_token(self, token):
        try:
            # Decode the token
            decoded = jwt.decode(token, settings.SECRET_KEY, algorithms=["HS256"])
            user_id = decoded.get('user_id')
            if user_id:
                User = get_user_model()
                return User.objects.get(id=user_id)
        except Exception as e:
            print(f"Token decode error: {e}")
        return None

    def __call__(self, request):

        # Get token from Authorization header
        auth_header = request.META.get('HTTP_AUTHORIZATION', '')
        if auth_header.startswith('Bearer '):
            token = auth_header.split(' ')[1]
            user = self.get_user_from_token(token)
            if user:
                request.user = user
                request.META['HTTP_USER'] = user.username
                request.META['HTTP_X_ORGANIZATION_ID'] = str(user.organization.id) if user.organization else 'N/A'
                request.META['HTTP_X_USER_ID'] = str(user.id)

        response = self.get_response(request)
        return response 