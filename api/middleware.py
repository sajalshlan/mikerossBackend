from django.contrib.auth import get_user_model
from rest_framework_simplejwt.tokens import AccessToken
from django.conf import settings
import jwt
from django.db import connection
import logging
from django.utils import timezone

logger = logging.getLogger(__name__)

class DatabaseConnectionMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        # Before the view is called
        try:
            # First ensure connection exists
            if connection.connection is None:
                logger.info("No database connection exists, creating new connection...")
                connection.connect()
            # Then check if it's usable
            elif not connection.is_usable():
                logger.warning("Database connection was stale, reconnecting... Connection age: %s", 
                             getattr(connection.connection, '_last_use_time', 'unknown'))
                connection.close()
                connection.connect()
            
            # Add a timestamp to track connection age
            connection.connection._last_use_time = timezone.now()
            
        except Exception as e:
            logger.error(f"Error checking database connection: {str(e)}", exc_info=True)

        # Process the request
        response = self.get_response(request)

        # After the view is called, before logging
        try:
            if connection.connection and not connection.is_usable():
                logger.warning("Database connection lost during request, reconnecting...")
                connection.close()
                connection.connect()
        except Exception as e:
            logger.error(f"Error checking database connection after request: {str(e)}", exc_info=True)
        return response

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