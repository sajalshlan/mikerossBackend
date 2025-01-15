import os

# Default to development settings
environment = os.getenv('DJANGO_ENVIRONMENT', 'development')
print(environment)

if environment == 'production':
    from .production import *
else:
    from .development import *