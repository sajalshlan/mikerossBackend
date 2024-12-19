class APILoggerMiddlewareCustom:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        # Add user and organization info to request headers
        if request.user.is_authenticated:
            request.META['HTTP_USER'] = request.user.username
            if request.user.organization:
                request.META['HTTP_X_ORGANIZATION_ID'] = str(request.user.organization.id)

        response = self.get_response(request)
        return response 