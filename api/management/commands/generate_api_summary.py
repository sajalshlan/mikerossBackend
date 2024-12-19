from django.core.management.base import BaseCommand
from django.utils import timezone
from datetime import datetime, timedelta
from drf_api_logger.models import APILogsModel
from api.models import Organization, User
import json
from collections import defaultdict
from decimal import Decimal
import os

class Command(BaseCommand):
    help = 'Generate API usage summary for a specific date'

    def add_arguments(self, parser):
        # Optional date argument, defaults to today
        parser.add_argument(
            '--date',
            type=str,
            help='Date in DD-MM-YYYY format (defaults to today)',
            required=False
        )
        parser.add_argument(
            '--output',
            type=str,
            help='Output file path',
            default='api_summary.json'
        )

    def decimal_default(self, obj):
        if isinstance(obj, Decimal):
            return float(obj)
        raise TypeError

    def handle(self, *args, **options):
        # Get the date to analyze
        if options['date']:
            try:
                target_date = datetime.strptime(options['date'], '%d-%m-%Y').date()
            except ValueError:
                self.stdout.write(self.style.ERROR('Invalid date format. Use DD-MM-YYYY'))
                return
        else:
            target_date = timezone.now().date()

        # Initialize summary dictionary
        summary = {
            'date': target_date.strftime('%d/%m/%y'),
            'total_api_calls': 0,
            'organizations': {},
            'status_code_distribution': defaultdict(int),
            'average_execution_time': 0,
            'peak_hour': None,
            'hourly_distribution': defaultdict(int)
        }

        # Get all API calls for the day
        api_logs = APILogsModel.objects.filter(
            added_on__date=target_date
        )

        # Calculate total API calls
        total_calls = api_logs.count()
        summary['total_api_calls'] = total_calls

        if total_calls == 0:
            self.stdout.write(self.style.WARNING(f'No API calls found for {target_date}'))
            return

        # Process each API call
        total_execution_time = 0
        org_data = defaultdict(lambda: {
            'total_calls': 0,
            'users': defaultdict(lambda: {
                'total_calls': 0,
                'status_codes': defaultdict(int),
                'endpoints': defaultdict(int),
                'total_execution_time': 0
            })
        })

        for log in api_logs:
            # Get headers
            try:
                headers = json.loads(log.headers) if isinstance(log.headers, str) else log.headers
            except json.JSONDecodeError:
                headers = {}

            # Get organization and user info
            org_id = headers.get('X_ORGANIZATION_ID', headers.get('x-organization-id', 'N/A'))
            username = headers.get('USER', headers.get('user', 'Anonymous'))

            # Update organization and user statistics
            org_name = 'No Organization'
            if org_id and org_id != 'N/A':
                try:
                    org = Organization.objects.get(id=org_id)
                    org_name = org.name
                except Organization.DoesNotExist:
                    org_name = f'Unknown Org ({org_id})'

            # Update statistics
            org_data[org_name]['total_calls'] += 1
            org_data[org_name]['users'][username]['total_calls'] += 1
            org_data[org_name]['users'][username]['status_codes'][str(log.status_code)] += 1
            org_data[org_name]['users'][username]['endpoints'][log.api] += 1
            org_data[org_name]['users'][username]['total_execution_time'] += log.execution_time

            # Update global statistics
            summary['status_code_distribution'][str(log.status_code)] += 1
            total_execution_time += log.execution_time

            # Update hourly distribution
            hour = (log.added_on + timedelta(hours=5, minutes=30)).strftime('%H')
            summary['hourly_distribution'][hour] += 1

        # Calculate averages and find peak hour
        summary['average_execution_time'] = float(total_execution_time) / total_calls
        peak_hour = max(summary['hourly_distribution'].items(), key=lambda x: x[1])
        summary['peak_hour'] = f"{peak_hour[0]}:00 IST ({peak_hour[1]} calls)"

        # Format organization data
        summary['organizations'] = {
            org_name: {
                'total_calls': org_stats['total_calls'],
                'users': {
                    username: {
                        'total_calls': user_stats['total_calls'],
                        'status_codes': dict(user_stats['status_codes']),
                        'endpoints': dict(user_stats['endpoints']),
                        'avg_execution_time': float(user_stats['total_execution_time']) / user_stats['total_calls']
                    }
                    for username, user_stats in org_stats['users'].items()
                }
            }
            for org_name, org_stats in org_data.items()
        }

        # Convert defaultdict to regular dict for JSON serialization
        summary['status_code_distribution'] = dict(summary['status_code_distribution'])
        summary['hourly_distribution'] = dict(summary['hourly_distribution'])

        # Save to file with custom JSON encoder
        output_path = options['output']
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2, default=self.decimal_default)

        self.stdout.write(
            self.style.SUCCESS(
                f'Summary generated for {target_date} and saved to {output_path}'
            )
        ) 