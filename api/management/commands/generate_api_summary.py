from django.core.management.base import BaseCommand
from django.utils import timezone
from datetime import datetime, timedelta
from drf_api_logger.models import APILogsModel
from api.models import Organization, User
import json
from collections import defaultdict
from decimal import Decimal
import os
from statistics import median

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
        # Add this at the beginning of the handle method
        COUNTED_ENDPOINTS = [
            'upload_file/',
            'perform_analysis/',
            'perform_conflict_check/',
            'chat/',
            'brainstorm_chat/',
            'explain_text/'
        ]

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
            'hourly_distribution': defaultdict(int),
            'endpoint_distribution': defaultdict(int)
        }

        # Get all API calls for the day
        api_logs = APILogsModel.objects.filter(
            added_on__date=target_date
        )

        # Calculate total API calls
        total_calls = sum(1 for log in api_logs if any(endpoint in log.api for endpoint in COUNTED_ENDPOINTS))
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

        # Modify the statistics tracking to include org and user level stats
        endpoint_stats = defaultdict(lambda: {
            'count': 0,
            'total_time': 0,
            'times': [],
            'max_time': 0,
            'min_time': float('inf'),
            'avg_time': 0,
            'organizations': defaultdict(lambda: {
                'count': 0,
                'total_time': 0,
                'times': [],
                'max_time': 0,
                'min_time': float('inf'),
                'users': defaultdict(lambda: {
                    'count': 0,
                    'total_time': 0,
                    'times': [],
                    'max_time': 0,
                    'min_time': float('inf')
                })
            })
        })

        # Add this near the beginning of the handle method
        def format_log_entry(log, org_name, username):
            return {
                'id': log.id,
                'endpoint': log.api,
                'method': log.method,
                'status_code': log.status_code,
                'execution_time': f"{log.execution_time:.5f}s",
                'timestamp': (log.added_on + timedelta(hours=5, minutes=30)).strftime('%d/%m/%y %H:%M:%S'),
                'organization': org_name,
                'user': username
            }

        for log in api_logs:
            # Extract just the endpoint name from the full URL
            endpoint = None
            for counted_endpoint in COUNTED_ENDPOINTS:
                if counted_endpoint in log.api:
                    endpoint = counted_endpoint
                    break
                    
            if endpoint:
                summary['endpoint_distribution'][endpoint] += 1
                # Only process if it's a counted endpoint
                if not any(endpoint in log.api for endpoint in COUNTED_ENDPOINTS):
                    continue

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

                # Update endpoint statistics
                execution_time = float(log.execution_time)
                stats = endpoint_stats[endpoint]
                stats['count'] += 1
                stats['total_time'] += execution_time
                stats['times'].append(execution_time)
                stats['max_time'] = max(stats['max_time'], execution_time)
                stats['min_time'] = min(stats['min_time'], execution_time)

                # Update organization-level stats
                org_stats = stats['organizations'][org_name]
                org_stats['count'] += 1
                org_stats['total_time'] += execution_time
                org_stats['times'].append(execution_time)
                org_stats['max_time'] = max(org_stats['max_time'], execution_time)
                org_stats['min_time'] = min(org_stats['min_time'], execution_time)

                # Update user-level stats
                user_stats = org_stats['users'][username]
                user_stats['count'] += 1
                user_stats['total_time'] += execution_time
                user_stats['times'].append(execution_time)
                user_stats['max_time'] = max(user_stats['max_time'], execution_time)
                user_stats['min_time'] = min(user_stats['min_time'], execution_time)

        # Calculate averages and find peak hour
        summary['average_execution_time'] = float(total_execution_time) / total_calls
        peak_hour = max(summary['hourly_distribution'].items(), key=lambda x: x[1])
        summary['peak_hour'] = f"{peak_hour[0]}:00 IST ({peak_hour[1]} calls)"

        # Calculate final statistics including org and user breakdowns
        summary['endpoint_distribution'] = {
            endpoint: {
                'count': stats['count'],
                'avg_time': f"{stats['total_time'] / stats['count']:.4f}s",
                'max_time': f"{stats['max_time']:.4f}s",
                'min_time': f"{stats['min_time']:.4f}s",
                'median_time': f"{median(stats['times']):.4f}s",
                'percentage': f"{(stats['count'] / total_calls * 100):.1f}%",
                'organizations': {
                    org_name: {
                        'count': org_stats['count'],
                        'avg_time': f"{org_stats['total_time'] / org_stats['count']:.4f}s",
                        'max_time': f"{org_stats['max_time']:.4f}s",
                        'min_time': f"{org_stats['min_time']:.4f}s",
                        'median_time': f"{median(org_stats['times']):.4f}s",
                        'percentage': f"{(org_stats['count'] / stats['count'] * 100):.1f}%",
                        'users': {
                            username: {
                                'count': user_stats['count'],
                                'avg_time': f"{user_stats['total_time'] / user_stats['count']:.4f}s",
                                'max_time': f"{user_stats['max_time']:.4f}s",
                                'min_time': f"{user_stats['min_time']:.4f}s",
                                'median_time': f"{median(user_stats['times']):.4f}s",
                                'percentage': f"{(user_stats['count'] / org_stats['count'] * 100):.1f}%"
                            }
                            for username, user_stats in org_stats['users'].items()
                        }
                    }
                    for org_name, org_stats in stats['organizations'].items()
                }
            }
            for endpoint, stats in endpoint_stats.items()
        }

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
        summary['endpoint_distribution'] = dict(summary['endpoint_distribution'])

        # Store all API logs
        formatted_logs = []
        for log in api_logs:
            if any(endpoint in log.api for endpoint in COUNTED_ENDPOINTS):
                try:
                    headers = json.loads(log.headers) if isinstance(log.headers, str) else log.headers
                except json.JSONDecodeError:
                    headers = {}
                
                org_id = headers.get('X_ORGANIZATION_ID', headers.get('x-organization-id', 'N/A'))
                username = headers.get('USER', headers.get('user', 'Anonymous'))
                
                org_name = 'No Organization'
                if org_id and org_id != 'N/A':
                    try:
                        org = Organization.objects.get(id=org_id)
                        org_name = org.name
                    except Organization.DoesNotExist:
                        org_name = f'Unknown Org ({org_id})'
                
                formatted_logs.append(format_log_entry(log, org_name, username))

        # Add logs to summary
        summary['logs'] = formatted_logs

        # Save to file with custom JSON encoder
        output_path = options['output']
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2, default=self.decimal_default)

        self.stdout.write(
            self.style.SUCCESS(
                f'Summary generated for {target_date} and saved to {output_path}'
            )
        ) 