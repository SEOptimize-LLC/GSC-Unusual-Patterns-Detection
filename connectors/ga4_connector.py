"""
Google Analytics 4 Data API Connector
"""
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from typing import Optional, List, Dict, Any
import json


class GA4Connector:
    """Google Analytics 4 Data API connector"""

    def __init__(self, credentials: Credentials):
        self.credentials = credentials
        self.service = self._build_service()
        self.admin_service = self._build_admin_service()

    def _build_service(self):
        """Build the GA4 Data API service"""
        try:
            return build('analyticsdata', 'v1beta', credentials=self.credentials)
        except Exception as e:
            st.error(f"Error building GA4 Data API service: {str(e)}")
            return None

    def _build_admin_service(self):
        """Build the GA4 Admin API service for listing properties"""
        try:
            return build('analyticsadmin', 'v1beta', credentials=self.credentials)
        except Exception as e:
            # Admin API might not be available, which is ok
            return None

    def get_properties(self) -> List[Dict[str, str]]:
        """Get list of GA4 properties the user has access to"""
        if not self.admin_service:
            return []

        properties = []
        try:
            # List all accounts
            accounts = self.admin_service.accounts().list().execute()

            for account in accounts.get('accounts', []):
                account_name = account['name']

                # List properties for each account
                props = self.admin_service.properties().list(
                    filter=f"parent:{account_name}"
                ).execute()

                for prop in props.get('properties', []):
                    properties.append({
                        'property_id': prop['name'].replace('properties/', ''),
                        'display_name': prop.get('displayName', 'Unknown'),
                        'account': account.get('displayName', 'Unknown Account')
                    })

        except HttpError as e:
            if e.resp.status == 403:
                st.warning("GA4 Admin API access not available. Please enter property ID manually.")
            else:
                st.error(f"Error listing GA4 properties: {e}")
        except Exception as e:
            st.warning(f"Could not list GA4 properties: {str(e)}")

        return properties

    def fetch_data(
        self,
        property_id: str,
        start_date: str,
        end_date: str,
        dimensions: List[str] = None,
        metrics: List[str] = None,
        dimension_filter: Dict = None,
        max_rows: int = 10000
    ) -> pd.DataFrame:
        """
        Fetch GA4 data using the Data API

        Args:
            property_id: GA4 property ID (just the number, e.g., "123456789")
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            dimensions: List of dimensions to include
            metrics: List of metrics to include
            dimension_filter: Optional filter expression
            max_rows: Maximum rows to return

        Returns:
            DataFrame with the requested data
        """
        if not self.service:
            return pd.DataFrame()

        # Default dimensions and metrics
        dimensions = dimensions or ['date']
        metrics = metrics or ['sessions', 'totalUsers', 'screenPageViews']

        # Build request body
        request_body = {
            'dateRanges': [{'startDate': start_date, 'endDate': end_date}],
            'dimensions': [{'name': dim} for dim in dimensions],
            'metrics': [{'name': metric} for metric in metrics],
            'limit': max_rows
        }

        if dimension_filter:
            request_body['dimensionFilter'] = dimension_filter

        try:
            response = self.service.properties().runReport(
                property=f'properties/{property_id}',
                body=request_body
            ).execute()

            # Parse response
            rows = []
            dimension_headers = [h['name'] for h in response.get('dimensionHeaders', [])]
            metric_headers = [h['name'] for h in response.get('metricHeaders', [])]

            for row in response.get('rows', []):
                row_data = {}

                # Add dimensions
                for i, dim_value in enumerate(row.get('dimensionValues', [])):
                    row_data[dimension_headers[i]] = dim_value['value']

                # Add metrics
                for i, metric_value in enumerate(row.get('metricValues', [])):
                    value = metric_value['value']
                    # Convert to appropriate type
                    try:
                        if '.' in value:
                            row_data[metric_headers[i]] = float(value)
                        else:
                            row_data[metric_headers[i]] = int(value)
                    except ValueError:
                        row_data[metric_headers[i]] = value

                rows.append(row_data)

            df = pd.DataFrame(rows)

            # Convert date column to datetime if present
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')

            return df

        except HttpError as e:
            error_details = json.loads(e.content.decode())
            error_message = error_details.get('error', {}).get('message', 'Unknown error')
            st.error(f"GA4 API Error: {error_message}")
            return pd.DataFrame()
        except Exception as e:
            st.error(f"Error fetching GA4 data: {str(e)}")
            return pd.DataFrame()

    def fetch_traffic_by_channel(
        self,
        property_id: str,
        start_date: str,
        end_date: str,
        max_rows: int = 10000
    ) -> pd.DataFrame:
        """Fetch traffic data segmented by channel grouping"""
        return self.fetch_data(
            property_id=property_id,
            start_date=start_date,
            end_date=end_date,
            dimensions=['date', 'sessionDefaultChannelGroup'],
            metrics=['sessions', 'totalUsers', 'newUsers', 'screenPageViews', 'bounceRate'],
            max_rows=max_rows
        )

    def fetch_traffic_by_source_medium(
        self,
        property_id: str,
        start_date: str,
        end_date: str,
        max_rows: int = 10000
    ) -> pd.DataFrame:
        """Fetch traffic data segmented by source/medium"""
        return self.fetch_data(
            property_id=property_id,
            start_date=start_date,
            end_date=end_date,
            dimensions=['date', 'sessionSource', 'sessionMedium'],
            metrics=['sessions', 'totalUsers', 'newUsers', 'screenPageViews'],
            max_rows=max_rows
        )

    def fetch_organic_search_traffic(
        self,
        property_id: str,
        start_date: str,
        end_date: str,
        max_rows: int = 10000
    ) -> pd.DataFrame:
        """Fetch only organic search traffic for comparison with GSC"""
        dimension_filter = {
            'filter': {
                'fieldName': 'sessionDefaultChannelGroup',
                'stringFilter': {
                    'matchType': 'EXACT',
                    'value': 'Organic Search'
                }
            }
        }

        return self.fetch_data(
            property_id=property_id,
            start_date=start_date,
            end_date=end_date,
            dimensions=['date'],
            metrics=['sessions', 'totalUsers', 'newUsers', 'screenPageViews', 'bounceRate', 'averageSessionDuration'],
            dimension_filter=dimension_filter,
            max_rows=max_rows
        )

    def fetch_yoy_data(
        self,
        property_id: str,
        current_start: str,
        current_end: str,
        dimensions: List[str] = None,
        metrics: List[str] = None,
        max_rows: int = 10000
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch current period and year-over-year comparison data

        Returns:
            Dictionary with 'current' and 'previous' DataFrames
        """
        # Parse dates
        current_start_dt = datetime.strptime(current_start, '%Y-%m-%d')
        current_end_dt = datetime.strptime(current_end, '%Y-%m-%d')

        # Calculate previous year dates
        previous_start_dt = current_start_dt - timedelta(days=365)
        previous_end_dt = current_end_dt - timedelta(days=365)

        previous_start = previous_start_dt.strftime('%Y-%m-%d')
        previous_end = previous_end_dt.strftime('%Y-%m-%d')

        dimensions = dimensions or ['date']
        metrics = metrics or ['sessions', 'totalUsers', 'screenPageViews']

        # Fetch both periods
        current_df = self.fetch_data(
            property_id=property_id,
            start_date=current_start,
            end_date=current_end,
            dimensions=dimensions,
            metrics=metrics,
            max_rows=max_rows
        )

        previous_df = self.fetch_data(
            property_id=property_id,
            start_date=previous_start,
            end_date=previous_end,
            dimensions=dimensions,
            metrics=metrics,
            max_rows=max_rows
        )

        return {
            'current': current_df,
            'previous': previous_df,
            'current_period': (current_start, current_end),
            'previous_period': (previous_start, previous_end)
        }

    def fetch_device_breakdown(
        self,
        property_id: str,
        start_date: str,
        end_date: str,
        max_rows: int = 10000
    ) -> pd.DataFrame:
        """Fetch traffic breakdown by device category"""
        return self.fetch_data(
            property_id=property_id,
            start_date=start_date,
            end_date=end_date,
            dimensions=['date', 'deviceCategory'],
            metrics=['sessions', 'totalUsers', 'screenPageViews', 'bounceRate'],
            max_rows=max_rows
        )

    def fetch_geo_breakdown(
        self,
        property_id: str,
        start_date: str,
        end_date: str,
        max_rows: int = 10000
    ) -> pd.DataFrame:
        """Fetch traffic breakdown by country"""
        return self.fetch_data(
            property_id=property_id,
            start_date=start_date,
            end_date=end_date,
            dimensions=['date', 'country'],
            metrics=['sessions', 'totalUsers', 'screenPageViews'],
            max_rows=max_rows
        )
