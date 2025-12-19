"""
Google Search Console API Connector
"""
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from typing import Optional, List, Dict, Any
import json

from config.settings import (
    GSC_API_SERVICE,
    GSC_API_VERSION,
    MAX_GSC_ROWS,
    GSC_DIMENSIONS,
    GSC_METRICS
)


class GSCConnector:
    """Google Search Console API connector with full data extraction capabilities"""

    def __init__(self, credentials: Credentials):
        self.credentials = credentials
        self.service = self._build_service()

    def _build_service(self):
        """Build the Search Console API service"""
        try:
            return build(GSC_API_SERVICE, GSC_API_VERSION, credentials=self.credentials)
        except Exception as e:
            st.error(f"Error building Search Console service: {str(e)}")
            return None

    def get_verified_sites(self) -> List[str]:
        """Get list of verified sites in the account"""
        if not self.service:
            return []

        try:
            sites = self.service.sites().list().execute()
            return [site['siteUrl'] for site in sites.get('siteEntry', [])]
        except HttpError as e:
            st.error(f"HTTP Error getting sites: {e}")
            return []
        except Exception as e:
            st.error(f"Error getting verified sites: {str(e)}")
            return []

    def fetch_data(
        self,
        site_url: str,
        start_date: str,
        end_date: str,
        dimensions: List[str] = None,
        filters: List[Dict] = None,
        max_rows: int = None,
        search_type: str = 'web'
    ) -> pd.DataFrame:
        """
        Fetch Search Console data with pagination support

        Args:
            site_url: The site URL to query
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            dimensions: List of dimensions (query, page, date, country, device)
            filters: Optional dimension filters
            max_rows: Maximum rows to fetch
            search_type: Type of search (web, image, video, news)

        Returns:
            DataFrame with the requested data
        """
        if not self.service:
            return pd.DataFrame()

        dimensions = dimensions or ['date', 'query']
        max_rows = max_rows or MAX_GSC_ROWS

        all_data = []
        start_row = 0
        rows_per_request = min(25000, max_rows)

        while start_row < max_rows:
            request_body = {
                'startDate': start_date,
                'endDate': end_date,
                'dimensions': dimensions,
                'rowLimit': rows_per_request,
                'startRow': start_row,
                'type': search_type
            }

            if filters:
                request_body['dimensionFilterGroups'] = [{'filters': filters}]

            try:
                response = self.service.searchanalytics().query(
                    siteUrl=site_url,
                    body=request_body
                ).execute()

                rows = response.get('rows', [])
                if not rows:
                    break

                for row in rows:
                    row_data = {}
                    for i, dim in enumerate(dimensions):
                        row_data[dim] = row['keys'][i]

                    row_data.update({
                        'clicks': row.get('clicks', 0),
                        'impressions': row.get('impressions', 0),
                        'ctr': row.get('ctr', 0),
                        'position': row.get('position', 0)
                    })
                    all_data.append(row_data)

                start_row += len(rows)

                if len(rows) < rows_per_request:
                    break

            except HttpError as e:
                error_details = json.loads(e.content.decode())
                error_message = error_details.get('error', {}).get('message', 'Unknown error')
                st.error(f"GSC API Error: {error_message}")
                break
            except Exception as e:
                st.error(f"Error fetching GSC data: {str(e)}")
                break

        df = pd.DataFrame(all_data)

        # Convert date column to datetime if present
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])

        return df

    def fetch_data_by_date_range(
        self,
        site_url: str,
        start_date: str,
        end_date: str,
        dimensions: List[str] = None,
        max_rows: int = None
    ) -> pd.DataFrame:
        """Fetch data ensuring date is included for time series analysis"""
        dimensions = dimensions or []
        if 'date' not in dimensions:
            dimensions = ['date'] + dimensions

        return self.fetch_data(
            site_url=site_url,
            start_date=start_date,
            end_date=end_date,
            dimensions=dimensions,
            max_rows=max_rows
        )

    def fetch_yoy_data(
        self,
        site_url: str,
        current_start: str,
        current_end: str,
        dimensions: List[str] = None,
        max_rows: int = None
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

        # Fetch both periods
        current_df = self.fetch_data_by_date_range(
            site_url=site_url,
            start_date=current_start,
            end_date=current_end,
            dimensions=dimensions,
            max_rows=max_rows
        )

        previous_df = self.fetch_data_by_date_range(
            site_url=site_url,
            start_date=previous_start,
            end_date=previous_end,
            dimensions=dimensions,
            max_rows=max_rows
        )

        return {
            'current': current_df,
            'previous': previous_df,
            'current_period': (current_start, current_end),
            'previous_period': (previous_start, previous_end)
        }

    def fetch_segmented_data(
        self,
        site_url: str,
        start_date: str,
        end_date: str,
        segment_by: str = 'query',
        max_rows: int = None
    ) -> pd.DataFrame:
        """
        Fetch data segmented by a specific dimension

        Args:
            segment_by: One of 'query', 'page', 'country', 'device'
        """
        dimensions = ['date', segment_by]
        return self.fetch_data_by_date_range(
            site_url=site_url,
            start_date=start_date,
            end_date=end_date,
            dimensions=dimensions,
            max_rows=max_rows
        )

    def fetch_multi_segment_data(
        self,
        site_url: str,
        start_date: str,
        end_date: str,
        segments: List[str] = None,
        max_rows: int = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch data segmented by multiple dimensions

        Returns:
            Dictionary with segment name as key and DataFrame as value
        """
        segments = segments or ['query', 'page', 'country', 'device']
        results = {}

        for segment in segments:
            results[segment] = self.fetch_segmented_data(
                site_url=site_url,
                start_date=start_date,
                end_date=end_date,
                segment_by=segment,
                max_rows=max_rows
            )

        return results
