"""
Utility functions and helpers
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Tuple


def format_number(value: float, decimal_places: int = 0) -> str:
    """Format number with thousands separator"""
    if pd.isna(value):
        return "N/A"
    if decimal_places == 0:
        return f"{int(value):,}"
    return f"{value:,.{decimal_places}f}"


def format_percentage(value: float, decimal_places: int = 2) -> str:
    """Format value as percentage"""
    if pd.isna(value):
        return "N/A"
    return f"{value * 100:.{decimal_places}f}%"


def format_change(value: float, is_percentage: bool = False) -> str:
    """Format change value with sign and color indicator"""
    if pd.isna(value):
        return "N/A"

    sign = "+" if value > 0 else ""
    if is_percentage:
        return f"{sign}{value:.1f}%"
    return f"{sign}{format_number(value)}"


class DataProcessor:
    """
    Data preprocessing and transformation utilities
    """

    @staticmethod
    def clean_gsc_data(df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and standardize GSC data

        Args:
            df: Raw GSC DataFrame

        Returns:
            Cleaned DataFrame
        """
        df = df.copy()

        # Standardize column names
        df.columns = df.columns.str.lower().str.strip()

        # Handle date column
        date_cols = [c for c in df.columns if 'date' in c.lower()]
        if date_cols:
            df['date'] = pd.to_datetime(df[date_cols[0]])
            if date_cols[0] != 'date':
                df = df.drop(columns=[date_cols[0]])

        # Clean numeric columns
        numeric_cols = ['clicks', 'impressions', 'ctr', 'position']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = DataProcessor._clean_numeric_column(df[col])

        # Sort by date
        if 'date' in df.columns:
            df = df.sort_values('date').reset_index(drop=True)

        return df

    @staticmethod
    def _clean_numeric_column(series: pd.Series) -> pd.Series:
        """Clean a numeric column"""
        series = series.astype(str).str.replace(',', '').str.replace(' ', '')

        # Handle percentage columns
        if series.str.contains('%').any():
            series = series.str.rstrip('%').astype(float) / 100
        else:
            series = pd.to_numeric(series, errors='coerce')

        return series

    @staticmethod
    def aggregate_daily(
        df: pd.DataFrame,
        date_col: str = 'date',
        metric_cols: List[str] = None
    ) -> pd.DataFrame:
        """
        Aggregate data to daily level

        Args:
            df: DataFrame with potentially multiple rows per day
            date_col: Date column name
            metric_cols: Metrics to aggregate

        Returns:
            Daily aggregated DataFrame
        """
        metric_cols = metric_cols or ['clicks', 'impressions', 'position']
        metric_cols = [c for c in metric_cols if c in df.columns]

        agg_dict = {}
        for col in metric_cols:
            if col in ['position', 'ctr', 'bounceRate', 'averageSessionDuration']:
                agg_dict[col] = 'mean'
            else:
                agg_dict[col] = 'sum'

        result = df.groupby(date_col).agg(agg_dict).reset_index()

        # Recalculate CTR if possible
        if 'clicks' in result.columns and 'impressions' in result.columns:
            result['ctr'] = result['clicks'] / result['impressions'].replace(0, 1)

        return result

    @staticmethod
    def calculate_rolling_stats(
        df: pd.DataFrame,
        metric_cols: List[str],
        windows: List[int] = None
    ) -> pd.DataFrame:
        """
        Calculate rolling statistics for metrics

        Args:
            df: DataFrame with metrics
            metric_cols: Columns to calculate stats for
            windows: Rolling window sizes

        Returns:
            DataFrame with rolling stats added
        """
        df = df.copy()
        windows = windows or [7, 14, 30]

        for col in metric_cols:
            if col not in df.columns:
                continue

            for window in windows:
                df[f'{col}_rolling_mean_{window}'] = df[col].rolling(window, min_periods=1).mean()
                df[f'{col}_rolling_std_{window}'] = df[col].rolling(window, min_periods=1).std()

        return df

    @staticmethod
    def detect_data_gaps(
        df: pd.DataFrame,
        date_col: str = 'date'
    ) -> List[Dict[str, Any]]:
        """
        Detect gaps in time series data

        Args:
            df: DataFrame with date column
            date_col: Date column name

        Returns:
            List of gap dictionaries with start/end dates
        """
        if date_col not in df.columns:
            return []

        df = df.sort_values(date_col)
        dates = pd.to_datetime(df[date_col])

        gaps = []
        for i in range(1, len(dates)):
            diff = (dates.iloc[i] - dates.iloc[i-1]).days
            if diff > 1:
                gaps.append({
                    'start': dates.iloc[i-1] + timedelta(days=1),
                    'end': dates.iloc[i] - timedelta(days=1),
                    'days_missing': diff - 1
                })

        return gaps

    @staticmethod
    def get_date_range_from_selection(
        selection: str,
        custom_start: datetime = None,
        custom_end: datetime = None
    ) -> Tuple[str, str]:
        """
        Convert date range selection to start/end dates

        Args:
            selection: Date range option string
            custom_start: Custom start date
            custom_end: Custom end date

        Returns:
            Tuple of (start_date, end_date) as strings
        """
        from config.settings import DATE_RANGE_OPTIONS

        end_date = datetime.now() - timedelta(days=1)

        if selection == "Custom range" and custom_start and custom_end:
            return custom_start.strftime('%Y-%m-%d'), custom_end.strftime('%Y-%m-%d')

        days = DATE_RANGE_OPTIONS.get(selection, 30)
        start_date = end_date - timedelta(days=days)

        return start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')

    @staticmethod
    def merge_gsc_ga4_data(
        gsc_df: pd.DataFrame,
        ga4_df: pd.DataFrame,
        date_col: str = 'date'
    ) -> pd.DataFrame:
        """
        Merge GSC and GA4 data for comparison

        Args:
            gsc_df: GSC DataFrame
            ga4_df: GA4 DataFrame
            date_col: Date column name

        Returns:
            Merged DataFrame
        """
        # Ensure date columns are datetime
        gsc_df = gsc_df.copy()
        ga4_df = ga4_df.copy()

        gsc_df[date_col] = pd.to_datetime(gsc_df[date_col])
        ga4_df[date_col] = pd.to_datetime(ga4_df[date_col])

        # Aggregate both to daily level
        gsc_daily = DataProcessor.aggregate_daily(gsc_df, date_col, ['clicks', 'impressions', 'position'])
        ga4_daily = DataProcessor.aggregate_daily(ga4_df, date_col, ['sessions', 'totalUsers', 'screenPageViews'])

        # Rename columns to avoid conflicts
        gsc_daily = gsc_daily.rename(columns={
            'clicks': 'gsc_clicks',
            'impressions': 'gsc_impressions',
            'position': 'gsc_position',
            'ctr': 'gsc_ctr'
        })

        ga4_daily = ga4_daily.rename(columns={
            'sessions': 'ga4_sessions',
            'totalUsers': 'ga4_users',
            'screenPageViews': 'ga4_pageviews'
        })

        # Merge on date
        merged = gsc_daily.merge(ga4_daily, on=date_col, how='outer')

        return merged.sort_values(date_col)

    @staticmethod
    def calculate_correlation(
        df: pd.DataFrame,
        col1: str,
        col2: str
    ) -> float:
        """Calculate correlation between two columns"""
        if col1 not in df.columns or col2 not in df.columns:
            return np.nan

        return df[col1].corr(df[col2])

    @staticmethod
    def get_summary_stats(
        df: pd.DataFrame,
        metric_cols: List[str] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculate summary statistics for metrics

        Args:
            df: DataFrame with metrics
            metric_cols: Columns to summarize

        Returns:
            Dictionary of statistics by metric
        """
        metric_cols = metric_cols or ['clicks', 'impressions', 'ctr', 'position']
        metric_cols = [c for c in metric_cols if c in df.columns]

        stats = {}
        for col in metric_cols:
            stats[col] = {
                'sum': df[col].sum(),
                'mean': df[col].mean(),
                'median': df[col].median(),
                'std': df[col].std(),
                'min': df[col].min(),
                'max': df[col].max(),
                'count': len(df)
            }

        return stats
