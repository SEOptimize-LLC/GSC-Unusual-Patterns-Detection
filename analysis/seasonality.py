"""
Year-over-Year Seasonality Analysis Module
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from scipy import stats


class SeasonalityAnalyzer:
    """
    Analyzes seasonality patterns and year-over-year changes

    Features:
    - YoY comparison with alignment
    - Seasonal pattern detection
    - Trend decomposition
    - Change point identification
    """

    def __init__(self):
        pass

    def align_yoy_data(
        self,
        current_df: pd.DataFrame,
        previous_df: pd.DataFrame,
        date_col: str = 'date'
    ) -> pd.DataFrame:
        """
        Align current and previous year data for comparison

        Args:
            current_df: Current period DataFrame
            previous_df: Previous year DataFrame
            date_col: Date column name

        Returns:
            Merged DataFrame with aligned dates
        """
        current = current_df.copy()
        previous = previous_df.copy()

        # Ensure date columns are datetime
        current[date_col] = pd.to_datetime(current[date_col])
        previous[date_col] = pd.to_datetime(previous[date_col])

        # Create day of year for alignment
        current['day_of_year'] = current[date_col].dt.dayofyear
        previous['day_of_year'] = previous[date_col].dt.dayofyear

        # Aggregate by day of year if needed
        metric_cols = [c for c in current.columns if c in ['clicks', 'impressions', 'ctr', 'position', 'sessions', 'totalUsers']]

        agg_dict = {col: 'sum' if col not in ['ctr', 'position'] else 'mean' for col in metric_cols}
        agg_dict[date_col] = 'first'

        current_agg = current.groupby('day_of_year').agg(agg_dict).reset_index()
        previous_agg = previous.groupby('day_of_year').agg(agg_dict).reset_index()

        # Rename columns for clarity
        current_agg = current_agg.rename(columns={col: f'{col}_current' for col in metric_cols})
        current_agg = current_agg.rename(columns={date_col: 'date_current'})

        previous_agg = previous_agg.rename(columns={col: f'{col}_previous' for col in metric_cols})
        previous_agg = previous_agg.rename(columns={date_col: 'date_previous'})

        # Merge on day of year
        merged = current_agg.merge(previous_agg, on='day_of_year', how='outer')

        # Calculate YoY changes
        for col in metric_cols:
            current_col = f'{col}_current'
            previous_col = f'{col}_previous'

            if current_col in merged.columns and previous_col in merged.columns:
                # Absolute change
                merged[f'{col}_change'] = merged[current_col] - merged[previous_col]

                # Percentage change
                merged[f'{col}_pct_change'] = (
                    (merged[current_col] - merged[previous_col]) /
                    merged[previous_col].replace(0, np.nan)
                ) * 100

        return merged.sort_values('day_of_year')

    def calculate_yoy_metrics(
        self,
        current_df: pd.DataFrame,
        previous_df: pd.DataFrame,
        metric_cols: List[str] = None
    ) -> Dict[str, Any]:
        """
        Calculate aggregate YoY comparison metrics

        Args:
            current_df: Current period DataFrame
            previous_df: Previous year DataFrame
            metric_cols: Metrics to compare

        Returns:
            Dictionary with YoY comparison
        """
        metric_cols = metric_cols or ['clicks', 'impressions']
        metric_cols = [c for c in metric_cols if c in current_df.columns and c in previous_df.columns]

        results = {}

        for col in metric_cols:
            current_total = current_df[col].sum()
            previous_total = previous_df[col].sum()

            change = current_total - previous_total
            pct_change = (change / previous_total * 100) if previous_total != 0 else 0

            results[col] = {
                'current': current_total,
                'previous': previous_total,
                'change': change,
                'pct_change': pct_change,
                'trend': 'up' if change > 0 else 'down' if change < 0 else 'flat'
            }

        # Calculate overall assessment
        if 'clicks' in results:
            results['overall_trend'] = results['clicks']['trend']
            results['overall_change_pct'] = results['clicks']['pct_change']

        return results

    def detect_seasonal_patterns(
        self,
        df: pd.DataFrame,
        date_col: str = 'date',
        metric_col: str = 'clicks'
    ) -> Dict[str, Any]:
        """
        Detect weekly and monthly seasonal patterns

        Args:
            df: DataFrame with time series data
            date_col: Date column name
            metric_col: Metric to analyze

        Returns:
            Dictionary with seasonal patterns
        """
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])

        patterns = {
            'weekly': {},
            'monthly': {},
            'day_of_week_effect': None,
            'monthly_effect': None
        }

        # Weekly pattern (day of week)
        df['day_of_week'] = df[date_col].dt.dayofweek
        weekly_avg = df.groupby('day_of_week')[metric_col].mean()
        overall_avg = df[metric_col].mean()

        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        patterns['weekly'] = {
            day_names[i]: {
                'average': weekly_avg.get(i, 0),
                'index': (weekly_avg.get(i, 0) / overall_avg * 100) if overall_avg > 0 else 100
            }
            for i in range(7)
        }

        # Best and worst days
        patterns['best_day'] = day_names[weekly_avg.idxmax()] if not weekly_avg.empty else None
        patterns['worst_day'] = day_names[weekly_avg.idxmin()] if not weekly_avg.empty else None

        # Monthly pattern
        df['month'] = df[date_col].dt.month
        monthly_avg = df.groupby('month')[metric_col].mean()

        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        patterns['monthly'] = {
            month_names[i-1]: {
                'average': monthly_avg.get(i, 0),
                'index': (monthly_avg.get(i, 0) / overall_avg * 100) if overall_avg > 0 else 100
            }
            for i in range(1, 13)
        }

        # Statistical test for day of week effect
        if len(df) > 14:  # Need at least 2 weeks
            day_groups = [group[metric_col].values for _, group in df.groupby('day_of_week')]
            if all(len(g) > 0 for g in day_groups):
                try:
                    f_stat, p_value = stats.f_oneway(*day_groups)
                    patterns['day_of_week_effect'] = {
                        'significant': p_value < 0.05,
                        'p_value': p_value,
                        'f_statistic': f_stat
                    }
                except:
                    pass

        return patterns

    def identify_change_points(
        self,
        df: pd.DataFrame,
        date_col: str = 'date',
        metric_col: str = 'clicks',
        threshold: float = 2.0
    ) -> List[Dict]:
        """
        Identify significant change points in the time series

        Args:
            df: DataFrame with time series data
            date_col: Date column name
            metric_col: Metric to analyze
            threshold: Z-score threshold for change detection

        Returns:
            List of change points with details
        """
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(date_col)

        change_points = []

        # Calculate rolling statistics
        window = 7
        df['rolling_mean'] = df[metric_col].rolling(window, min_periods=1).mean()
        df['rolling_std'] = df[metric_col].rolling(window, min_periods=1).std().fillna(1)

        # Calculate day-over-day change
        df['daily_change'] = df[metric_col].diff()
        df['change_zscore'] = (df['daily_change'] - df['daily_change'].mean()) / df['daily_change'].std()

        # Identify significant changes
        significant = df[abs(df['change_zscore']) > threshold]

        for _, row in significant.iterrows():
            change_points.append({
                'date': row[date_col],
                'metric_value': row[metric_col],
                'change': row['daily_change'],
                'z_score': row['change_zscore'],
                'direction': 'increase' if row['daily_change'] > 0 else 'decrease',
                'severity': 'high' if abs(row['change_zscore']) > 3 else 'medium'
            })

        return change_points

    def compare_periods(
        self,
        df: pd.DataFrame,
        period1_start: str,
        period1_end: str,
        period2_start: str,
        period2_end: str,
        date_col: str = 'date',
        metric_cols: List[str] = None
    ) -> Dict[str, Any]:
        """
        Compare two arbitrary time periods

        Args:
            df: DataFrame with time series data
            period1_start, period1_end: First period boundaries
            period2_start, period2_end: Second period boundaries
            date_col: Date column name
            metric_cols: Metrics to compare

        Returns:
            Dictionary with period comparison
        """
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])

        period1_start = pd.to_datetime(period1_start)
        period1_end = pd.to_datetime(period1_end)
        period2_start = pd.to_datetime(period2_start)
        period2_end = pd.to_datetime(period2_end)

        period1_df = df[(df[date_col] >= period1_start) & (df[date_col] <= period1_end)]
        period2_df = df[(df[date_col] >= period2_start) & (df[date_col] <= period2_end)]

        metric_cols = metric_cols or ['clicks', 'impressions']
        metric_cols = [c for c in metric_cols if c in df.columns]

        comparison = {
            'period1': {
                'start': period1_start.strftime('%Y-%m-%d'),
                'end': period1_end.strftime('%Y-%m-%d'),
                'days': (period1_end - period1_start).days + 1
            },
            'period2': {
                'start': period2_start.strftime('%Y-%m-%d'),
                'end': period2_end.strftime('%Y-%m-%d'),
                'days': (period2_end - period2_start).days + 1
            },
            'metrics': {}
        }

        for col in metric_cols:
            p1_total = period1_df[col].sum()
            p2_total = period2_df[col].sum()

            # Daily averages for fair comparison
            p1_daily = p1_total / max(comparison['period1']['days'], 1)
            p2_daily = p2_total / max(comparison['period2']['days'], 1)

            change = p1_total - p2_total
            daily_change = p1_daily - p2_daily
            pct_change = (change / p2_total * 100) if p2_total != 0 else 0

            comparison['metrics'][col] = {
                'period1_total': p1_total,
                'period2_total': p2_total,
                'period1_daily_avg': p1_daily,
                'period2_daily_avg': p2_daily,
                'absolute_change': change,
                'daily_change': daily_change,
                'pct_change': pct_change
            }

        return comparison

    def get_seasonality_summary(
        self,
        df: pd.DataFrame,
        date_col: str = 'date',
        metric_col: str = 'clicks'
    ) -> str:
        """
        Generate a human-readable seasonality summary

        Args:
            df: DataFrame with time series data
            date_col: Date column name
            metric_col: Metric to analyze

        Returns:
            Summary string
        """
        patterns = self.detect_seasonal_patterns(df, date_col, metric_col)

        summary_parts = []

        # Weekly pattern
        if patterns['best_day'] and patterns['worst_day']:
            summary_parts.append(
                f"Weekly pattern: Best day is {patterns['best_day']}, "
                f"worst day is {patterns['worst_day']}."
            )

        # Day of week effect significance
        if patterns['day_of_week_effect']:
            if patterns['day_of_week_effect']['significant']:
                summary_parts.append(
                    "Day of week has a statistically significant effect on performance."
                )
            else:
                summary_parts.append(
                    "No significant day of week pattern detected."
                )

        return " ".join(summary_parts)
