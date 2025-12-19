"""
Data Segmentation Module for Multi-dimensional Analysis
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
import re


class DataSegmenter:
    """
    Segments data by various dimensions for pattern analysis

    Supports segmentation by:
    - URL patterns (directories, file types)
    - Search intent (informational, navigational, transactional)
    - Device type
    - Geographic location
    - Query clusters
    """

    # Search intent patterns
    INTENT_PATTERNS = {
        'informational': [
            r'\b(what|how|why|when|where|who|which|guide|tutorial|tips|learn|understand)\b',
            r'\b(definition|meaning|example|explain|difference|vs|compare)\b'
        ],
        'navigational': [
            r'\b(login|sign in|signin|account|dashboard|portal|official)\b',
            r'\b(website|site|page|home|contact)\b'
        ],
        'transactional': [
            r'\b(buy|purchase|order|shop|price|cost|cheap|discount|deal|coupon)\b',
            r'\b(free|download|subscribe|register|signup|sign up|trial)\b'
        ],
        'commercial': [
            r'\b(best|top|review|comparison|vs|versus|alternative)\b',
            r'\b(recommended|rating|rank|list)\b'
        ]
    }

    def __init__(self):
        self.intent_patterns = {
            intent: [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
            for intent, patterns in self.INTENT_PATTERNS.items()
        }

    def segment_by_url_directory(
        self,
        df: pd.DataFrame,
        url_col: str = 'page'
    ) -> pd.DataFrame:
        """
        Segment URLs by directory structure

        Args:
            df: DataFrame with URL column
            url_col: Name of URL column

        Returns:
            DataFrame with directory segments
        """
        df = df.copy()

        if url_col not in df.columns:
            return df

        def extract_directory(url):
            try:
                # Remove protocol and domain
                path = re.sub(r'^https?://[^/]+', '', str(url))
                # Get first directory level
                parts = path.strip('/').split('/')
                return '/' + parts[0] if parts and parts[0] else '/'
            except:
                return '/'

        def extract_full_path(url):
            try:
                path = re.sub(r'^https?://[^/]+', '', str(url))
                return path if path else '/'
            except:
                return '/'

        df['url_directory'] = df[url_col].apply(extract_directory)
        df['url_path'] = df[url_col].apply(extract_full_path)

        return df

    def segment_by_url_type(
        self,
        df: pd.DataFrame,
        url_col: str = 'page'
    ) -> pd.DataFrame:
        """
        Segment URLs by page type (blog, product, category, etc.)

        Args:
            df: DataFrame with URL column
            url_col: Name of URL column

        Returns:
            DataFrame with URL type segments
        """
        df = df.copy()

        if url_col not in df.columns:
            return df

        def classify_url(url):
            url_lower = str(url).lower()

            if re.search(r'/blog/|/article/|/post/|/news/', url_lower):
                return 'blog'
            elif re.search(r'/product/|/item/|/p/', url_lower):
                return 'product'
            elif re.search(r'/category/|/cat/|/c/', url_lower):
                return 'category'
            elif re.search(r'/tag/|/tags/', url_lower):
                return 'tag'
            elif re.search(r'/search|/s\?|/results', url_lower):
                return 'search'
            elif re.search(r'/checkout|/cart|/basket', url_lower):
                return 'conversion'
            elif re.search(r'/about|/contact|/faq|/help', url_lower):
                return 'info'
            elif url_lower.rstrip('/').count('/') <= 3:  # Near homepage
                return 'landing'
            else:
                return 'other'

        df['url_type'] = df[url_col].apply(classify_url)

        return df

    def segment_by_intent(
        self,
        df: pd.DataFrame,
        query_col: str = 'query'
    ) -> pd.DataFrame:
        """
        Segment queries by search intent

        Args:
            df: DataFrame with query column
            query_col: Name of query column

        Returns:
            DataFrame with intent classification
        """
        df = df.copy()

        if query_col not in df.columns:
            return df

        def classify_intent(query):
            query = str(query).lower()

            scores = {}
            for intent, patterns in self.intent_patterns.items():
                scores[intent] = sum(1 for p in patterns if p.search(query))

            if max(scores.values()) == 0:
                return 'unknown'

            return max(scores, key=scores.get)

        df['search_intent'] = df[query_col].apply(classify_intent)

        return df

    def segment_by_query_length(
        self,
        df: pd.DataFrame,
        query_col: str = 'query'
    ) -> pd.DataFrame:
        """
        Segment queries by length (head, torso, long-tail)

        Args:
            df: DataFrame with query column
            query_col: Name of query column

        Returns:
            DataFrame with query length classification
        """
        df = df.copy()

        if query_col not in df.columns:
            return df

        def classify_length(query):
            words = len(str(query).split())
            if words <= 2:
                return 'head'
            elif words <= 4:
                return 'torso'
            else:
                return 'long_tail'

        df['query_length_type'] = df[query_col].apply(classify_length)
        df['word_count'] = df[query_col].apply(lambda x: len(str(x).split()))

        return df

    def aggregate_by_segment(
        self,
        df: pd.DataFrame,
        segment_col: str,
        metric_cols: List[str] = None,
        date_col: str = None
    ) -> pd.DataFrame:
        """
        Aggregate metrics by segment

        Args:
            df: DataFrame with segment column
            segment_col: Column to group by
            metric_cols: Metrics to aggregate
            date_col: Optional date column for time series

        Returns:
            Aggregated DataFrame
        """
        if segment_col not in df.columns:
            raise ValueError(f"Column '{segment_col}' not found")

        metric_cols = metric_cols or ['clicks', 'impressions', 'position']
        metric_cols = [c for c in metric_cols if c in df.columns]

        agg_dict = {}
        for col in metric_cols:
            if col in ['position', 'ctr', 'averageSessionDuration', 'bounceRate']:
                agg_dict[col] = 'mean'
            else:
                agg_dict[col] = 'sum'

        group_cols = [segment_col]
        if date_col and date_col in df.columns:
            group_cols = [date_col, segment_col]

        result = df.groupby(group_cols).agg(agg_dict).reset_index()

        # Recalculate CTR if we have clicks and impressions
        if 'clicks' in result.columns and 'impressions' in result.columns:
            result['ctr'] = result['clicks'] / result['impressions'].replace(0, 1)

        return result

    def get_segment_summary(
        self,
        df: pd.DataFrame,
        segment_col: str,
        metric_cols: List[str] = None
    ) -> Dict[str, Any]:
        """
        Generate summary statistics for each segment

        Args:
            df: DataFrame with segment column
            segment_col: Column to analyze
            metric_cols: Metrics to summarize

        Returns:
            Dictionary with segment summaries
        """
        metric_cols = metric_cols or ['clicks', 'impressions', 'ctr', 'position']
        metric_cols = [c for c in metric_cols if c in df.columns]

        agg_df = self.aggregate_by_segment(df, segment_col, metric_cols)

        summary = {
            'total_segments': df[segment_col].nunique(),
            'segments': {}
        }

        total_clicks = agg_df['clicks'].sum() if 'clicks' in agg_df.columns else 0
        total_impressions = agg_df['impressions'].sum() if 'impressions' in agg_df.columns else 0

        for _, row in agg_df.iterrows():
            segment = row[segment_col]
            summary['segments'][segment] = {}

            for col in metric_cols:
                summary['segments'][segment][col] = row[col]

            # Calculate share
            if 'clicks' in agg_df.columns and total_clicks > 0:
                summary['segments'][segment]['clicks_share'] = row['clicks'] / total_clicks * 100

            if 'impressions' in agg_df.columns and total_impressions > 0:
                summary['segments'][segment]['impressions_share'] = row['impressions'] / total_impressions * 100

        return summary

    def identify_segment_anomalies(
        self,
        current_df: pd.DataFrame,
        previous_df: pd.DataFrame,
        segment_col: str,
        metric_col: str = 'clicks',
        threshold_pct: float = 20.0
    ) -> List[Dict]:
        """
        Identify segments with unusual changes compared to previous period

        Args:
            current_df: Current period data
            previous_df: Previous period data
            segment_col: Segment column to analyze
            metric_col: Metric to compare
            threshold_pct: Percentage change threshold

        Returns:
            List of segments with unusual changes
        """
        current_agg = self.aggregate_by_segment(current_df, segment_col, [metric_col])
        previous_agg = self.aggregate_by_segment(previous_df, segment_col, [metric_col])

        merged = current_agg.merge(
            previous_agg,
            on=segment_col,
            suffixes=('_current', '_previous'),
            how='outer'
        ).fillna(0)

        current_col = f'{metric_col}_current'
        previous_col = f'{metric_col}_previous'

        merged['change'] = merged[current_col] - merged[previous_col]
        merged['pct_change'] = (merged['change'] / merged[previous_col].replace(0, 1)) * 100

        # Find segments with changes exceeding threshold
        anomalies = merged[abs(merged['pct_change']) > threshold_pct]

        results = []
        for _, row in anomalies.iterrows():
            results.append({
                'segment': row[segment_col],
                'current_value': row[current_col],
                'previous_value': row[previous_col],
                'change': row['change'],
                'pct_change': row['pct_change'],
                'direction': 'increase' if row['change'] > 0 else 'decrease'
            })

        # Sort by absolute percentage change
        results.sort(key=lambda x: abs(x['pct_change']), reverse=True)

        return results

    def apply_all_segmentations(
        self,
        df: pd.DataFrame,
        query_col: str = 'query',
        url_col: str = 'page'
    ) -> pd.DataFrame:
        """
        Apply all available segmentations to the DataFrame

        Args:
            df: Input DataFrame
            query_col: Query column name
            url_col: URL column name

        Returns:
            DataFrame with all segmentations applied
        """
        result = df.copy()

        if query_col in result.columns:
            result = self.segment_by_intent(result, query_col)
            result = self.segment_by_query_length(result, query_col)

        if url_col in result.columns:
            result = self.segment_by_url_directory(result, url_col)
            result = self.segment_by_url_type(result, url_col)

        return result
