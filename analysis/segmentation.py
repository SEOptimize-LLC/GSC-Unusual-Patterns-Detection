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

    # Expanded search intent patterns for better classification
    INTENT_PATTERNS = {
        'informational': [
            # Question words
            r'\b(what|how|why|when|where|who|which|whose|whom)\b',
            # Learning/education
            r'\b(guide|tutorial|tips|learn|understand|explain|explained|meaning|definition)\b',
            r'\b(course|training|lesson|education|teach|teaching|instructions)\b',
            # Comparison/research
            r'\b(difference|differences|between|compare|comparison|vs|versus)\b',
            # Examples and explanations
            r'\b(example|examples|sample|samples|template|templates)\b',
            r'\b(ideas|inspiration|ways to|steps to|how to)\b',
            # Facts and info
            r'\b(facts|benefits|advantages|disadvantages|pros|cons|features)\b',
            r'\b(history|overview|introduction|basics|beginner|beginners)\b',
            r'\b(cause|causes|reason|reasons|effect|effects|impact)\b',
            # DIY and solutions
            r'\b(diy|homemade|make your own|fix|fixing|solve|solving|solution)\b',
            r'\b(recipe|recipes|calculator|converter|tool)\b',
            # Question patterns
            r'^(is|are|can|does|do|will|should|could|would|has|have|was|were)\s',
            r'\?$'
        ],
        'navigational': [
            # Login/account
            r'\b(login|log in|signin|sign in|logout|sign out|account|my account)\b',
            r'\b(dashboard|portal|admin|panel|console)\b',
            # Official/direct
            r'\b(official|website|site|homepage|home page|webpage)\b',
            r'\b(contact|contact us|about|about us|location|directions|address)\b',
            # Support
            r'\b(support|help|customer service|phone number|email|hours)\b',
            r'\b(faq|faqs|help center|knowledge base)\b',
            # App/platform specific
            r'\b(app|application|download app|mobile app|ios|android)\b',
            # Near me / local
            r'\b(near me|nearby|closest|nearest|local|locations)\b',
            # Brand + product direct searches (short queries)
            r'^[a-z]+ (login|account|support|contact|app)$'
        ],
        'transactional': [
            # Purchase intent
            r'\b(buy|purchase|order|shop|shopping|add to cart|checkout)\b',
            r'\b(price|pricing|cost|costs|fee|fees|rate|rates|quote)\b',
            # Deals and savings
            r'\b(cheap|cheapest|affordable|budget|inexpensive|low cost)\b',
            r'\b(discount|discounts|coupon|coupons|promo|promo code|deal|deals|sale|sales|offer|offers)\b',
            r'\b(free shipping|fast shipping|same day|next day delivery)\b',
            # Subscription/signup
            r'\b(subscribe|subscription|register|registration|signup|sign up|join)\b',
            r'\b(trial|free trial|demo|get started|start now|try)\b',
            # Download/get
            r'\b(download|downloads|free download|get|install)\b',
            # Booking/reservation
            r'\b(book|booking|reserve|reservation|schedule|appointment)\b',
            # Hire/services
            r'\b(hire|hiring|services|service|contractor|agency|company)\b',
            # For sale
            r'\b(for sale|on sale|clearance|outlet|wholesale)\b'
        ],
        'commercial': [
            # Comparison shopping
            r'\b(best|top|top 10|top 5|most popular|popular|leading)\b',
            r'\b(review|reviews|rating|ratings|rated|recommended|recommendation)\b',
            r'\b(comparison|compare|vs|versus|or|better|worse)\b',
            # Alternatives
            r'\b(alternative|alternatives|similar|like|competitor|competitors)\b',
            # Rankings/lists
            r'\b(ranking|rankings|rank|ranked|list|winners|award|awards)\b',
            # Quality assessment
            r'\b(quality|reliable|reliability|worth|value|legit|legitimate|scam|safe)\b',
            r'\b(honest|truthful|real|genuine|authentic|trusted)\b',
            # Year-specific (looking for latest)
            r'\b(2024|2025|latest|newest|updated|new|current)\b',
            # Specific product research
            r'\b(specs|specifications|features|compatibility|requirements)\b',
            r'\b(size|sizes|dimensions|weight|color|colors|model|models|version)\b',
            # Pros/cons
            r'\b(pros and cons|advantages|disadvantages|benefits|drawbacks)\b'
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
        query_col: str = 'query',
        brand_terms: List[str] = None
    ) -> pd.DataFrame:
        """
        Segment queries by search intent

        Args:
            df: DataFrame with query column
            query_col: Name of query column
            brand_terms: Optional list of brand terms for better classification

        Returns:
            DataFrame with intent classification
        """
        df = df.copy()

        if query_col not in df.columns:
            return df

        # Compile brand patterns if provided
        brand_patterns = []
        if brand_terms:
            brand_patterns = [re.compile(re.escape(term), re.IGNORECASE) for term in brand_terms]

        def classify_intent(query):
            query_str = str(query).lower().strip()
            word_count = len(query_str.split())

            # Calculate scores for each intent
            scores = {}
            for intent, patterns in self.intent_patterns.items():
                scores[intent] = sum(1 for p in patterns if p.search(query_str))

            max_score = max(scores.values())

            # If we found a clear match, use it
            if max_score > 0:
                return max(scores, key=scores.get)

            # Smart fallback for unmatched queries
            # Check if it's a branded query (navigational)
            if brand_patterns:
                for pattern in brand_patterns:
                    if pattern.search(query_str):
                        return 'navigational'

            # Heuristics based on query structure
            # Very short queries (1-2 words) without modifiers are usually navigational
            # (people searching for a specific brand, product, or website)
            if word_count <= 2:
                return 'navigational'

            # Medium length queries (3-4 words) without intent words
            # are often commercial investigation
            if word_count <= 4:
                return 'commercial'

            # Longer queries without clear intent are usually informational
            # (people asking questions or looking for specific info)
            return 'informational'

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
        url_col: str = 'page',
        brand_terms: List[str] = None
    ) -> pd.DataFrame:
        """
        Apply all available segmentations to the DataFrame

        Args:
            df: Input DataFrame
            query_col: Query column name
            url_col: URL column name
            brand_terms: Optional brand terms for better intent classification

        Returns:
            DataFrame with all segmentations applied
        """
        result = df.copy()

        if query_col in result.columns:
            result = self.segment_by_intent(result, query_col, brand_terms)
            result = self.segment_by_query_length(result, query_col)

        if url_col in result.columns:
            result = self.segment_by_url_directory(result, url_col)
            result = self.segment_by_url_type(result, url_col)

        return result
