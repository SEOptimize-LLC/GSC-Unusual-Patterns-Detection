"""
Branded vs Generic Keyword Classification Module
"""
import pandas as pd
import re
from typing import List, Dict, Optional, Set
from collections import defaultdict


class BrandClassifier:
    """
    Classifies search queries as branded or generic

    Supports:
    - Manual brand term input
    - Automatic brand term detection
    - Fuzzy matching for brand variations
    """

    def __init__(self, brand_terms: List[str] = None):
        """
        Initialize the classifier

        Args:
            brand_terms: List of brand-related terms to match
        """
        self.brand_terms = self._normalize_terms(brand_terms or [])
        self.brand_patterns = self._compile_patterns()

    def _normalize_terms(self, terms: List[str]) -> Set[str]:
        """Normalize brand terms to lowercase"""
        return {term.lower().strip() for term in terms if term.strip()}

    def _compile_patterns(self) -> List[re.Pattern]:
        """Compile regex patterns for brand matching"""
        patterns = []
        for term in self.brand_terms:
            # Escape special regex characters
            escaped = re.escape(term)
            # Create pattern that matches the term as whole word or part of compound
            pattern = re.compile(rf'\b{escaped}\b|\b{escaped}|{escaped}\b', re.IGNORECASE)
            patterns.append(pattern)
        return patterns

    def set_brand_terms(self, brand_terms: List[str]) -> None:
        """Update brand terms"""
        self.brand_terms = self._normalize_terms(brand_terms)
        self.brand_patterns = self._compile_patterns()

    def add_brand_terms(self, new_terms: List[str]) -> None:
        """Add new brand terms to existing set"""
        self.brand_terms.update(self._normalize_terms(new_terms))
        self.brand_patterns = self._compile_patterns()

    def is_branded(self, query: str) -> bool:
        """
        Check if a query is branded

        Args:
            query: Search query string

        Returns:
            True if query contains brand terms
        """
        if not self.brand_patterns:
            return False

        query_lower = query.lower()
        return any(pattern.search(query_lower) for pattern in self.brand_patterns)

    def classify_query(self, query: str) -> str:
        """
        Classify a single query

        Args:
            query: Search query string

        Returns:
            'branded' or 'generic'
        """
        return 'branded' if self.is_branded(query) else 'generic'

    def classify_dataframe(
        self,
        df: pd.DataFrame,
        query_col: str = 'query'
    ) -> pd.DataFrame:
        """
        Classify all queries in a DataFrame

        Args:
            df: DataFrame with query column
            query_col: Name of the query column

        Returns:
            DataFrame with 'query_type' column added
        """
        df_result = df.copy()

        if query_col not in df_result.columns:
            raise ValueError(f"Column '{query_col}' not found in DataFrame")

        df_result['query_type'] = df_result[query_col].apply(self.classify_query)
        df_result['is_branded'] = df_result['query_type'] == 'branded'

        return df_result

    def get_classification_summary(
        self,
        df: pd.DataFrame,
        metric_cols: List[str] = None
    ) -> Dict:
        """
        Get summary statistics by query type

        Args:
            df: DataFrame with query_type column
            metric_cols: Metric columns to aggregate

        Returns:
            Dictionary with summary by query type
        """
        if 'query_type' not in df.columns:
            raise ValueError("DataFrame must have 'query_type' column. Run classify_dataframe first.")

        metric_cols = metric_cols or ['clicks', 'impressions', 'ctr', 'position']
        metric_cols = [c for c in metric_cols if c in df.columns]

        summary = {
            'branded': {},
            'generic': {},
            'totals': {}
        }

        for query_type in ['branded', 'generic']:
            type_df = df[df['query_type'] == query_type]
            summary[query_type] = {
                'query_count': len(type_df[type_df.columns[0]].unique()) if not type_df.empty else 0,
                'row_count': len(type_df)
            }
            for col in metric_cols:
                if col == 'position':
                    summary[query_type][col] = type_df[col].mean() if not type_df.empty else 0
                elif col == 'ctr':
                    total_clicks = type_df['clicks'].sum() if 'clicks' in type_df.columns else 0
                    total_impressions = type_df['impressions'].sum() if 'impressions' in type_df.columns else 0
                    summary[query_type][col] = total_clicks / total_impressions if total_impressions > 0 else 0
                else:
                    summary[query_type][col] = type_df[col].sum() if not type_df.empty else 0

        # Totals
        summary['totals'] = {
            'total_queries': len(df),
            'branded_percentage': (df['query_type'] == 'branded').mean() * 100 if len(df) > 0 else 0,
            'generic_percentage': (df['query_type'] == 'generic').mean() * 100 if len(df) > 0 else 0
        }

        return summary

    def aggregate_by_type_and_date(
        self,
        df: pd.DataFrame,
        date_col: str = 'date'
    ) -> pd.DataFrame:
        """
        Aggregate metrics by query type and date for time series analysis

        Args:
            df: DataFrame with query_type and date columns
            date_col: Name of date column

        Returns:
            Aggregated DataFrame
        """
        if 'query_type' not in df.columns:
            raise ValueError("DataFrame must have 'query_type' column")

        if date_col not in df.columns:
            raise ValueError(f"Column '{date_col}' not found")

        # Define aggregation
        agg_dict = {}
        if 'clicks' in df.columns:
            agg_dict['clicks'] = 'sum'
        if 'impressions' in df.columns:
            agg_dict['impressions'] = 'sum'
        if 'position' in df.columns:
            agg_dict['position'] = 'mean'

        if not agg_dict:
            return df

        # Group and aggregate
        result = df.groupby([date_col, 'query_type']).agg(agg_dict).reset_index()

        # Calculate CTR
        if 'clicks' in result.columns and 'impressions' in result.columns:
            result['ctr'] = result['clicks'] / result['impressions'].replace(0, 1)

        return result

    @staticmethod
    def suggest_brand_terms(
        df: pd.DataFrame,
        query_col: str = 'query',
        top_n: int = 20
    ) -> List[str]:
        """
        Suggest potential brand terms based on query patterns

        This looks for terms that appear frequently in high-CTR, high-position queries,
        which are often branded queries.

        Args:
            df: DataFrame with queries and metrics
            query_col: Name of query column
            top_n: Number of suggestions to return

        Returns:
            List of potential brand terms
        """
        if query_col not in df.columns:
            return []

        suggestions = []

        # Look for high-performing single/double word queries
        # (branded queries often have high CTR and good position)
        if 'ctr' in df.columns and 'position' in df.columns:
            high_perf = df[
                (df['ctr'] > df['ctr'].quantile(0.75)) &
                (df['position'] < df['position'].quantile(0.25))
            ]

            if not high_perf.empty:
                # Get unique words from high-performing queries
                word_freq = defaultdict(int)
                for query in high_perf[query_col].unique():
                    words = str(query).lower().split()
                    for word in words:
                        if len(word) > 2:  # Skip very short words
                            word_freq[word] += 1

                # Sort by frequency
                suggestions = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
                suggestions = [word for word, _ in suggestions[:top_n]]

        return suggestions

    def analyze_brand_trend(
        self,
        df: pd.DataFrame,
        date_col: str = 'date'
    ) -> Dict:
        """
        Analyze trends in branded vs generic traffic over time

        Args:
            df: DataFrame with classified queries
            date_col: Date column name

        Returns:
            Dictionary with trend analysis
        """
        if 'query_type' not in df.columns or date_col not in df.columns:
            return {}

        agg_df = self.aggregate_by_type_and_date(df, date_col)

        # Pivot for comparison
        if 'clicks' in agg_df.columns:
            pivot = agg_df.pivot(index=date_col, columns='query_type', values='clicks').fillna(0)

            if 'branded' in pivot.columns and 'generic' in pivot.columns:
                # Calculate trends
                branded_trend = pivot['branded'].pct_change().mean()
                generic_trend = pivot['generic'].pct_change().mean()

                # Calculate share over time
                total = pivot['branded'] + pivot['generic']
                branded_share = (pivot['branded'] / total.replace(0, 1)).mean()

                return {
                    'branded_avg_daily_change': branded_trend,
                    'generic_avg_daily_change': generic_trend,
                    'branded_traffic_share': branded_share,
                    'generic_traffic_share': 1 - branded_share,
                    'trend_divergence': branded_trend - generic_trend
                }

        return {}
