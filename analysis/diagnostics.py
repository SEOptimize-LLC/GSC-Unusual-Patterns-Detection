"""
Advanced Diagnostic Analysis Module

Implements the "Gap Analysis" checks:
1. Dark Search Calculator - Quantify hidden/anonymized query traffic
2. CTR vs Position Divergence - Detect "Illusion of Stability"
3. URL Volatility Detector - Find internal cannibalization
4. Engagement Leading Indicator - Correlate GA4 engagement with rankings
5. Search Appearance Analysis - Track rich snippet gains/losses
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from scipy import stats


class DiagnosticAnalyzer:
    """
    Advanced diagnostic checks for SEO anomaly detection

    Based on the "Unified Master Diagnostic Framework" covering:
    - Data Integrity & Measurement Hygiene
    - SERP Forensics
    - Technical Health & Internal Conflict
    - User Signals (Leading Indicators)
    """

    def __init__(self):
        pass

    # =========================================================================
    # PHASE 1: Data Integrity - "Dark Search" Calculator
    # =========================================================================

    def calculate_dark_search(
        self,
        total_clicks: int,
        query_level_clicks: int
    ) -> Dict[str, Any]:
        """
        Calculate the percentage of traffic hidden in anonymized queries.

        "Dark Search" = clicks that appear in totals but NOT in query-level data.
        This represents (not provided) style traffic Google hides for privacy.

        Args:
            total_clicks: Total clicks from GSC overview/daily data
            query_level_clicks: Sum of clicks from query-level report

        Returns:
            Dictionary with dark search metrics
        """
        if total_clicks == 0:
            return {
                'dark_search_clicks': 0,
                'dark_search_percentage': 0,
                'visible_percentage': 100,
                'status': 'no_data',
                'interpretation': 'No click data available'
            }

        dark_clicks = total_clicks - query_level_clicks
        dark_percentage = (dark_clicks / total_clicks) * 100
        visible_percentage = 100 - dark_percentage

        # Interpret the results
        if dark_percentage < 10:
            status = 'healthy'
            interpretation = 'Excellent query visibility. Most traffic sources are identifiable.'
        elif dark_percentage < 25:
            status = 'normal'
            interpretation = 'Normal levels of anonymized queries. Some long-tail traffic is hidden.'
        elif dark_percentage < 40:
            status = 'warning'
            interpretation = 'Elevated dark search. You may be losing visibility on privacy-sensitive or long-tail queries.'
        else:
            status = 'critical'
            interpretation = 'High dark search percentage. Significant traffic sources are hidden. Consider if this aligns with your audience (privacy-conscious users, sensitive topics).'

        return {
            'total_clicks': total_clicks,
            'query_level_clicks': query_level_clicks,
            'dark_search_clicks': dark_clicks,
            'dark_search_percentage': round(dark_percentage, 2),
            'visible_percentage': round(visible_percentage, 2),
            'status': status,
            'interpretation': interpretation
        }

    def analyze_dark_search_trend(
        self,
        daily_data: pd.DataFrame,
        query_data: pd.DataFrame,
        date_col: str = 'date'
    ) -> pd.DataFrame:
        """
        Analyze dark search percentage over time.

        A growing dark search gap may indicate:
        - Increasing privacy-conscious audience
        - Loss of visibility on long-tail terms
        - Potential tracking issues

        Args:
            daily_data: Daily aggregated GSC data
            query_data: Query-level GSC data with dates
            date_col: Date column name

        Returns:
            DataFrame with daily dark search metrics
        """
        if daily_data.empty or query_data.empty:
            return pd.DataFrame()

        # Aggregate query data by date
        query_daily = query_data.groupby(date_col)['clicks'].sum().reset_index()
        query_daily.columns = [date_col, 'query_clicks']

        # Merge with daily totals
        merged = daily_data[[date_col, 'clicks']].merge(
            query_daily,
            on=date_col,
            how='left'
        ).fillna(0)

        merged['dark_clicks'] = merged['clicks'] - merged['query_clicks']
        merged['dark_percentage'] = (merged['dark_clicks'] / merged['clicks'].replace(0, 1)) * 100

        return merged

    # =========================================================================
    # PHASE 3: SERP Forensics - CTR vs Position Divergence
    # =========================================================================

    def detect_ctr_position_divergence(
        self,
        df: pd.DataFrame,
        date_col: str = 'date',
        window: int = 7,
        threshold: float = 20.0
    ) -> Dict[str, Any]:
        """
        Detect "Illusion of Stability" - when position is stable but CTR drops.

        This indicates SERP layout changes (more ads, featured snippets, etc.)
        that push your result down visually even if the ranking number stays same.

        Args:
            df: DataFrame with date, position, ctr, clicks, impressions
            date_col: Date column name
            window: Rolling window for trend calculation
            threshold: Percentage change threshold for flagging

        Returns:
            Dictionary with divergence analysis
        """
        if df.empty or 'position' not in df.columns or 'ctr' not in df.columns:
            return {'status': 'insufficient_data', 'alerts': []}

        df = df.sort_values(date_col).copy()

        # Calculate rolling averages
        df['position_ma'] = df['position'].rolling(window, min_periods=1).mean()
        df['ctr_ma'] = df['ctr'].rolling(window, min_periods=1).mean()

        # Calculate period-over-period changes
        half = len(df) // 2
        if half < 3:
            return {'status': 'insufficient_data', 'alerts': []}

        first_half = df.iloc[:half]
        second_half = df.iloc[half:]

        # Position change (lower is better, so negative change = improvement)
        position_first = first_half['position'].mean()
        position_second = second_half['position'].mean()
        position_change = ((position_second - position_first) / position_first) * 100

        # CTR change
        ctr_first = first_half['ctr'].mean()
        ctr_second = second_half['ctr'].mean()
        ctr_change = ((ctr_second - ctr_first) / (ctr_first + 0.0001)) * 100

        alerts = []

        # Detect divergence patterns
        # Pattern 1: Position stable/improving but CTR dropping
        if position_change <= 5 and ctr_change < -threshold:
            alerts.append({
                'type': 'serp_layout_change',
                'severity': 'high',
                'message': f'Position stable ({position_change:+.1f}%) but CTR dropped {ctr_change:.1f}%',
                'interpretation': 'SERP layout likely changed. More ads, featured snippets, or other elements may be pushing your result down visually.',
                'action': 'Check Search Appearance filters in GSC. Review SERP for new features above your position.'
            })

        # Pattern 2: Position dropping but CTR stable (you still get clicks - brand strength)
        if position_change > threshold and abs(ctr_change) < 10:
            alerts.append({
                'type': 'brand_resilience',
                'severity': 'medium',
                'message': f'Position dropped {position_change:.1f}% but CTR held steady ({ctr_change:+.1f}%)',
                'interpretation': 'Users are scrolling to find you specifically. This suggests brand loyalty but ranking decline.',
                'action': 'Investigate why rankings dropped while monitoring if CTR eventually follows.'
            })

        # Pattern 3: Both dropping together (expected correlation)
        if position_change > threshold and ctr_change < -threshold:
            alerts.append({
                'type': 'ranking_decline',
                'severity': 'high',
                'message': f'Both position ({position_change:+.1f}%) and CTR ({ctr_change:.1f}%) declining',
                'interpretation': 'Standard ranking decline pattern. Position loss is causing CTR loss.',
                'action': 'Focus on recovering rankings. Check for algorithm updates, technical issues, or content decay.'
            })

        # Pattern 4: Position improving but CTR not following (opportunity)
        if position_change < -threshold and ctr_change < 10:
            alerts.append({
                'type': 'ctr_optimization_opportunity',
                'severity': 'low',
                'message': f'Position improved {abs(position_change):.1f}% but CTR only changed {ctr_change:+.1f}%',
                'interpretation': 'You\'re ranking better but not capturing proportionally more clicks.',
                'action': 'Optimize title tags and meta descriptions to improve CTR at new positions.'
            })

        return {
            'status': 'analyzed',
            'position_change_pct': round(position_change, 2),
            'ctr_change_pct': round(ctr_change, 2),
            'position_first_period': round(position_first, 2),
            'position_second_period': round(position_second, 2),
            'ctr_first_period': round(ctr_first * 100, 2),
            'ctr_second_period': round(ctr_second * 100, 2),
            'alerts': alerts,
            'data': df[[date_col, 'position', 'ctr', 'position_ma', 'ctr_ma']].to_dict('records')
        }

    # =========================================================================
    # PHASE 4: Technical Health - URL Volatility / Cannibalization
    # =========================================================================

    def detect_url_cannibalization(
        self,
        query_page_data: pd.DataFrame,
        date_col: str = 'date',
        query_col: str = 'query',
        page_col: str = 'page',
        min_impressions: int = 10
    ) -> Dict[str, Any]:
        """
        Detect internal cannibalization by finding queries with multiple ranking URLs.

        Cannibalization occurs when multiple pages from your site compete for
        the same query, causing position volatility and wasted crawl budget.

        Args:
            query_page_data: DataFrame with date, query, page, and metrics
            date_col: Date column name
            query_col: Query column name
            page_col: Page/URL column name
            min_impressions: Minimum impressions to consider a query

        Returns:
            Dictionary with cannibalization analysis
        """
        if query_page_data.empty:
            return {'status': 'no_data', 'cannibalized_queries': []}

        required_cols = [query_col, page_col, 'clicks', 'impressions']
        if not all(col in query_page_data.columns for col in required_cols):
            return {'status': 'missing_columns', 'cannibalized_queries': []}

        # Filter for queries with minimum impressions
        query_totals = query_page_data.groupby(query_col)['impressions'].sum()
        significant_queries = query_totals[query_totals >= min_impressions].index
        df = query_page_data[query_page_data[query_col].isin(significant_queries)]

        # Find queries with multiple URLs
        query_url_counts = df.groupby(query_col)[page_col].nunique()
        multi_url_queries = query_url_counts[query_url_counts > 1].index

        cannibalized = []

        for query in multi_url_queries:
            query_df = df[df[query_col] == query]
            urls = query_df.groupby(page_col).agg({
                'clicks': 'sum',
                'impressions': 'sum',
                'position': 'mean'
            }).reset_index()

            urls = urls.sort_values('clicks', ascending=False)
            total_clicks = urls['clicks'].sum()
            total_impressions = urls['impressions'].sum()

            # Calculate click concentration (how much goes to top URL)
            top_url_share = (urls.iloc[0]['clicks'] / total_clicks * 100) if total_clicks > 0 else 0

            # Position variance indicates ranking instability
            position_variance = urls['position'].var()

            cannibalized.append({
                'query': query,
                'url_count': len(urls),
                'total_clicks': int(total_clicks),
                'total_impressions': int(total_impressions),
                'top_url': urls.iloc[0][page_col],
                'top_url_clicks': int(urls.iloc[0]['clicks']),
                'top_url_share_pct': round(top_url_share, 1),
                'position_variance': round(position_variance, 2) if not pd.isna(position_variance) else 0,
                'competing_urls': urls[page_col].tolist(),
                'severity': 'high' if top_url_share < 50 else 'medium' if top_url_share < 75 else 'low'
            })

        # Sort by severity and click volume
        cannibalized.sort(key=lambda x: (-{'high': 3, 'medium': 2, 'low': 1}[x['severity']], -x['total_clicks']))

        return {
            'status': 'analyzed',
            'total_queries_analyzed': len(significant_queries),
            'cannibalized_query_count': len(cannibalized),
            'cannibalization_rate': round(len(cannibalized) / len(significant_queries) * 100, 2) if significant_queries.any() else 0,
            'high_severity_count': sum(1 for c in cannibalized if c['severity'] == 'high'),
            'cannibalized_queries': cannibalized[:50]  # Top 50
        }

    def detect_url_volatility(
        self,
        query_page_date_data: pd.DataFrame,
        date_col: str = 'date',
        query_col: str = 'query',
        page_col: str = 'page',
        min_days: int = 7
    ) -> Dict[str, Any]:
        """
        Detect URL volatility - when different URLs rank for the same query on different days.

        This is a stronger signal of cannibalization than just multiple URLs,
        as it shows Google is actively switching which page it prefers.

        Args:
            query_page_date_data: DataFrame with date, query, page columns
            date_col: Date column name
            query_col: Query column name
            page_col: Page column name
            min_days: Minimum days of data required

        Returns:
            Dictionary with volatility analysis
        """
        if query_page_date_data.empty:
            return {'status': 'no_data', 'volatile_queries': []}

        df = query_page_date_data.copy()

        # Get queries that appear on multiple days
        query_day_counts = df.groupby(query_col)[date_col].nunique()
        multi_day_queries = query_day_counts[query_day_counts >= min_days].index

        volatile_queries = []

        for query in multi_day_queries:
            query_df = df[df[query_col] == query].sort_values(date_col)

            # Get the ranking URL for each day (top position)
            daily_top_url = query_df.loc[query_df.groupby(date_col)['position'].idxmin()]

            # Count URL switches
            urls_over_time = daily_top_url[page_col].tolist()
            switches = sum(1 for i in range(1, len(urls_over_time)) if urls_over_time[i] != urls_over_time[i-1])

            unique_urls = daily_top_url[page_col].nunique()

            if unique_urls > 1:  # Only include if multiple URLs ranked
                volatility_score = switches / (len(urls_over_time) - 1) if len(urls_over_time) > 1 else 0

                volatile_queries.append({
                    'query': query,
                    'unique_urls': unique_urls,
                    'url_switches': switches,
                    'days_tracked': len(urls_over_time),
                    'volatility_score': round(volatility_score, 2),
                    'urls': list(daily_top_url[page_col].unique()),
                    'severity': 'high' if volatility_score > 0.5 else 'medium' if volatility_score > 0.25 else 'low'
                })

        volatile_queries.sort(key=lambda x: -x['volatility_score'])

        return {
            'status': 'analyzed',
            'total_queries_tracked': len(multi_day_queries),
            'volatile_query_count': len(volatile_queries),
            'high_volatility_count': sum(1 for v in volatile_queries if v['severity'] == 'high'),
            'volatile_queries': volatile_queries[:30]
        }

    # =========================================================================
    # PHASE 5: User Signals - Engagement Leading Indicator
    # =========================================================================

    def analyze_engagement_leading_indicator(
        self,
        gsc_data: pd.DataFrame,
        ga4_data: pd.DataFrame,
        date_col: str = 'date',
        lag_days: int = 14
    ) -> Dict[str, Any]:
        """
        Analyze if engagement drops predict ranking drops.

        Theory: User engagement (bounce rate, time on page) drops often
        precede ranking drops by 1-4 weeks as Google detects user dissatisfaction.

        Args:
            gsc_data: GSC daily data with position
            ga4_data: GA4 data with engagement metrics
            date_col: Date column name
            lag_days: Days to look ahead for ranking impact

        Returns:
            Dictionary with leading indicator analysis
        """
        if gsc_data.empty or ga4_data.empty:
            return {'status': 'insufficient_data', 'correlation': None}

        # Standardize date columns
        gsc = gsc_data.copy()
        ga4 = ga4_data.copy()

        gsc[date_col] = pd.to_datetime(gsc[date_col])
        ga4[date_col] = pd.to_datetime(ga4[date_col])

        # Merge datasets
        merged = gsc.merge(ga4, on=date_col, how='inner', suffixes=('_gsc', '_ga4'))

        if len(merged) < 14:
            return {'status': 'insufficient_data', 'correlation': None}

        results = {
            'status': 'analyzed',
            'data_points': len(merged),
            'correlations': {},
            'leading_indicators': []
        }

        # Check for engagement columns
        engagement_cols = ['bounceRate', 'averageSessionDuration', 'engagementRate',
                         'screenPageViews', 'sessions']
        available_engagement = [col for col in engagement_cols if col in merged.columns]

        if not available_engagement:
            return {'status': 'no_engagement_metrics', 'correlation': None}

        # Calculate lagged correlations
        for eng_col in available_engagement:
            if eng_col in merged.columns and 'position' in merged.columns:
                # Create lagged position (future position)
                merged[f'position_lag_{lag_days}'] = merged['position'].shift(-lag_days)

                # Calculate correlation between current engagement and future position
                valid_data = merged[[eng_col, f'position_lag_{lag_days}']].dropna()

                if len(valid_data) > 10:
                    correlation, p_value = stats.pearsonr(
                        valid_data[eng_col],
                        valid_data[f'position_lag_{lag_days}']
                    )

                    results['correlations'][eng_col] = {
                        'correlation': round(correlation, 3),
                        'p_value': round(p_value, 4),
                        'significant': p_value < 0.05,
                        'interpretation': self._interpret_engagement_correlation(eng_col, correlation)
                    }

                    # Flag as leading indicator if significant
                    if p_value < 0.05 and abs(correlation) > 0.3:
                        results['leading_indicators'].append({
                            'metric': eng_col,
                            'correlation': correlation,
                            'direction': 'Drops in engagement predict ranking drops' if correlation < 0 else 'Higher engagement predicts better rankings',
                            'strength': 'strong' if abs(correlation) > 0.5 else 'moderate'
                        })

        return results

    def _interpret_engagement_correlation(self, metric: str, correlation: float) -> str:
        """Interpret engagement-ranking correlation"""
        direction = "negative" if correlation < 0 else "positive"
        strength = "strong" if abs(correlation) > 0.5 else "moderate" if abs(correlation) > 0.3 else "weak"

        interpretations = {
            'bounceRate': f'{strength.capitalize()} {direction} correlation. {"Higher bounce rates may predict ranking drops." if correlation > 0 else "Lower bounce rates associated with better rankings."}',
            'averageSessionDuration': f'{strength.capitalize()} {direction} correlation. {"Longer sessions may predict ranking improvements." if correlation < 0 else "Session duration changes may impact rankings."}',
            'engagementRate': f'{strength.capitalize()} {direction} correlation. {"Higher engagement predicts better rankings." if correlation < 0 else "Engagement changes impact rankings."}',
        }

        return interpretations.get(metric, f'{strength.capitalize()} {direction} correlation with rankings.')

    # =========================================================================
    # PHASE 3: SERP Forensics - Search Appearance Analysis
    # =========================================================================

    def analyze_search_appearance(
        self,
        current_data: pd.DataFrame,
        previous_data: pd.DataFrame,
        appearance_col: str = 'searchAppearance'
    ) -> Dict[str, Any]:
        """
        Analyze changes in search appearance (rich snippets, featured snippets, etc.)

        Losing rich results can cause massive CTR drops even if position stays same.

        Args:
            current_data: Current period data with search appearance
            previous_data: Previous period data for comparison
            appearance_col: Search appearance column name

        Returns:
            Dictionary with appearance analysis
        """
        if current_data.empty:
            return {'status': 'no_data', 'appearances': {}}

        # Aggregate by appearance type
        current_appearances = {}
        previous_appearances = {}

        if appearance_col in current_data.columns:
            current_agg = current_data.groupby(appearance_col).agg({
                'clicks': 'sum',
                'impressions': 'sum'
            }).to_dict('index')
            current_appearances = current_agg

        if not previous_data.empty and appearance_col in previous_data.columns:
            previous_agg = previous_data.groupby(appearance_col).agg({
                'clicks': 'sum',
                'impressions': 'sum'
            }).to_dict('index')
            previous_appearances = previous_agg

        # Calculate changes
        all_appearances = set(current_appearances.keys()) | set(previous_appearances.keys())

        changes = []
        for appearance in all_appearances:
            current = current_appearances.get(appearance, {'clicks': 0, 'impressions': 0})
            previous = previous_appearances.get(appearance, {'clicks': 0, 'impressions': 0})

            click_change = current['clicks'] - previous['clicks']
            impression_change = current['impressions'] - previous['impressions']

            click_pct_change = (click_change / previous['clicks'] * 100) if previous['clicks'] > 0 else (100 if current['clicks'] > 0 else 0)

            status = 'gained' if previous['clicks'] == 0 and current['clicks'] > 0 else \
                    'lost' if current['clicks'] == 0 and previous['clicks'] > 0 else \
                    'increased' if click_change > 0 else \
                    'decreased' if click_change < 0 else 'stable'

            changes.append({
                'appearance_type': appearance,
                'current_clicks': current['clicks'],
                'previous_clicks': previous['clicks'],
                'click_change': click_change,
                'click_change_pct': round(click_pct_change, 1),
                'current_impressions': current['impressions'],
                'previous_impressions': previous['impressions'],
                'status': status,
                'severity': 'high' if status in ['lost', 'gained'] or abs(click_pct_change) > 50 else 'medium' if abs(click_pct_change) > 20 else 'low'
            })

        # Sort by impact
        changes.sort(key=lambda x: -abs(x['click_change']))

        # Generate alerts for significant changes
        alerts = []
        for change in changes:
            if change['status'] == 'lost':
                alerts.append({
                    'type': 'appearance_lost',
                    'severity': 'high',
                    'message': f"Lost '{change['appearance_type']}' rich result ({change['previous_clicks']} clicks lost)",
                    'action': 'Check if structured data is still valid. Review if content still qualifies for this feature.'
                })
            elif change['status'] == 'gained':
                alerts.append({
                    'type': 'appearance_gained',
                    'severity': 'positive',
                    'message': f"Gained '{change['appearance_type']}' rich result ({change['current_clicks']} new clicks)",
                    'action': 'Monitor to ensure this feature is maintained. Consider expanding to other pages.'
                })
            elif change['click_change_pct'] < -30:
                alerts.append({
                    'type': 'appearance_declining',
                    'severity': 'medium',
                    'message': f"'{change['appearance_type']}' clicks dropped {abs(change['click_change_pct']):.0f}%",
                    'action': 'Investigate if competitors are taking this SERP feature or if eligibility criteria changed.'
                })

        return {
            'status': 'analyzed',
            'appearance_changes': changes,
            'alerts': alerts,
            'total_appearances_tracked': len(all_appearances)
        }

    # =========================================================================
    # Summary Report Generator
    # =========================================================================

    def generate_diagnostic_summary(
        self,
        dark_search: Dict,
        ctr_divergence: Dict,
        cannibalization: Dict,
        engagement: Dict,
        search_appearance: Dict
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive diagnostic summary with prioritized action items.

        Args:
            dark_search: Results from calculate_dark_search
            ctr_divergence: Results from detect_ctr_position_divergence
            cannibalization: Results from detect_url_cannibalization
            engagement: Results from analyze_engagement_leading_indicator
            search_appearance: Results from analyze_search_appearance

        Returns:
            Prioritized summary with action items
        """
        critical_issues = []
        warnings = []
        opportunities = []

        # Check Dark Search
        if dark_search.get('status') == 'critical':
            critical_issues.append({
                'area': 'Data Visibility',
                'issue': f"{dark_search['dark_search_percentage']}% of traffic is hidden (Dark Search)",
                'impact': 'You cannot optimize what you cannot see',
                'action': 'Review if this aligns with your audience. Consider privacy-focused content strategy.'
            })
        elif dark_search.get('status') == 'warning':
            warnings.append({
                'area': 'Data Visibility',
                'issue': f"{dark_search['dark_search_percentage']}% Dark Search",
                'action': 'Monitor trend. Growing dark search may indicate changing audience or tracking issues.'
            })

        # Check CTR Divergence
        for alert in ctr_divergence.get('alerts', []):
            if alert['severity'] == 'high':
                critical_issues.append({
                    'area': 'SERP Forensics',
                    'issue': alert['message'],
                    'impact': alert['interpretation'],
                    'action': alert['action']
                })
            elif alert['severity'] == 'medium':
                warnings.append({
                    'area': 'SERP Forensics',
                    'issue': alert['message'],
                    'action': alert['action']
                })
            elif alert['type'] == 'ctr_optimization_opportunity':
                opportunities.append({
                    'area': 'CTR Optimization',
                    'opportunity': alert['message'],
                    'action': alert['action']
                })

        # Check Cannibalization
        if cannibalization.get('high_severity_count', 0) > 0:
            critical_issues.append({
                'area': 'Technical Health',
                'issue': f"{cannibalization['high_severity_count']} queries with severe URL cannibalization",
                'impact': 'Multiple pages competing reduces overall ranking potential',
                'action': 'Consolidate content or implement clear canonicalization strategy for affected queries.'
            })
        elif cannibalization.get('cannibalized_query_count', 0) > 10:
            warnings.append({
                'area': 'Technical Health',
                'issue': f"{cannibalization['cannibalized_query_count']} queries show cannibalization signals",
                'action': 'Review top affected queries and consider content consolidation.'
            })

        # Check Engagement Leading Indicators
        for indicator in engagement.get('leading_indicators', []):
            if indicator['strength'] == 'strong':
                warnings.append({
                    'area': 'User Signals',
                    'issue': f"{indicator['metric']} shows strong correlation with future rankings",
                    'action': indicator['direction'] + '. Monitor this metric as early warning system.'
                })

        # Check Search Appearance
        for alert in search_appearance.get('alerts', []):
            if alert['severity'] == 'high':
                critical_issues.append({
                    'area': 'Rich Results',
                    'issue': alert['message'],
                    'impact': 'Lost rich results can cause 30-60% CTR drops',
                    'action': alert['action']
                })
            elif alert['severity'] == 'positive':
                opportunities.append({
                    'area': 'Rich Results',
                    'opportunity': alert['message'],
                    'action': alert['action']
                })

        return {
            'critical_issues': critical_issues,
            'warnings': warnings,
            'opportunities': opportunities,
            'health_score': self._calculate_health_score(critical_issues, warnings, opportunities)
        }

    def _calculate_health_score(
        self,
        critical: List,
        warnings: List,
        opportunities: List
    ) -> Dict[str, Any]:
        """Calculate overall diagnostic health score"""
        # Deduct points for issues, add points for opportunities
        score = 100
        score -= len(critical) * 20
        score -= len(warnings) * 5
        score += len(opportunities) * 5
        score = max(0, min(100, score))  # Clamp to 0-100

        if score >= 80:
            status = 'healthy'
            color = 'green'
        elif score >= 60:
            status = 'needs_attention'
            color = 'yellow'
        elif score >= 40:
            status = 'warning'
            color = 'orange'
        else:
            status = 'critical'
            color = 'red'

        return {
            'score': score,
            'status': status,
            'color': color,
            'critical_count': len(critical),
            'warning_count': len(warnings),
            'opportunity_count': len(opportunities)
        }
