"""
Interactive Visualization Module with Plotly
"""
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Any
from config.settings import COLORS


class ChartBuilder:
    """
    Creates interactive visualizations for GSC/GA4 data analysis

    Chart types:
    - Time series with anomaly highlighting
    - YoY comparison charts
    - Segment breakdown charts
    - Brand vs generic trends
    - Heatmaps for patterns
    - Distribution plots
    """

    def __init__(self):
        self.colors = COLORS

    def create_anomaly_timeline(
        self,
        df: pd.DataFrame,
        date_col: str = 'date',
        metric_col: str = 'clicks',
        anomaly_col: str = 'anomaly_consensus',
        anomaly_type_col: str = 'anomaly_type',
        title: str = 'Anomaly Detection Timeline'
    ) -> go.Figure:
        """
        Create time series chart with anomalies highlighted

        Args:
            df: DataFrame with date, metric, and anomaly columns
            date_col: Date column name
            metric_col: Metric column to plot
            anomaly_col: Boolean column for anomaly flag
            anomaly_type_col: Column with anomaly classification
            title: Chart title

        Returns:
            Plotly figure
        """
        fig = go.Figure()

        # Main time series line
        fig.add_trace(go.Scatter(
            x=df[date_col],
            y=df[metric_col],
            mode='lines',
            name=metric_col.capitalize(),
            line=dict(color=self.colors['primary'], width=2),
            hovertemplate=f'{metric_col}: %{{y:,.0f}}<br>Date: %{{x}}<extra></extra>'
        ))

        # Positive anomalies
        if anomaly_type_col in df.columns:
            positive = df[df[anomaly_type_col] == 'Positive Anomaly']
            if not positive.empty:
                fig.add_trace(go.Scatter(
                    x=positive[date_col],
                    y=positive[metric_col],
                    mode='markers',
                    name='Positive Anomaly',
                    marker=dict(
                        color=self.colors['positive_anomaly'],
                        size=12,
                        symbol='triangle-up',
                        line=dict(width=2, color='white')
                    ),
                    hovertemplate=f'Positive Anomaly<br>{metric_col}: %{{y:,.0f}}<br>Date: %{{x}}<extra></extra>'
                ))

            # Negative anomalies
            negative = df[df[anomaly_type_col] == 'Negative Anomaly']
            if not negative.empty:
                fig.add_trace(go.Scatter(
                    x=negative[date_col],
                    y=negative[metric_col],
                    mode='markers',
                    name='Negative Anomaly',
                    marker=dict(
                        color=self.colors['negative_anomaly'],
                        size=12,
                        symbol='triangle-down',
                        line=dict(width=2, color='white')
                    ),
                    hovertemplate=f'Negative Anomaly<br>{metric_col}: %{{y:,.0f}}<br>Date: %{{x}}<extra></extra>'
                ))

        fig.update_layout(
            title=title,
            xaxis_title='Date',
            yaxis_title=metric_col.capitalize(),
            hovermode='x unified',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
            template='plotly_white'
        )

        return fig

    def create_yoy_comparison(
        self,
        df: pd.DataFrame,
        metric_col: str = 'clicks',
        title: str = 'Year-over-Year Comparison'
    ) -> go.Figure:
        """
        Create YoY comparison chart

        Args:
            df: Aligned YoY DataFrame from SeasonalityAnalyzer
            metric_col: Base metric to compare
            title: Chart title

        Returns:
            Plotly figure
        """
        current_col = f'{metric_col}_current'
        previous_col = f'{metric_col}_previous'

        if current_col not in df.columns or previous_col not in df.columns:
            return go.Figure()

        fig = go.Figure()

        # Current period
        fig.add_trace(go.Scatter(
            x=df['day_of_year'],
            y=df[current_col],
            mode='lines',
            name='Current Year',
            line=dict(color=self.colors['primary'], width=2),
            fill='tozeroy',
            fillcolor='rgba(31, 119, 180, 0.1)'
        ))

        # Previous period
        fig.add_trace(go.Scatter(
            x=df['day_of_year'],
            y=df[previous_col],
            mode='lines',
            name='Previous Year',
            line=dict(color=self.colors['neutral'], width=2, dash='dot')
        ))

        fig.update_layout(
            title=title,
            xaxis_title='Day of Year',
            yaxis_title=metric_col.capitalize(),
            hovermode='x unified',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
            template='plotly_white'
        )

        return fig

    def create_brand_generic_chart(
        self,
        df: pd.DataFrame,
        date_col: str = 'date',
        metric_col: str = 'clicks',
        title: str = 'Branded vs Generic Traffic'
    ) -> go.Figure:
        """
        Create stacked area chart for branded vs generic traffic

        Args:
            df: DataFrame with query_type and metrics
            date_col: Date column name
            metric_col: Metric to plot
            title: Chart title

        Returns:
            Plotly figure
        """
        if 'query_type' not in df.columns:
            return go.Figure()

        # Aggregate by date and query type
        agg = df.groupby([date_col, 'query_type'])[metric_col].sum().reset_index()
        pivot = agg.pivot(index=date_col, columns='query_type', values=metric_col).fillna(0)

        fig = go.Figure()

        if 'branded' in pivot.columns:
            fig.add_trace(go.Scatter(
                x=pivot.index,
                y=pivot['branded'],
                mode='lines',
                name='Branded',
                stackgroup='one',
                fillcolor=self.colors['branded'],
                line=dict(color=self.colors['branded'], width=0.5)
            ))

        if 'generic' in pivot.columns:
            fig.add_trace(go.Scatter(
                x=pivot.index,
                y=pivot['generic'],
                mode='lines',
                name='Generic',
                stackgroup='one',
                fillcolor=self.colors['generic'],
                line=dict(color=self.colors['generic'], width=0.5)
            ))

        fig.update_layout(
            title=title,
            xaxis_title='Date',
            yaxis_title=metric_col.capitalize(),
            hovermode='x unified',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
            template='plotly_white'
        )

        return fig

    def create_segment_breakdown(
        self,
        df: pd.DataFrame,
        segment_col: str,
        metric_col: str = 'clicks',
        top_n: int = 10,
        title: str = None
    ) -> go.Figure:
        """
        Create horizontal bar chart for segment breakdown

        Args:
            df: DataFrame with segment and metrics
            segment_col: Segment column name
            metric_col: Metric to plot
            top_n: Number of top segments to show
            title: Chart title

        Returns:
            Plotly figure
        """
        # Aggregate by segment
        agg = df.groupby(segment_col)[metric_col].sum().reset_index()
        agg = agg.nlargest(top_n, metric_col)
        agg = agg.sort_values(metric_col, ascending=True)

        fig = go.Figure(go.Bar(
            x=agg[metric_col],
            y=agg[segment_col],
            orientation='h',
            marker_color=self.colors['primary'],
            text=agg[metric_col].apply(lambda x: f'{x:,.0f}'),
            textposition='outside'
        ))

        fig.update_layout(
            title=title or f'Top {top_n} {segment_col.replace("_", " ").title()} by {metric_col.capitalize()}',
            xaxis_title=metric_col.capitalize(),
            yaxis_title=segment_col.replace('_', ' ').title(),
            template='plotly_white',
            height=max(400, top_n * 40)
        )

        return fig

    def create_multi_metric_dashboard(
        self,
        df: pd.DataFrame,
        date_col: str = 'date',
        metrics: List[str] = None,
        title: str = 'Performance Dashboard'
    ) -> go.Figure:
        """
        Create multi-panel dashboard with key metrics

        Args:
            df: DataFrame with date and metrics
            date_col: Date column name
            metrics: List of metrics to plot
            title: Dashboard title

        Returns:
            Plotly figure with subplots
        """
        metrics = metrics or ['clicks', 'impressions', 'ctr', 'position']
        metrics = [m for m in metrics if m in df.columns]

        if not metrics:
            return go.Figure()

        rows = (len(metrics) + 1) // 2
        fig = make_subplots(
            rows=rows,
            cols=2,
            subplot_titles=[m.replace('_', ' ').title() for m in metrics],
            vertical_spacing=0.12,
            horizontal_spacing=0.1
        )

        colors = [self.colors['primary'], self.colors['generic'],
                  self.colors['positive_anomaly'], self.colors['warning']]

        for i, metric in enumerate(metrics):
            row = i // 2 + 1
            col = i % 2 + 1

            fig.add_trace(
                go.Scatter(
                    x=df[date_col],
                    y=df[metric],
                    mode='lines',
                    name=metric.replace('_', ' ').title(),
                    line=dict(color=colors[i % len(colors)], width=2),
                    showlegend=False
                ),
                row=row,
                col=col
            )

        fig.update_layout(
            title=title,
            height=300 * rows,
            template='plotly_white',
            showlegend=False
        )

        return fig

    def create_heatmap_weekly_pattern(
        self,
        df: pd.DataFrame,
        date_col: str = 'date',
        metric_col: str = 'clicks',
        title: str = 'Weekly Performance Pattern'
    ) -> go.Figure:
        """
        Create heatmap showing weekly patterns

        Args:
            df: DataFrame with date and metric
            date_col: Date column name
            metric_col: Metric to analyze
            title: Chart title

        Returns:
            Plotly figure
        """
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])
        df['week'] = df[date_col].dt.isocalendar().week
        df['day_of_week'] = df[date_col].dt.dayofweek

        # Aggregate
        pivot = df.pivot_table(
            values=metric_col,
            index='day_of_week',
            columns='week',
            aggfunc='sum'
        ).fillna(0)

        day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

        fig = go.Figure(data=go.Heatmap(
            z=pivot.values,
            x=[f'W{w}' for w in pivot.columns],
            y=[day_names[i] for i in pivot.index],
            colorscale='Blues',
            hovertemplate='Week: %{x}<br>Day: %{y}<br>Value: %{z:,.0f}<extra></extra>'
        ))

        fig.update_layout(
            title=title,
            xaxis_title='Week',
            yaxis_title='Day of Week',
            template='plotly_white'
        )

        return fig

    def create_channel_comparison(
        self,
        gsc_df: pd.DataFrame,
        ga4_df: pd.DataFrame,
        date_col: str = 'date',
        title: str = 'GSC vs GA4 Comparison'
    ) -> go.Figure:
        """
        Create comparison chart between GSC and GA4 data

        Args:
            gsc_df: GSC DataFrame with date and clicks
            ga4_df: GA4 DataFrame with date and sessions
            date_col: Date column name
            title: Chart title

        Returns:
            Plotly figure
        """
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        # GSC Clicks
        if 'clicks' in gsc_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=gsc_df[date_col],
                    y=gsc_df['clicks'],
                    mode='lines',
                    name='GSC Clicks',
                    line=dict(color=self.colors['primary'], width=2)
                ),
                secondary_y=False
            )

        # GA4 Sessions
        if 'sessions' in ga4_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=ga4_df[date_col],
                    y=ga4_df['sessions'],
                    mode='lines',
                    name='GA4 Organic Sessions',
                    line=dict(color=self.colors['positive_anomaly'], width=2)
                ),
                secondary_y=True
            )

        fig.update_layout(
            title=title,
            hovermode='x unified',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
            template='plotly_white'
        )

        fig.update_yaxes(title_text="GSC Clicks", secondary_y=False)
        fig.update_yaxes(title_text="GA4 Sessions", secondary_y=True)

        return fig

    def create_anomaly_score_distribution(
        self,
        df: pd.DataFrame,
        score_col: str = 'anomaly_score_lof',
        title: str = 'Anomaly Score Distribution'
    ) -> go.Figure:
        """
        Create histogram of anomaly scores

        Args:
            df: DataFrame with anomaly scores
            score_col: Anomaly score column
            title: Chart title

        Returns:
            Plotly figure
        """
        if score_col not in df.columns:
            return go.Figure()

        fig = go.Figure(data=go.Histogram(
            x=df[score_col],
            nbinsx=50,
            marker_color=self.colors['primary'],
            opacity=0.7
        ))

        # Add threshold line
        threshold = df[score_col].quantile(0.95)
        fig.add_vline(
            x=threshold,
            line_dash="dash",
            line_color=self.colors['negative_anomaly'],
            annotation_text=f"95th percentile: {threshold:.2f}"
        )

        fig.update_layout(
            title=title,
            xaxis_title='Anomaly Score',
            yaxis_title='Count',
            template='plotly_white'
        )

        return fig

    def create_segment_comparison_over_time(
        self,
        df: pd.DataFrame,
        date_col: str = 'date',
        segment_col: str = 'query_type',
        metric_col: str = 'clicks',
        title: str = None
    ) -> go.Figure:
        """
        Create line chart comparing segments over time

        Args:
            df: DataFrame with date, segment, and metric
            date_col: Date column name
            segment_col: Segment column name
            metric_col: Metric to plot
            title: Chart title

        Returns:
            Plotly figure
        """
        # Aggregate by date and segment
        agg = df.groupby([date_col, segment_col])[metric_col].sum().reset_index()

        fig = px.line(
            agg,
            x=date_col,
            y=metric_col,
            color=segment_col,
            title=title or f'{metric_col.capitalize()} by {segment_col.replace("_", " ").title()}',
            template='plotly_white'
        )

        fig.update_layout(
            hovermode='x unified',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
        )

        return fig

    def create_metrics_summary_cards(
        self,
        current_metrics: Dict[str, float],
        previous_metrics: Dict[str, float] = None
    ) -> Dict[str, Dict]:
        """
        Create data for metric summary cards

        Args:
            current_metrics: Current period metrics
            previous_metrics: Previous period metrics for comparison

        Returns:
            Dictionary with card data
        """
        cards = {}

        for metric, value in current_metrics.items():
            card = {
                'value': value,
                'formatted': f'{value:,.0f}' if value >= 1 else f'{value:.2%}'
            }

            if previous_metrics and metric in previous_metrics:
                prev_value = previous_metrics[metric]
                change = value - prev_value
                pct_change = (change / prev_value * 100) if prev_value != 0 else 0

                card['change'] = change
                card['pct_change'] = pct_change
                card['trend'] = 'up' if change > 0 else 'down' if change < 0 else 'flat'

            cards[metric] = card

        return cards
