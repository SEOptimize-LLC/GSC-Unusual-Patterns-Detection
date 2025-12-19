"""
GSC Unusual Patterns Detection - Main Streamlit Application

A comprehensive tool for detecting anomalies and patterns in Google Search Console
and Google Analytics 4 data with AI-powered insights.
"""
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.settings import (
    DATE_RANGE_OPTIONS, ANOMALY_DETECTION_METHODS,
    GSC_DIMENSIONS, COLORS
)
from auth.google_auth import GoogleAuthManager
from auth.credentials import CredentialManager
from connectors.gsc_connector import GSCConnector
from connectors.ga4_connector import GA4Connector
from analysis.anomaly_detection import AnomalyDetector
from analysis.brand_classifier import BrandClassifier
from analysis.seasonality import SeasonalityAnalyzer
from analysis.segmentation import DataSegmenter
from analysis.diagnostics import DiagnosticAnalyzer
from ai.openrouter import AIAnalyzer
from visualization.charts import ChartBuilder
from utils.helpers import DataProcessor, format_number, format_percentage, format_change

# Page configuration
st.set_page_config(
    page_title="GSC Anomaly Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 1rem;
        padding: 1rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin-bottom: 0.5rem;
    }
    .anomaly-positive {
        background-color: #d4edda;
        border-left-color: #28a745;
    }
    .anomaly-negative {
        background-color: #f8d7da;
        border-left-color: #dc3545;
    }
    .section-header {
        font-size: 1.3rem;
        font-weight: 600;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #e9ecef;
    }
    .insight-box {
        background-color: #e7f3ff;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
    }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """Initialize session state variables"""
    defaults = {
        'gsc_data': None,
        'ga4_data': None,
        'analysis_results': None,
        'brand_terms': [],
        'selected_site': None,
        'selected_ga4_property': None
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def render_sidebar(credentials):
    """Render sidebar with configuration options"""
    with st.sidebar:
        st.header("⚙️ Configuration")

        # Logout button
        if st.button("🚪 Logout", use_container_width=True):
            CredentialManager.clear_credentials()
            st.rerun()

        st.markdown("---")

        # Data source selection
        st.subheader("📊 Data Sources")

        # GSC Configuration - cache sites in session state to avoid repeated API calls
        if 'available_sites' not in st.session_state:
            gsc_connector = GSCConnector(credentials)
            st.session_state['available_sites'] = gsc_connector.get_verified_sites()

        sites = st.session_state['available_sites']

        if sites:
            # Get previously selected site or default to first
            previously_selected = st.session_state.get('selected_site_value', sites[0])
            if previously_selected not in sites:
                previously_selected = sites[0]

            # Find the index of the previously selected site
            default_idx = sites.index(previously_selected) if previously_selected in sites else 0

            selected_site = st.selectbox(
                "🔍 GSC Property",
                sites,
                index=default_idx
            )

            # Store the selected value (not using widget key to avoid conflicts)
            st.session_state['selected_site_value'] = selected_site
        else:
            st.warning("No GSC sites found")
            selected_site = None

        # GA4 Configuration
        st.markdown("---")
        ga4_connector = GA4Connector(credentials)
        ga4_properties = ga4_connector.get_properties()

        use_ga4 = st.checkbox("Include GA4 Data", value=False)

        if use_ga4:
            if ga4_properties:
                property_options = {
                    f"{p['display_name']} ({p['property_id']})": p['property_id']
                    for p in ga4_properties
                }
                selected_property_label = st.selectbox(
                    "📈 GA4 Property",
                    list(property_options.keys())
                )
                st.session_state['selected_ga4_property'] = property_options.get(selected_property_label)
            else:
                manual_property = st.text_input(
                    "GA4 Property ID",
                    placeholder="123456789",
                    help="Enter your GA4 property ID (numbers only)"
                )
                st.session_state['selected_ga4_property'] = manual_property if manual_property else None

        st.markdown("---")

        # Date range
        st.subheader("📅 Date Range")
        date_range = st.selectbox(
            "Select Period",
            list(DATE_RANGE_OPTIONS.keys())
        )

        if date_range == "Custom range":
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input(
                    "Start",
                    value=datetime.now() - timedelta(days=30)
                )
            with col2:
                end_date = st.date_input(
                    "End",
                    value=datetime.now() - timedelta(days=1)
                )
        else:
            days = DATE_RANGE_OPTIONS[date_range]
            end_date = datetime.now() - timedelta(days=1)
            start_date = end_date - timedelta(days=days)

        include_yoy = st.checkbox("Include Year-over-Year Comparison", value=True)

        st.markdown("---")

        # Brand terms
        st.subheader("🏷️ Brand Configuration")
        brand_input = st.text_area(
            "Brand Terms (one per line)",
            value="\n".join(st.session_state.get('brand_terms', [])),
            help="Enter your brand name and variations"
        )
        st.session_state['brand_terms'] = [t.strip() for t in brand_input.split('\n') if t.strip()]

        st.markdown("---")

        # Anomaly detection settings
        st.subheader("🎯 Detection Settings")
        detection_methods = st.multiselect(
            "Methods",
            ['LOF', 'Isolation Forest', 'Z-Score', 'IQR'],
            default=['LOF', 'Isolation Forest']
        )

        contamination = st.slider(
            "Sensitivity",
            min_value=0.01,
            max_value=0.20,
            value=0.05,
            step=0.01,
            help="Expected proportion of anomalies"
        )

        st.markdown("---")

        # Fetch data button
        fetch_clicked = st.button("🚀 Fetch & Analyze Data", type="primary", use_container_width=True)

        return {
            'selected_site': selected_site,
            'use_ga4': use_ga4,
            'start_date': start_date,
            'end_date': end_date,
            'include_yoy': include_yoy,
            'detection_methods': detection_methods,
            'contamination': contamination,
            'fetch_clicked': fetch_clicked
        }


def fetch_data(credentials, config):
    """Fetch data from GSC and optionally GA4"""
    start_str = config['start_date'].strftime('%Y-%m-%d')
    end_str = config['end_date'].strftime('%Y-%m-%d')

    results = {
        'gsc_data': None,
        'ga4_data': None,
        'gsc_yoy': None,
        'ga4_yoy': None
    }

    # Fetch GSC data
    if config['selected_site']:
        with st.spinner("Fetching GSC data..."):
            gsc_connector = GSCConnector(credentials)

            # Debug: Show what we're fetching
            with st.expander("🔍 Debug: Data Fetch Parameters", expanded=False):
                st.write(f"**Property:** {config['selected_site']}")
                st.write(f"**Date Range:** {start_str} to {end_str}")
                st.write(f"**Search Type:** web (default)")

            # Main data with query dimension
            gsc_query_data = gsc_connector.fetch_data_by_date_range(
                site_url=config['selected_site'],
                start_date=start_str,
                end_date=end_str,
                dimensions=['query'],
                max_rows=25000
            )

            # Daily aggregated data
            gsc_daily_data = gsc_connector.fetch_data_by_date_range(
                site_url=config['selected_site'],
                start_date=start_str,
                end_date=end_str,
                dimensions=[],
                max_rows=25000
            )

            # Debug: Show raw data stats
            with st.expander("🔍 Debug: Raw Data Stats", expanded=False):
                st.write(f"**Query data rows:** {len(gsc_query_data)}")
                st.write(f"**Daily data rows:** {len(gsc_daily_data)}")
                if not gsc_daily_data.empty:
                    st.write(f"**Daily data columns:** {list(gsc_daily_data.columns)}")
                    st.write(f"**Daily clicks sum:** {gsc_daily_data['clicks'].sum() if 'clicks' in gsc_daily_data.columns else 'N/A'}")
                    st.write(f"**Daily impressions sum:** {gsc_daily_data['impressions'].sum() if 'impressions' in gsc_daily_data.columns else 'N/A'}")

            results['gsc_data'] = {
                'by_query': gsc_query_data,
                'daily': gsc_daily_data
            }

            # YoY data
            if config['include_yoy']:
                yoy_data = gsc_connector.fetch_yoy_data(
                    site_url=config['selected_site'],
                    current_start=start_str,
                    current_end=end_str,
                    dimensions=[]
                )
                results['gsc_yoy'] = yoy_data

            # Diagnostic data: Query+Page for cannibalization detection
            with st.spinner("Fetching diagnostic data..."):
                query_page_data = gsc_connector.fetch_query_page_data(
                    site_url=config['selected_site'],
                    start_date=start_str,
                    end_date=end_str,
                    max_rows=10000
                )
                results['gsc_data']['query_page'] = query_page_data

                # Search Appearance data
                search_appearance = gsc_connector.fetch_search_appearance_data(
                    site_url=config['selected_site'],
                    start_date=start_str,
                    end_date=end_str
                )
                results['gsc_data']['search_appearance'] = search_appearance

                # Previous period search appearance for comparison
                if config['include_yoy']:
                    prev_start = (datetime.strptime(start_str, '%Y-%m-%d') - timedelta(days=365)).strftime('%Y-%m-%d')
                    prev_end = (datetime.strptime(end_str, '%Y-%m-%d') - timedelta(days=365)).strftime('%Y-%m-%d')
                    prev_search_appearance = gsc_connector.fetch_search_appearance_data(
                        site_url=config['selected_site'],
                        start_date=prev_start,
                        end_date=prev_end
                    )
                    results['gsc_data']['search_appearance_previous'] = prev_search_appearance

    # Fetch GA4 data
    if config['use_ga4'] and st.session_state.get('selected_ga4_property'):
        with st.spinner("Fetching GA4 data..."):
            ga4_connector = GA4Connector(credentials)
            property_id = st.session_state['selected_ga4_property']

            # Organic search traffic
            ga4_organic = ga4_connector.fetch_organic_search_traffic(
                property_id=property_id,
                start_date=start_str,
                end_date=end_str
            )

            # Channel breakdown
            ga4_channels = ga4_connector.fetch_traffic_by_channel(
                property_id=property_id,
                start_date=start_str,
                end_date=end_str
            )

            results['ga4_data'] = {
                'organic': ga4_organic,
                'channels': ga4_channels
            }

            # YoY data
            if config['include_yoy']:
                yoy_data = ga4_connector.fetch_yoy_data(
                    property_id=property_id,
                    current_start=start_str,
                    current_end=end_str
                )
                results['ga4_yoy'] = yoy_data

    return results


def run_analysis(data, config):
    """Run all analysis modules on the fetched data"""
    results = {
        'anomalies': None,
        'brand_classification': None,
        'seasonality': None,
        'segments': None
    }

    gsc_data = data.get('gsc_data', {})

    if not gsc_data:
        return results

    daily_data = gsc_data.get('daily', pd.DataFrame())
    query_data = gsc_data.get('by_query', pd.DataFrame())

    if daily_data.empty:
        return results

    # Anomaly Detection
    method_map = {
        'LOF': 'lof',
        'Isolation Forest': 'iforest',
        'Z-Score': 'zscore',
        'IQR': 'iqr'
    }
    methods = [method_map.get(m, m.lower()) for m in config['detection_methods']]

    detector = AnomalyDetector(contamination=config['contamination'])
    anomaly_results = detector.detect_all(
        daily_data,
        metric_cols=['clicks', 'impressions', 'ctr', 'position'],
        methods=methods
    )
    anomaly_results = detector.classify_anomalies(anomaly_results, 'clicks')
    results['anomalies'] = {
        'data': anomaly_results,
        'summary': detector.get_anomaly_summary(anomaly_results)
    }

    # Brand Classification
    if not query_data.empty and st.session_state.get('brand_terms'):
        classifier = BrandClassifier(st.session_state['brand_terms'])
        classified = classifier.classify_dataframe(query_data)
        results['brand_classification'] = {
            'data': classified,
            'summary': classifier.get_classification_summary(classified),
            'trend': classifier.analyze_brand_trend(classified) if 'date' in classified.columns else {}
        }

    # Seasonality Analysis
    if data.get('gsc_yoy'):
        analyzer = SeasonalityAnalyzer()
        gsc_yoy = data['gsc_yoy']
        current_yoy = gsc_yoy.get('current', pd.DataFrame())
        previous_yoy = gsc_yoy.get('previous', pd.DataFrame())

        # Only proceed if both dataframes have data and required columns
        if (not current_yoy.empty and not previous_yoy.empty and
            'date' in current_yoy.columns and 'date' in previous_yoy.columns):
            yoy_aligned = analyzer.align_yoy_data(current_yoy, previous_yoy)
            yoy_metrics = analyzer.calculate_yoy_metrics(current_yoy, previous_yoy)
        else:
            yoy_aligned = pd.DataFrame()
            yoy_metrics = {}

        patterns = analyzer.detect_seasonal_patterns(daily_data)
        results['seasonality'] = {
            'yoy_aligned': yoy_aligned,
            'yoy_metrics': yoy_metrics,
            'patterns': patterns
        }

    # Segmentation
    if not query_data.empty:
        segmenter = DataSegmenter()
        brand_terms = st.session_state.get('brand_terms', [])
        segmented = segmenter.apply_all_segmentations(query_data, brand_terms=brand_terms)
        results['segments'] = {
            'data': segmented,
            'by_intent': segmenter.get_segment_summary(segmented, 'search_intent') if 'search_intent' in segmented.columns else None,
            'by_length': segmenter.get_segment_summary(segmented, 'query_length_type') if 'query_length_type' in segmented.columns else None
        }

    # Advanced Diagnostics
    diagnostics = DiagnosticAnalyzer()
    diag_results = {}

    # 1. Dark Search Calculator
    total_clicks = daily_data['clicks'].sum() if 'clicks' in daily_data.columns else 0
    query_clicks = query_data['clicks'].sum() if not query_data.empty and 'clicks' in query_data.columns else 0
    diag_results['dark_search'] = diagnostics.calculate_dark_search(total_clicks, query_clicks)

    # 2. CTR vs Position Divergence
    diag_results['ctr_divergence'] = diagnostics.detect_ctr_position_divergence(daily_data)

    # 3. URL Cannibalization
    query_page_data = gsc_data.get('query_page', pd.DataFrame())
    if not query_page_data.empty:
        diag_results['cannibalization'] = diagnostics.detect_url_cannibalization(query_page_data)
    else:
        diag_results['cannibalization'] = {'status': 'no_data', 'cannibalized_queries': []}

    # 4. Engagement Leading Indicator (if GA4 data available)
    ga4_data = data.get('ga4_data', {}).get('organic', pd.DataFrame())
    if not ga4_data.empty and not daily_data.empty:
        diag_results['engagement'] = diagnostics.analyze_engagement_leading_indicator(daily_data, ga4_data)
    else:
        diag_results['engagement'] = {'status': 'no_ga4_data', 'correlations': {}}

    # 5. Search Appearance Analysis
    search_appearance = gsc_data.get('search_appearance', pd.DataFrame())
    search_appearance_prev = gsc_data.get('search_appearance_previous', pd.DataFrame())
    diag_results['search_appearance'] = diagnostics.analyze_search_appearance(
        search_appearance, search_appearance_prev
    )

    # Generate diagnostic summary
    diag_results['summary'] = diagnostics.generate_diagnostic_summary(
        diag_results['dark_search'],
        diag_results['ctr_divergence'],
        diag_results['cannibalization'],
        diag_results['engagement'],
        diag_results['search_appearance']
    )

    results['diagnostics'] = diag_results

    return results


def render_overview_tab(data, analysis_results):
    """Render the overview dashboard tab"""
    st.markdown('<div class="section-header">📊 Performance Overview</div>', unsafe_allow_html=True)

    gsc_data = data.get('gsc_data', {}).get('daily', pd.DataFrame())

    if gsc_data.empty:
        st.warning("No data available")
        return

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        total_clicks = gsc_data['clicks'].sum()
        st.metric("Total Clicks", format_number(total_clicks))

    with col2:
        total_impressions = gsc_data['impressions'].sum()
        st.metric("Total Impressions", format_number(total_impressions))

    with col3:
        avg_ctr = total_clicks / total_impressions if total_impressions > 0 else 0
        st.metric("Avg CTR", format_percentage(avg_ctr))

    with col4:
        avg_position = gsc_data['position'].mean()
        st.metric("Avg Position", f"{avg_position:.1f}")

    # YoY comparison if available
    if analysis_results.get('seasonality', {}).get('yoy_metrics'):
        yoy = analysis_results['seasonality']['yoy_metrics']

        st.markdown("#### Year-over-Year Comparison")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            if 'clicks' in yoy:
                change = yoy['clicks']['pct_change']
                prev_clicks = yoy['clicks'].get('previous', 0)
                st.metric(
                    "Clicks (Last Year)",
                    format_number(prev_clicks),
                    f"{change:+.1f}% now",
                    delta_color="normal"
                )

        with col2:
            if 'impressions' in yoy:
                change = yoy['impressions']['pct_change']
                prev_impressions = yoy['impressions'].get('previous', 0)
                st.metric(
                    "Impressions (Last Year)",
                    format_number(prev_impressions),
                    f"{change:+.1f}% now",
                    delta_color="normal"
                )

    # Performance chart
    chart_builder = ChartBuilder()

    anomaly_data = analysis_results.get('anomalies', {}).get('data')
    if anomaly_data is not None and not anomaly_data.empty:
        fig = chart_builder.create_anomaly_timeline(
            anomaly_data,
            title="Daily Clicks with Anomaly Detection"
        )
        st.plotly_chart(fig, use_container_width=True)

    # Multi-metric dashboard
    if not gsc_data.empty:
        fig = chart_builder.create_multi_metric_dashboard(gsc_data)
        st.plotly_chart(fig, use_container_width=True)


def render_anomalies_tab(analysis_results):
    """Render the anomaly detection tab"""
    st.markdown('<div class="section-header">🎯 Anomaly Detection Results</div>', unsafe_allow_html=True)

    anomaly_info = analysis_results.get('anomalies', {})

    if not anomaly_info:
        st.warning("No anomaly analysis available")
        return

    summary = anomaly_info.get('summary', {})
    anomaly_data = anomaly_info.get('data', pd.DataFrame())

    # Summary cards
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Records", summary.get('total_records', 0))

    with col2:
        st.metric("Consensus Anomalies", summary.get('consensus_anomalies', 0))

    with col3:
        st.metric("Positive Anomalies", summary.get('positive_anomalies', 0))

    with col4:
        st.metric("Negative Anomalies", summary.get('negative_anomalies', 0))

    # Anomalies by method
    st.markdown("#### Detection by Method")
    method_cols = st.columns(len(summary.get('anomalies_by_method', {})) or 1)

    for i, (method, count) in enumerate(summary.get('anomalies_by_method', {}).items()):
        with method_cols[i]:
            st.metric(method.upper(), count)

    # Anomaly timeline
    chart_builder = ChartBuilder()

    if not anomaly_data.empty:
        fig = chart_builder.create_anomaly_timeline(anomaly_data)
        st.plotly_chart(fig, use_container_width=True)

        # Score distribution
        if 'anomaly_score_lof' in anomaly_data.columns:
            fig = chart_builder.create_anomaly_score_distribution(anomaly_data)
            st.plotly_chart(fig, use_container_width=True)

    # Anomaly details table
    st.markdown("#### Anomaly Details")

    if 'anomaly_consensus' in anomaly_data.columns:
        anomalies_only = anomaly_data[anomaly_data['anomaly_consensus']].copy()

        if not anomalies_only.empty:
            display_cols = ['date', 'clicks', 'impressions', 'ctr', 'position', 'anomaly_type', 'anomaly_vote_count']
            display_cols = [c for c in display_cols if c in anomalies_only.columns]

            st.dataframe(
                anomalies_only[display_cols].sort_values('date', ascending=False),
                use_container_width=True
            )
        else:
            st.info("No consensus anomalies detected in this period")


def render_brand_tab(analysis_results):
    """Render the branded vs generic analysis tab"""
    st.markdown('<div class="section-header">🏷️ Branded vs Generic Analysis</div>', unsafe_allow_html=True)

    brand_info = analysis_results.get('brand_classification', {})

    if not brand_info:
        st.info("Configure brand terms in the sidebar to enable this analysis")
        return

    summary = brand_info.get('summary', {})
    chart_builder = ChartBuilder()

    # Summary metrics
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Branded Traffic")
        branded = summary.get('branded', {})
        st.metric("Clicks", format_number(branded.get('clicks', 0)))
        st.metric("Impressions", format_number(branded.get('impressions', 0)))

    with col2:
        st.markdown("#### Generic Traffic")
        generic = summary.get('generic', {})
        st.metric("Clicks", format_number(generic.get('clicks', 0)))
        st.metric("Impressions", format_number(generic.get('impressions', 0)))

    # Traffic split
    totals = summary.get('totals', {})
    st.markdown("#### Traffic Distribution")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Branded Share", f"{totals.get('branded_percentage', 0):.1f}%")
    with col2:
        st.metric("Generic Share", f"{totals.get('generic_percentage', 0):.1f}%")

    # Chart
    brand_data = brand_info.get('data')
    if brand_data is not None and not brand_data.empty and 'date' in brand_data.columns:
        fig = chart_builder.create_brand_generic_chart(brand_data)
        st.plotly_chart(fig, use_container_width=True)


def render_seasonality_tab(analysis_results):
    """Render the seasonality analysis tab"""
    st.markdown('<div class="section-header">📈 Seasonality & YoY Analysis</div>', unsafe_allow_html=True)

    seasonality = analysis_results.get('seasonality', {})

    if not seasonality:
        st.warning("Enable YoY comparison in sidebar for seasonality analysis")
        return

    chart_builder = ChartBuilder()

    # YoY metrics
    yoy_metrics = seasonality.get('yoy_metrics', {})
    if yoy_metrics:
        st.markdown("#### Year-over-Year Performance")

        for metric, values in yoy_metrics.items():
            if isinstance(values, dict) and 'current' in values:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(f"{metric.capitalize()} (Current)", format_number(values['current']))
                with col2:
                    st.metric(f"{metric.capitalize()} (Previous)", format_number(values['previous']))
                with col3:
                    trend_icon = "📈" if values['trend'] == 'up' else "📉" if values['trend'] == 'down' else "➡️"
                    st.metric(f"Change {trend_icon}", f"{values['pct_change']:+.1f}%")

    # YoY chart
    yoy_aligned = seasonality.get('yoy_aligned')
    if yoy_aligned is not None and not yoy_aligned.empty:
        fig = chart_builder.create_yoy_comparison(yoy_aligned)
        st.plotly_chart(fig, use_container_width=True)

    # Weekly patterns
    patterns = seasonality.get('patterns', {})
    if patterns.get('weekly'):
        st.markdown("#### Weekly Pattern")

        weekly_data = []
        for day, values in patterns['weekly'].items():
            weekly_data.append({
                'Day': day,
                'Average': values['average'],
                'Index': values['index']
            })

        weekly_df = pd.DataFrame(weekly_data)
        st.dataframe(weekly_df, use_container_width=True)

        if patterns.get('best_day') and patterns.get('worst_day'):
            st.info(f"Best performing day: **{patterns['best_day']}** | Worst performing day: **{patterns['worst_day']}**")


def render_segments_tab(analysis_results):
    """Render the segmentation analysis tab"""
    st.markdown('<div class="section-header">🧩 Segmentation Analysis</div>', unsafe_allow_html=True)

    segments = analysis_results.get('segments', {})

    if not segments:
        st.warning("No segmentation data available")
        return

    chart_builder = ChartBuilder()
    segment_data = segments.get('data', pd.DataFrame())

    # Search Intent Analysis
    if segments.get('by_intent'):
        st.markdown("#### By Search Intent")

        intent_summary = segments['by_intent']
        for segment, metrics in intent_summary.get('segments', {}).items():
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                st.write(f"**{segment.title()}**")
            with col2:
                st.write(f"Clicks: {format_number(metrics.get('clicks', 0))}")
            with col3:
                st.write(f"Share: {metrics.get('clicks_share', 0):.1f}%")

        if 'search_intent' in segment_data.columns:
            fig = chart_builder.create_segment_breakdown(
                segment_data,
                'search_intent',
                title="Clicks by Search Intent"
            )
            st.plotly_chart(fig, use_container_width=True)

    # Query Length Analysis
    if segments.get('by_length'):
        st.markdown("#### By Query Length")

        length_summary = segments['by_length']
        cols = st.columns(3)
        for i, (segment, metrics) in enumerate(length_summary.get('segments', {}).items()):
            with cols[i % 3]:
                st.metric(
                    segment.replace('_', ' ').title(),
                    format_number(metrics.get('clicks', 0)),
                    f"{metrics.get('clicks_share', 0):.1f}% share"
                )


def render_ai_insights_tab(data, analysis_results, credentials):
    """Render the AI insights tab"""
    st.markdown('<div class="section-header">🤖 AI-Powered Insights</div>', unsafe_allow_html=True)

    ai_analyzer = AIAnalyzer()

    if not ai_analyzer.is_available():
        st.warning("⚠️ OpenRouter API key not configured. Add it to your Streamlit secrets to enable AI insights.")
        st.code("""
# In your .streamlit/secrets.toml file:
[openrouter]
api_key = "your-openrouter-api-key"
        """)
        return

    # Model selection
    model = st.selectbox(
        "AI Model",
        AIAnalyzer.AVAILABLE_MODELS,
        index=0
    )
    ai_analyzer.model = model

    # Analysis options
    st.markdown("#### Select Analysis Type")

    col1, col2 = st.columns(2)

    with col1:
        analyze_anomalies = st.checkbox("Analyze Anomalies", value=True)
        analyze_brand = st.checkbox("Analyze Brand/Generic Split", value=True)

    with col2:
        analyze_segments = st.checkbox("Analyze Segments", value=True)
        generate_summary = st.checkbox("Generate Executive Summary", value=True)

    if st.button("🧠 Generate AI Insights", type="primary"):
        insights = {}

        # Anomaly analysis
        if analyze_anomalies and analysis_results.get('anomalies'):
            with st.spinner("Analyzing anomalies..."):
                anomaly_summary = analysis_results['anomalies']['summary']
                context = {
                    'brand_terms': st.session_state.get('brand_terms', [])
                }
                insight = ai_analyzer.analyze_anomalies(anomaly_summary, context)
                if insight:
                    insights['anomalies'] = insight

        # Brand analysis
        if analyze_brand and analysis_results.get('brand_classification'):
            with st.spinner("Analyzing brand/generic split..."):
                brand_summary = analysis_results['brand_classification']['summary']
                trend_data = analysis_results['brand_classification'].get('trend', {})
                insight = ai_analyzer.analyze_brand_generic_split(brand_summary, trend_data)
                if insight:
                    insights['brand'] = insight

        # Segment analysis
        if analyze_segments and analysis_results.get('segments', {}).get('by_intent'):
            with st.spinner("Analyzing segments..."):
                segment_summary = analysis_results['segments']['by_intent']
                insight = ai_analyzer.analyze_segments(segment_summary, 'search_intent')
                if insight:
                    insights['segments'] = insight

        # Executive summary
        if generate_summary and insights:
            with st.spinner("Generating executive summary..."):
                all_findings = {
                    'anomalies': analysis_results.get('anomalies', {}).get('summary'),
                    'brand': analysis_results.get('brand_classification', {}).get('summary'),
                    'seasonality': analysis_results.get('seasonality', {}).get('yoy_metrics'),
                    'segments': analysis_results.get('segments', {}).get('by_intent')
                }
                summary = ai_analyzer.generate_executive_summary(all_findings)
                if summary:
                    insights['executive_summary'] = summary

        # Display insights
        if insights:
            st.markdown("---")

            if 'executive_summary' in insights:
                st.markdown("### 📋 Executive Summary")
                st.markdown(insights['executive_summary'])
                st.markdown("---")

            if 'anomalies' in insights:
                with st.expander("🎯 Anomaly Analysis", expanded=True):
                    st.markdown(insights['anomalies'])

            if 'brand' in insights:
                with st.expander("🏷️ Brand/Generic Analysis", expanded=True):
                    st.markdown(insights['brand'])

            if 'segments' in insights:
                with st.expander("🧩 Segment Analysis", expanded=True):
                    st.markdown(insights['segments'])
        else:
            st.info("No insights generated. Please select at least one analysis type.")


def render_data_tab(data):
    """Render the raw data tab"""
    st.markdown('<div class="section-header">📋 Raw Data</div>', unsafe_allow_html=True)

    gsc_data = data.get('gsc_data', {})

    tab1, tab2 = st.tabs(["Daily Data", "Query Data"])

    with tab1:
        daily = gsc_data.get('daily', pd.DataFrame())
        if not daily.empty:
            st.dataframe(daily, use_container_width=True)

            csv = daily.to_csv(index=False)
            st.download_button(
                "📥 Download Daily Data",
                csv,
                f"gsc_daily_{datetime.now().strftime('%Y%m%d')}.csv",
                "text/csv"
            )
        else:
            st.info("No daily data available")

    with tab2:
        query = gsc_data.get('by_query', pd.DataFrame())
        if not query.empty:
            st.dataframe(query.head(1000), use_container_width=True)
            st.caption(f"Showing first 1000 of {len(query)} rows")

            csv = query.to_csv(index=False)
            st.download_button(
                "📥 Download Query Data",
                csv,
                f"gsc_queries_{datetime.now().strftime('%Y%m%d')}.csv",
                "text/csv"
            )
        else:
            st.info("No query data available")


def render_diagnostics_tab(analysis_results):
    """Render the advanced diagnostics tab"""
    st.markdown('<div class="section-header">🔬 Advanced Diagnostics</div>', unsafe_allow_html=True)

    diagnostics = analysis_results.get('diagnostics', {})

    if not diagnostics:
        st.warning("No diagnostic data available. Fetch data with diagnostics enabled.")
        return

    summary = diagnostics.get('summary', {})

    # Health Score Card
    st.markdown("#### 🏥 Data Health Score")
    health_score_data = summary.get('health_score', {})
    # Handle both dict format and legacy number format
    if isinstance(health_score_data, dict):
        health_score = health_score_data.get('score', 0)
    else:
        health_score = health_score_data if health_score_data is not None else 0

    # Ensure health_score is a number
    try:
        health_score = float(health_score)
    except (TypeError, ValueError):
        health_score = 0

    health_color = "🟢" if health_score >= 80 else "🟡" if health_score >= 60 else "🔴"

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.metric(
            f"{health_color} Overall Health Score",
            f"{int(health_score)}/100",
            help="Based on dark search rate, CTR stability, cannibalization issues, and search appearances"
        )

    # Show status indicators
    st.markdown("#### Status Indicators")
    status_cols = st.columns(5)

    indicators = [
        ("Dark Search", diagnostics.get('dark_search', {}).get('status', 'unknown')),
        ("CTR Stability", diagnostics.get('ctr_divergence', {}).get('status', 'unknown')),
        ("Cannibalization", diagnostics.get('cannibalization', {}).get('status', 'unknown')),
        ("Engagement", diagnostics.get('engagement', {}).get('status', 'unknown')),
        ("Rich Results", diagnostics.get('search_appearance', {}).get('status', 'unknown'))
    ]

    status_emoji = {'healthy': '✅', 'warning': '⚠️', 'critical': '🔴', 'no_data': '⬜', 'unknown': '❓', 'no_ga4_data': '⬜'}

    for i, (name, status) in enumerate(indicators):
        with status_cols[i]:
            st.metric(name, status_emoji.get(status, '❓'))

    st.markdown("---")

    # 1. Dark Search Analysis
    st.markdown("#### 🔍 Dark Search Analysis")
    dark_search = diagnostics.get('dark_search', {})

    if dark_search.get('status') != 'no_data':
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Clicks", format_number(dark_search.get('total_clicks', 0)))
        with col2:
            st.metric("Query-Attributed Clicks", format_number(dark_search.get('query_level_clicks', dark_search.get('query_clicks', 0))))
        with col3:
            dark_rate = dark_search.get('dark_search_percentage', dark_search.get('dark_search_rate', 0))
            st.metric(
                "Dark Search Rate",
                f"{dark_rate:.1f}%",
                help="Percentage of clicks not attributed to specific queries"
            )

        if dark_rate > 30:
            st.warning(f"⚠️ High dark search rate ({dark_rate:.1f}%). A significant portion of traffic cannot be attributed to specific queries. This is common for sites with high branded search.")
        elif dark_rate > 15:
            st.info(f"ℹ️ Moderate dark search rate ({dark_rate:.1f}%). Some traffic is hidden, but this is within normal ranges.")
        else:
            st.success(f"✅ Low dark search rate ({dark_rate:.1f}%). Most traffic is properly attributed.")
    else:
        st.info("Dark search data not available")

    st.markdown("---")

    # 2. CTR vs Position Divergence
    st.markdown("#### 📊 CTR vs Position Divergence")
    ctr_div = diagnostics.get('ctr_divergence', {})

    if ctr_div.get('status') not in ['no_data', 'unknown']:
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Divergence Score", f"{ctr_div.get('divergence_score', 0):.2f}")
        with col2:
            st.metric("Correlation", f"{ctr_div.get('correlation', 0):.2f}")

        divergent_days = ctr_div.get('divergent_days', [])
        if divergent_days:
            st.warning(f"⚠️ Found {len(divergent_days)} days with CTR/Position divergence - position improved but CTR dropped (or vice versa). This could indicate SERP feature changes or click-through behavior shifts.")
            with st.expander("View Divergent Days"):
                st.dataframe(pd.DataFrame(divergent_days), use_container_width=True)
        else:
            st.success("✅ No significant CTR/Position divergence detected. Rankings and CTR are moving together as expected.")
    else:
        st.info("CTR divergence analysis requires time-series data")

    st.markdown("---")

    # 3. URL Cannibalization
    st.markdown("#### 🔄 URL Cannibalization Detection")
    cannibal = diagnostics.get('cannibalization', {})

    if cannibal.get('status') not in ['no_data', 'unknown']:
        cannibalized = cannibal.get('cannibalized_queries', [])

        if cannibalized:
            st.warning(f"⚠️ Found {len(cannibalized)} queries with potential cannibalization (multiple URLs competing)")

            # Show top cannibalized queries
            with st.expander(f"View Cannibalized Queries ({len(cannibalized)})"):
                cannibal_data = []
                for item in cannibalized[:20]:  # Show top 20
                    cannibal_data.append({
                        'Query': item.get('query', 'N/A'),
                        'URLs Competing': item.get('url_count', 0),
                        'Total Clicks': item.get('total_clicks', 0),
                        'Top URL': item.get('top_url', 'N/A'),
                        'Severity': item.get('severity', 'unknown')
                    })
                st.dataframe(pd.DataFrame(cannibal_data), use_container_width=True)

            st.info("💡 **Recommendation**: Consider consolidating content for these queries or using canonical tags to indicate the preferred URL.")
        else:
            st.success("✅ No significant URL cannibalization detected. URLs are well-differentiated for their target queries.")
    else:
        st.info("Cannibalization analysis requires query+page data")

    st.markdown("---")

    # 4. Engagement Leading Indicator
    st.markdown("#### 📈 Engagement Leading Indicator")
    engagement = diagnostics.get('engagement', {})

    if engagement.get('status') == 'no_ga4_data':
        st.info("Enable GA4 integration to analyze engagement as a leading indicator for rankings")
    elif engagement.get('status') not in ['no_data', 'unknown']:
        correlations = engagement.get('correlations', {})

        if correlations:
            st.markdown("**Correlation between GA4 Engagement and GSC Position:**")
            corr_cols = st.columns(len(correlations))

            for i, (metric, value) in enumerate(correlations.items()):
                with corr_cols[i]:
                    color = "🟢" if abs(value) > 0.5 else "🟡" if abs(value) > 0.3 else "⬜"
                    st.metric(f"{color} {metric}", f"{value:.2f}")

            leading_indicator = engagement.get('leading_indicator_detected', False)
            if leading_indicator:
                st.success("✅ Engagement metrics appear to be a leading indicator for rankings. Improving user engagement may help improve positions.")
            else:
                st.info("ℹ️ No strong leading indicator relationship detected between engagement and rankings.")
    else:
        st.info("Engagement analysis not available")

    st.markdown("---")

    # 5. Search Appearance Analysis
    st.markdown("#### ✨ Search Appearance (Rich Results)")
    search_app = diagnostics.get('search_appearance', {})

    if search_app.get('status') not in ['no_data', 'unknown']:
        current_appearances = search_app.get('current', {})
        changes = search_app.get('changes', [])

        if current_appearances:
            st.markdown("**Current Search Appearances:**")
            app_cols = st.columns(min(4, len(current_appearances)) or 1)

            for i, (appearance, metrics) in enumerate(list(current_appearances.items())[:4]):
                with app_cols[i % 4]:
                    st.metric(
                        appearance.replace('_', ' ').title(),
                        format_number(metrics.get('clicks', 0)),
                        f"{metrics.get('impressions', 0):,.0f} imp"
                    )

        if changes:
            st.markdown("**Year-over-Year Changes:**")
            changes_df = pd.DataFrame(changes)
            if not changes_df.empty:
                # Color code changes
                st.dataframe(changes_df, use_container_width=True)

                gains = [c for c in changes if c.get('change', 0) > 0]
                losses = [c for c in changes if c.get('change', 0) < 0]

                if gains:
                    st.success(f"📈 Gained visibility in: {', '.join([g['appearance'] for g in gains])}")
                if losses:
                    st.warning(f"📉 Lost visibility in: {', '.join([l['appearance'] for l in losses])}")
        else:
            st.info("No year-over-year search appearance data available for comparison")
    else:
        st.info("Search appearance data not available for this property")

    # Summary recommendations
    st.markdown("---")
    st.markdown("#### 💡 Diagnostic Recommendations")

    recommendations = summary.get('recommendations', [])
    if recommendations:
        for rec in recommendations:
            priority = rec.get('priority', 'low')
            icon = "🔴" if priority == 'high' else "🟡" if priority == 'medium' else "🟢"
            st.markdown(f"{icon} **{rec.get('area', 'General')}**: {rec.get('message', '')}")
    else:
        st.success("✅ No critical issues detected. Your data appears healthy!")


def main():
    """Main application entry point"""
    init_session_state()

    # Header
    st.markdown('<div class="main-header">🔍 GSC Unusual Patterns Detection</div>', unsafe_allow_html=True)
    st.markdown(
        "<p style='text-align: center; color: #666;'>Advanced anomaly detection and pattern analysis for Google Search Console</p>",
        unsafe_allow_html=True
    )

    # Authentication
    auth_manager = GoogleAuthManager()
    credentials = auth_manager.render_auth_ui()

    if not credentials:
        return

    # Sidebar configuration
    config = render_sidebar(credentials)

    # Fetch and analyze data
    if config['fetch_clicked']:
        if not config['selected_site']:
            st.error("Please select a GSC property")
            return

        data = fetch_data(credentials, config)

        if data['gsc_data']:
            st.session_state['gsc_data'] = data['gsc_data']
            st.session_state['ga4_data'] = data.get('ga4_data')

            with st.spinner("Running analysis..."):
                analysis_results = run_analysis(data, config)
                st.session_state['analysis_results'] = analysis_results
                st.session_state['fetched_data'] = data

            st.success("✅ Analysis complete!")

    # Display results
    if st.session_state.get('analysis_results'):
        data = st.session_state.get('fetched_data', {})
        analysis_results = st.session_state['analysis_results']

        # Main tabs
        tabs = st.tabs([
            "📊 Overview",
            "🎯 Anomalies",
            "🏷️ Brand Analysis",
            "📈 Seasonality",
            "🧩 Segments",
            "🔬 Diagnostics",
            "🤖 AI Insights",
            "📋 Data"
        ])

        with tabs[0]:
            render_overview_tab(data, analysis_results)

        with tabs[1]:
            render_anomalies_tab(analysis_results)

        with tabs[2]:
            render_brand_tab(analysis_results)

        with tabs[3]:
            render_seasonality_tab(analysis_results)

        with tabs[4]:
            render_segments_tab(analysis_results)

        with tabs[5]:
            render_diagnostics_tab(analysis_results)

        with tabs[6]:
            render_ai_insights_tab(data, analysis_results, credentials)

        with tabs[7]:
            render_data_tab(data)

    else:
        st.info("👈 Configure your settings in the sidebar and click 'Fetch & Analyze Data' to begin")


if __name__ == "__main__":
    main()
