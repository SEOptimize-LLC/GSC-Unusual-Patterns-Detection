"""
Configuration settings for GSC Unusual Patterns Detection App
"""

# Google OAuth Scopes
GSC_SCOPES = [
    'https://www.googleapis.com/auth/webmasters.readonly',
    'https://www.googleapis.com/auth/webmasters'
]

GA4_SCOPES = [
    'https://www.googleapis.com/auth/analytics.readonly'
]

ALL_SCOPES = GSC_SCOPES + GA4_SCOPES

# API Configuration
GSC_API_SERVICE = 'searchconsole'
GSC_API_VERSION = 'v1'
GA4_API_SERVICE = 'analyticsdata'
GA4_API_VERSION = 'v1beta'

# Data limits
MAX_GSC_ROWS = 25000
MAX_GA4_ROWS = 100000
DEFAULT_ROWS = 10000

# Anomaly Detection Settings
ANOMALY_DETECTION_METHODS = [
    'Local Outlier Factor (LOF)',
    'Isolation Forest',
    'Z-Score',
    'IQR (Interquartile Range)',
    'DBSCAN'
]

DEFAULT_CONTAMINATION = 0.05
DEFAULT_N_NEIGHBORS = 20
DEFAULT_ZSCORE_THRESHOLD = 3.0

# Segmentation dimensions
GSC_DIMENSIONS = ['query', 'page', 'date', 'country', 'device']
GA4_DIMENSIONS = ['date', 'sessionSource', 'sessionMedium', 'country', 'deviceCategory', 'pagePath']

# Metrics
GSC_METRICS = ['clicks', 'impressions', 'ctr', 'position']
GA4_METRICS = ['sessions', 'totalUsers', 'newUsers', 'screenPageViews', 'bounceRate', 'averageSessionDuration']

# Date ranges
DATE_RANGE_OPTIONS = {
    'Last 7 days': 7,
    'Last 14 days': 14,
    'Last 30 days': 30,
    'Last 3 months': 90,
    'Last 6 months': 180,
    'Last 12 months': 365,
    'Custom range': None
}

# Chart colors
COLORS = {
    'primary': '#1f77b4',
    'positive_anomaly': '#2ecc71',
    'negative_anomaly': '#e74c3c',
    'warning': '#f39c12',
    'neutral': '#95a5a6',
    'branded': '#9b59b6',
    'generic': '#3498db'
}

# OpenRouter settings
OPENROUTER_API_URL = 'https://openrouter.ai/api/v1/chat/completions'
DEFAULT_AI_MODEL = 'anthropic/claude-3.5-sonnet'
