# GSC Unusual Patterns Detection

Advanced anomaly detection and pattern analysis tool for Google Search Console and Google Analytics 4 data, powered by AI insights.

## Features

### 🎯 Multi-Algorithm Anomaly Detection
- **Local Outlier Factor (LOF)** - Density-based outlier detection
- **Isolation Forest** - Tree-based anomaly detection
- **Z-Score** - Statistical deviation analysis
- **IQR (Interquartile Range)** - Distribution-based detection
- **Consensus voting** - Combines multiple methods for reliability

### 🏷️ Branded vs Generic Classification
- Automatic query classification
- Custom brand term configuration
- Traffic split analysis
- Trend comparison between branded and generic traffic

### 📈 Year-over-Year Seasonality Analysis
- Aligned YoY comparison
- Seasonal pattern detection (weekly, monthly)
- Change point identification
- Statistical significance testing

### 🧩 Multi-Dimensional Segmentation
- **Search Intent**: Informational, Navigational, Transactional, Commercial
- **Query Length**: Head, Torso, Long-tail
- **URL Structure**: Directory, page type analysis
- **Device & Geography**: When data includes these dimensions

### 🤖 AI-Powered Insights (OpenRouter)
- Anomaly interpretation and root cause analysis
- Brand/generic pattern analysis
- Segment performance insights
- Executive summary generation
- Supports multiple AI models (Claude, GPT-4, Gemini, Llama)

### 📊 Data Source Integration
- **Google Search Console**: Full API integration with pagination
- **Google Analytics 4**: Organic search traffic, channel comparison
- Cross-platform correlation analysis

## Installation

### Local Development

1. Clone the repository:
```bash
git clone https://github.com/yourusername/GSC-Unusual-Patterns-Detection.git
cd GSC-Unusual-Patterns-Detection
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure secrets:
```bash
mkdir -p .streamlit
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# Edit .streamlit/secrets.toml with your credentials
```

5. Run the app:
```bash
streamlit run app.py
```

### Streamlit Cloud Deployment

1. Push your code to GitHub
2. Go to [Streamlit Cloud](https://streamlit.io/cloud)
3. Deploy from your repository
4. Configure secrets in the Streamlit Cloud dashboard:
   - Go to App Settings → Secrets
   - Add your Google OAuth and OpenRouter credentials

## Configuration

### Google Cloud Setup

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select existing
3. Enable the following APIs:
   - Google Search Console API
   - Google Analytics Data API
   - Google Analytics Admin API (optional, for property listing)
4. Create OAuth 2.0 credentials:
   - Go to APIs & Services → Credentials
   - Create OAuth Client ID (Web application)
   - Add authorized redirect URIs:
     - For local: `http://localhost:8501/`
     - For Streamlit Cloud: `https://your-app.streamlit.app/`
5. Download or copy the client ID and secret

### OpenRouter Setup

1. Go to [OpenRouter](https://openrouter.ai/)
2. Create an account and get an API key
3. Add credits to your account for API usage

### Secrets Configuration

For **local development**, create `.streamlit/secrets.toml`:

```toml
[google]
client_id = "your-client-id.apps.googleusercontent.com"
client_secret = "your-client-secret"
redirect_uri = "http://localhost:8501/"

[openrouter]
api_key = "your-openrouter-api-key"
```

For **Streamlit Cloud**, add these same values in the Secrets section of your app settings.

## Usage

1. **Authenticate**: Sign in with Google when prompted
2. **Select Property**: Choose your GSC site from the dropdown
3. **Configure Analysis**:
   - Set date range
   - Add brand terms (optional)
   - Select detection methods
   - Adjust sensitivity
4. **Fetch & Analyze**: Click the button to run the analysis
5. **Explore Results**: Navigate through the tabs to see different insights
6. **AI Insights**: Use the AI tab for deeper analysis (requires OpenRouter key)

## Project Structure

```
GSC-Unusual-Patterns-Detection/
├── app.py                    # Main Streamlit application
├── requirements.txt          # Python dependencies
├── config/
│   ├── __init__.py
│   └── settings.py          # Configuration constants
├── auth/
│   ├── __init__.py
│   ├── google_auth.py       # OAuth flow handler
│   └── credentials.py       # Credential management
├── connectors/
│   ├── __init__.py
│   ├── gsc_connector.py     # GSC API connector
│   └── ga4_connector.py     # GA4 API connector
├── analysis/
│   ├── __init__.py
│   ├── anomaly_detection.py # Multi-algorithm detection
│   ├── brand_classifier.py  # Brand/generic classification
│   ├── seasonality.py       # YoY and seasonal analysis
│   └── segmentation.py      # Data segmentation
├── ai/
│   ├── __init__.py
│   └── openrouter.py        # AI integration
├── visualization/
│   ├── __init__.py
│   └── charts.py            # Plotly visualizations
└── utils/
    ├── __init__.py
    └── helpers.py           # Utility functions
```

## Analysis Methodology

### Anomaly Detection

The tool uses multiple algorithms to detect anomalies:

1. **Data Preparation**: Rolling statistics (7-day mean, std), deviation scores, percentage changes
2. **Feature Scaling**: StandardScaler for normalized feature space
3. **Multi-Algorithm Detection**: Each method votes on anomaly status
4. **Consensus**: Points flagged by majority of methods are marked as consensus anomalies
5. **Classification**: Anomalies classified as positive (unexpected increase) or negative (unexpected decrease)

### Brand Classification

Queries are classified based on:
- Exact match of brand terms
- Partial match with word boundaries
- Case-insensitive matching

### Seasonality Analysis

- **YoY Alignment**: Matches days by day-of-year for fair comparison
- **Weekly Patterns**: Aggregates by day of week, identifies best/worst days
- **Statistical Testing**: ANOVA test for day-of-week effect significance

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License - see LICENSE file for details.

## Acknowledgments

- Inspired by SEO best practices for anomaly detection
- Built with Streamlit, Plotly, and scikit-learn
- AI insights powered by OpenRouter
