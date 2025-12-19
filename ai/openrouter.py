"""
OpenRouter AI Integration for Deep Data Analysis
"""
import streamlit as st
import requests
import json
import pandas as pd
from typing import Dict, List, Optional, Any
from config.settings import OPENROUTER_API_URL, DEFAULT_AI_MODEL


class AIAnalyzer:
    """
    AI-powered analysis using OpenRouter API

    Features:
    - Anomaly interpretation
    - Pattern explanation
    - Actionable recommendations
    - Root cause analysis
    """

    AVAILABLE_MODELS = [
        'anthropic/claude-3.5-sonnet',
        'anthropic/claude-3-haiku',
        'openai/gpt-4o',
        'openai/gpt-4o-mini',
        'google/gemini-pro-1.5',
        'meta-llama/llama-3.1-70b-instruct'
    ]

    def __init__(self, model: str = None):
        """
        Initialize the AI analyzer

        Args:
            model: OpenRouter model to use
        """
        self.model = model or DEFAULT_AI_MODEL
        self.api_key = self._get_api_key()

    def _get_api_key(self) -> Optional[str]:
        """Get API key from Streamlit secrets"""
        try:
            return st.secrets.get("openrouter", {}).get("api_key")
        except:
            return None

    def _make_request(
        self,
        messages: List[Dict],
        temperature: float = 0.3,
        max_tokens: int = 2000
    ) -> Optional[str]:
        """Make request to OpenRouter API"""
        if not self.api_key:
            st.error("OpenRouter API key not found. Please add it to Streamlit secrets.")
            return None

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://gsc-anomaly-detector.streamlit.app",
            "X-Title": "GSC Anomaly Detector"
        }

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }

        try:
            response = requests.post(
                OPENROUTER_API_URL,
                headers=headers,
                json=payload,
                timeout=60
            )
            response.raise_for_status()

            result = response.json()
            return result.get("choices", [{}])[0].get("message", {}).get("content", "")

        except requests.exceptions.Timeout:
            st.error("AI request timed out. Please try again.")
            return None
        except requests.exceptions.RequestException as e:
            st.error(f"AI request failed: {str(e)}")
            return None

    def analyze_anomalies(
        self,
        anomaly_data: Dict[str, Any],
        context: Dict[str, Any] = None
    ) -> Optional[str]:
        """
        Analyze detected anomalies and provide insights

        Args:
            anomaly_data: Dictionary with anomaly detection results
            context: Additional context (brand terms, industry, etc.)

        Returns:
            AI analysis and recommendations
        """
        prompt = self._build_anomaly_prompt(anomaly_data, context)

        messages = [
            {
                "role": "system",
                "content": """You are an expert SEO analyst specializing in Google Search Console data analysis.
Your task is to analyze anomalies in organic search performance data and provide:
1. Clear interpretation of what each anomaly means
2. Potential root causes (algorithm updates, technical issues, seasonal patterns, etc.)
3. Specific, actionable recommendations to address negative anomalies
4. Strategies to capitalize on positive anomalies

Be concise but thorough. Use data-driven reasoning. Format your response with clear headers and bullet points."""
            },
            {
                "role": "user",
                "content": prompt
            }
        ]

        return self._make_request(messages)

    def _build_anomaly_prompt(
        self,
        anomaly_data: Dict[str, Any],
        context: Dict[str, Any] = None
    ) -> str:
        """Build prompt for anomaly analysis"""
        context = context or {}

        prompt_parts = [
            "## Anomaly Detection Results\n",
            f"Total records analyzed: {anomaly_data.get('total_records', 'N/A')}",
            f"Detection methods used: {', '.join(anomaly_data.get('methods_used', []))}",
            f"\n### Anomalies by Method:"
        ]

        for method, count in anomaly_data.get('anomalies_by_method', {}).items():
            prompt_parts.append(f"- {method}: {count} anomalies detected")

        prompt_parts.extend([
            f"\n### Consensus Anomalies: {anomaly_data.get('consensus_anomalies', 0)}",
            f"- Positive anomalies (unexpected increases): {anomaly_data.get('positive_anomalies', 0)}",
            f"- Negative anomalies (unexpected decreases): {anomaly_data.get('negative_anomalies', 0)}"
        ])

        if context.get('brand_terms'):
            prompt_parts.append(f"\n### Brand Context:")
            prompt_parts.append(f"Brand terms: {', '.join(context['brand_terms'])}")

        if context.get('yoy_data'):
            prompt_parts.append(f"\n### Year-over-Year Context:")
            prompt_parts.append(json.dumps(context['yoy_data'], indent=2))

        if context.get('top_affected_queries'):
            prompt_parts.append(f"\n### Top Affected Queries:")
            for q in context['top_affected_queries'][:10]:
                prompt_parts.append(f"- {q}")

        if context.get('top_affected_pages'):
            prompt_parts.append(f"\n### Top Affected Pages:")
            for p in context['top_affected_pages'][:10]:
                prompt_parts.append(f"- {p}")

        prompt_parts.append("\n\nPlease analyze these anomalies and provide insights on:")
        prompt_parts.append("1. What patterns do you see?")
        prompt_parts.append("2. What are the likely causes?")
        prompt_parts.append("3. What actions should be taken?")
        prompt_parts.append("4. What should be monitored going forward?")

        return "\n".join(prompt_parts)

    def analyze_brand_generic_split(
        self,
        brand_data: Dict[str, Any],
        trend_data: Dict[str, Any] = None
    ) -> Optional[str]:
        """
        Analyze branded vs generic traffic patterns

        Args:
            brand_data: Dictionary with brand classification summary
            trend_data: Optional trend analysis data

        Returns:
            AI analysis of brand/generic patterns
        """
        prompt_parts = [
            "## Branded vs Generic Search Analysis\n",
            "### Traffic Distribution:"
        ]

        if 'branded' in brand_data:
            prompt_parts.append(f"\n**Branded Traffic:**")
            for key, value in brand_data['branded'].items():
                prompt_parts.append(f"- {key}: {value}")

        if 'generic' in brand_data:
            prompt_parts.append(f"\n**Generic Traffic:**")
            for key, value in brand_data['generic'].items():
                prompt_parts.append(f"- {key}: {value}")

        if trend_data:
            prompt_parts.append(f"\n### Trend Analysis:")
            prompt_parts.append(json.dumps(trend_data, indent=2))

        prompt_parts.append("\n\nAnalyze this branded vs generic traffic split and provide insights on:")
        prompt_parts.append("1. Is the brand/generic balance healthy?")
        prompt_parts.append("2. Are there signs of brand erosion or growth?")
        prompt_parts.append("3. What does this indicate about SEO vs brand marketing?")
        prompt_parts.append("4. Recommendations for improving the mix")

        messages = [
            {
                "role": "system",
                "content": """You are an expert SEO strategist analyzing branded vs generic search traffic patterns.
Consider the relationship between brand awareness and organic search performance.
A healthy site typically has both strong branded traffic (indicating brand awareness) and growing generic traffic (indicating SEO success).
If branded drops but generic is stable, it might be a marketing/PR issue, not SEO."""
            },
            {
                "role": "user",
                "content": "\n".join(prompt_parts)
            }
        ]

        return self._make_request(messages)

    def analyze_channel_comparison(
        self,
        gsc_data: Dict[str, Any],
        ga4_data: Dict[str, Any]
    ) -> Optional[str]:
        """
        Compare GSC and GA4 data to identify measurement issues or cannibalization

        Args:
            gsc_data: Summary from Google Search Console
            ga4_data: Summary from Google Analytics 4

        Returns:
            AI analysis of cross-channel patterns
        """
        prompt_parts = [
            "## Cross-Channel Analysis: GSC vs GA4\n",
            "### Google Search Console Data:",
            json.dumps(gsc_data, indent=2, default=str),
            "\n### Google Analytics 4 Data:",
            json.dumps(ga4_data, indent=2, default=str),
            "\n\nAnalyze these cross-channel patterns:",
            "1. Do GSC clicks correlate with GA4 organic sessions?",
            "2. Are there signs of measurement issues?",
            "3. Could there be channel cannibalization?",
            "4. What discrepancies need investigation?"
        ]

        messages = [
            {
                "role": "system",
                "content": """You are an expert digital analyst comparing Google Search Console and Google Analytics data.
Key considerations:
- GSC shows clicks/impressions, GA4 shows sessions/users
- Discrepancies are normal but large gaps indicate issues
- If all channels drop together, it's likely a site-wide issue
- If only organic drops, investigate SEO specifically
- Channel cannibalization can shift traffic between paid/organic"""
            },
            {
                "role": "user",
                "content": "\n".join(prompt_parts)
            }
        ]

        return self._make_request(messages)

    def analyze_segments(
        self,
        segment_summary: Dict[str, Any],
        segment_type: str
    ) -> Optional[str]:
        """
        Analyze performance patterns across segments

        Args:
            segment_summary: Summary statistics by segment
            segment_type: Type of segmentation (intent, url_type, device, country)

        Returns:
            AI analysis of segment patterns
        """
        prompt_parts = [
            f"## Segment Analysis: {segment_type}\n",
            f"Total segments: {segment_summary.get('total_segments', 'N/A')}",
            "\n### Segment Performance:"
        ]

        for segment, metrics in segment_summary.get('segments', {}).items():
            prompt_parts.append(f"\n**{segment}:**")
            for metric, value in metrics.items():
                if isinstance(value, float):
                    prompt_parts.append(f"- {metric}: {value:.2f}")
                else:
                    prompt_parts.append(f"- {metric}: {value}")

        prompt_parts.append(f"\n\nAnalyze these {segment_type} segments and provide:")
        prompt_parts.append("1. Which segments are performing best/worst?")
        prompt_parts.append("2. What optimization opportunities exist?")
        prompt_parts.append("3. Are there concerning patterns?")
        prompt_parts.append("4. Priority recommendations")

        messages = [
            {
                "role": "system",
                "content": f"""You are an expert SEO analyst examining performance by {segment_type} segments.
Look for:
- Underperforming segments with optimization potential
- High-performing segments to double down on
- Unusual patterns that need investigation
- Quick wins vs long-term improvements"""
            },
            {
                "role": "user",
                "content": "\n".join(prompt_parts)
            }
        ]

        return self._make_request(messages)

    def generate_executive_summary(
        self,
        all_findings: Dict[str, Any]
    ) -> Optional[str]:
        """
        Generate an executive summary of all findings

        Args:
            all_findings: Dictionary with all analysis results

        Returns:
            Executive summary with key findings and recommendations
        """
        prompt_parts = [
            "## Complete Analysis Findings\n",
            json.dumps(all_findings, indent=2, default=str),
            "\n\nGenerate a concise executive summary that includes:",
            "1. **Key Findings** (3-5 most important insights)",
            "2. **Risk Assessment** (what needs immediate attention)",
            "3. **Opportunities** (growth potential identified)",
            "4. **Priority Actions** (top 3-5 recommendations)",
            "5. **Monitoring Plan** (what to track going forward)"
        ]

        messages = [
            {
                "role": "system",
                "content": """You are a senior SEO consultant preparing an executive summary for a client.
Be concise, data-driven, and focus on actionable insights.
Avoid jargon when possible.
Prioritize findings by business impact.
Include specific metrics when relevant."""
            },
            {
                "role": "user",
                "content": "\n".join(prompt_parts)
            }
        ]

        return self._make_request(messages, max_tokens=3000)

    def is_available(self) -> bool:
        """Check if AI analysis is available (API key configured)"""
        return self.api_key is not None
