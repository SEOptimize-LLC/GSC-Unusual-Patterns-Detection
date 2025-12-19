"""
Google OAuth authentication handler for GSC and GA4
"""
import streamlit as st
from google_auth_oauthlib.flow import Flow
from google.oauth2.credentials import Credentials
from typing import Optional, Tuple, List
from config.settings import ALL_SCOPES


class GoogleAuthManager:
    """Handles Google OAuth authentication flow"""

    # Support multiple common naming conventions
    KEY_ALIASES = {
        "client_id": ["client_id", "GOOGLE_CLIENT_ID", "clientId", "CLIENT_ID"],
        "client_secret": ["client_secret", "GOOGLE_CLIENT_SECRET", "clientSecret", "CLIENT_SECRET"],
        "redirect_uri": ["redirect_uri", "GOOGLE_REDIRECT_URI", "redirectUri", "REDIRECT_URI", "redirect_url"]
    }

    def __init__(self, scopes: List[str] = None):
        self.scopes = scopes or ALL_SCOPES

    def _get_secret_value(self, key: str) -> Optional[str]:
        """Get secret value, checking multiple possible key names and locations"""
        aliases = self.KEY_ALIASES.get(key, [key])

        # First check under [google] section
        if "google" in st.secrets:
            for alias in aliases:
                if alias in st.secrets["google"]:
                    return st.secrets["google"][alias]

        # Then check at root level
        for alias in aliases:
            if alias in st.secrets:
                return st.secrets[alias]

        # Check for GOOGLE_ prefixed at root
        for alias in aliases:
            prefixed = f"GOOGLE_{alias.upper()}"
            if prefixed in st.secrets:
                return st.secrets[prefixed]

        return None

    def _validate_secrets(self) -> Tuple[bool, List[str]]:
        """Validate that required secrets are configured"""
        missing = []
        for key in ["client_id", "client_secret", "redirect_uri"]:
            if self._get_secret_value(key) is None:
                missing.append(key)

        return len(missing) == 0, missing

    def _show_secrets_debug(self):
        """Show debug info about available secrets (without revealing values)"""
        with st.expander("🔧 Debug: Check Secrets Configuration"):
            st.markdown("**Expected format in Streamlit Secrets:**")
            st.code("""[google]
client_id = "your-client-id.apps.googleusercontent.com"
client_secret = "your-client-secret"
redirect_uri = "https://your-app.streamlit.app/"
            """, language="toml")

            st.markdown("**What the app found:**")

            # Check what sections exist
            available_sections = list(st.secrets.keys()) if hasattr(st.secrets, 'keys') else []
            st.write(f"Available secret sections: `{available_sections}`")

            # Check each required key
            for key in ["client_id", "client_secret", "redirect_uri"]:
                value = self._get_secret_value(key)
                if value:
                    # Show masked value
                    masked = value[:10] + "..." if len(value) > 10 else "***"
                    st.success(f"✅ `{key}`: Found ({masked})")
                else:
                    st.error(f"❌ `{key}`: Not found")

            st.markdown("**Alternative formats also supported:**")
            st.code("""# Root level (no [google] section)
GOOGLE_CLIENT_ID = "your-client-id"
GOOGLE_CLIENT_SECRET = "your-client-secret"
GOOGLE_REDIRECT_URI = "https://your-app.streamlit.app/"
            """, language="toml")

    def _get_client_config(self) -> dict:
        """Build OAuth client configuration from secrets"""
        return {
            "web": {
                "client_id": self._get_secret_value("client_id"),
                "client_secret": self._get_secret_value("client_secret"),
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
                "redirect_uris": [self._get_secret_value("redirect_uri")]
            }
        }

    def get_auth_url(self) -> Optional[Tuple[str, Flow]]:
        """Generate OAuth authorization URL"""
        is_valid, missing = self._validate_secrets()
        if not is_valid:
            st.error(f"Google OAuth credentials not found. Missing: {', '.join(missing)}")
            self._show_secrets_debug()
            return None

        try:
            flow = Flow.from_client_config(
                self._get_client_config(),
                scopes=self.scopes,
                redirect_uri=self._get_secret_value("redirect_uri")
            )

            auth_url, _ = flow.authorization_url(
                access_type='offline',
                include_granted_scopes='true',
                prompt='consent'
            )

            return auth_url, flow

        except Exception as e:
            st.error(f"Error generating auth URL: {str(e)}")
            return None

    def get_credentials_from_code(self, auth_code: str) -> Optional[Credentials]:
        """Exchange authorization code for credentials"""
        is_valid, missing = self._validate_secrets()
        if not is_valid:
            st.error(f"Missing credentials: {', '.join(missing)}")
            return None

        try:
            flow = Flow.from_client_config(
                self._get_client_config(),
                scopes=self.scopes,
                redirect_uri=self._get_secret_value("redirect_uri")
            )

            flow.fetch_token(code=auth_code)
            return flow.credentials

        except Exception as e:
            st.error(f"Error exchanging code for credentials: {str(e)}")
            return None

    def render_auth_ui(self) -> Optional[Credentials]:
        """Render authentication UI and handle OAuth flow"""
        from auth.credentials import CredentialManager

        # Check for auth code in URL
        auth_code = st.query_params.get("code")

        # Load existing credentials
        credentials = CredentialManager.load_credentials()

        # Handle OAuth callback
        if auth_code and not credentials:
            with st.spinner("Processing authorization..."):
                credentials = self.get_credentials_from_code(auth_code)
                if credentials:
                    CredentialManager.save_credentials(credentials)
                    st.success("✅ Successfully authenticated!")
                    st.query_params.clear()
                    st.rerun()

        # If already authenticated, return credentials
        if credentials:
            return credentials

        # Check secrets configuration first
        is_valid, missing = self._validate_secrets()

        # Show authentication UI
        st.markdown("### 🔐 Authentication Required")

        if not is_valid:
            st.error(f"⚠️ Google OAuth credentials not properly configured. Missing: **{', '.join(missing)}**")
            self._show_secrets_debug()
            return None

        st.info("Please authenticate with Google to access your Search Console and Analytics data.")

        col1, col2 = st.columns([1, 2])

        with col1:
            if st.button("🔑 Sign in with Google", type="primary", use_container_width=True):
                auth_result = self.get_auth_url()
                if auth_result:
                    auth_url, _ = auth_result
                    st.session_state["auth_url"] = auth_url

        if "auth_url" in st.session_state:
            st.markdown(f"""
            **Step 1:** Click the link below to authorize the application:

            🔗 [Authorize Access]({st.session_state["auth_url"]})

            **Step 2:** After authorizing, you'll be redirected back to this app.
            """)

        # Manual code input fallback
        st.markdown("---")
        with st.expander("📝 Manual Authorization Code Input"):
            manual_code = st.text_input("Paste the authorization code here:")
            if manual_code and st.button("Submit Code"):
                credentials = self.get_credentials_from_code(manual_code)
                if credentials:
                    CredentialManager.save_credentials(credentials)
                    st.success("✅ Successfully authenticated!")
                    st.rerun()

        return None
