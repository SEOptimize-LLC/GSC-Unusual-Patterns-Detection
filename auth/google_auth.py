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

    def __init__(self, scopes: List[str] = None):
        self.scopes = scopes or ALL_SCOPES
        self._validate_secrets()

    def _validate_secrets(self) -> bool:
        """Validate that required secrets are configured"""
        if "google" not in st.secrets:
            return False

        required_keys = ["client_id", "client_secret", "redirect_uri"]
        return all(key in st.secrets["google"] for key in required_keys)

    def _get_client_config(self) -> dict:
        """Build OAuth client configuration from secrets"""
        return {
            "web": {
                "client_id": st.secrets["google"]["client_id"],
                "client_secret": st.secrets["google"]["client_secret"],
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
                "redirect_uris": [st.secrets["google"]["redirect_uri"]]
            }
        }

    def get_auth_url(self) -> Optional[Tuple[str, Flow]]:
        """Generate OAuth authorization URL"""
        if not self._validate_secrets():
            st.error("Google OAuth credentials not found in secrets. Please configure your secrets.toml file.")
            return None

        try:
            flow = Flow.from_client_config(
                self._get_client_config(),
                scopes=self.scopes,
                redirect_uri=st.secrets["google"]["redirect_uri"]
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
        if not self._validate_secrets():
            return None

        try:
            flow = Flow.from_client_config(
                self._get_client_config(),
                scopes=self.scopes,
                redirect_uri=st.secrets["google"]["redirect_uri"]
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

        # Show authentication UI
        st.markdown("### 🔐 Authentication Required")
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
