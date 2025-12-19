"""
Credential management for Google OAuth
"""
import streamlit as st
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from typing import Optional, Dict, Any


class CredentialManager:
    """Manages storage and retrieval of OAuth credentials in Streamlit session state"""

    CREDENTIALS_KEY = "google_credentials"

    @classmethod
    def save_credentials(cls, credentials: Credentials) -> None:
        """Save credentials to session state"""
        creds_dict = {
            "token": credentials.token,
            "refresh_token": credentials.refresh_token,
            "token_uri": credentials.token_uri,
            "client_id": credentials.client_id,
            "client_secret": credentials.client_secret,
            "scopes": list(credentials.scopes) if credentials.scopes else []
        }
        st.session_state[cls.CREDENTIALS_KEY] = creds_dict

    @classmethod
    def load_credentials(cls) -> Optional[Credentials]:
        """Load credentials from session state and refresh if needed"""
        if cls.CREDENTIALS_KEY not in st.session_state:
            return None

        creds_dict = st.session_state[cls.CREDENTIALS_KEY]

        try:
            credentials = Credentials(
                token=creds_dict["token"],
                refresh_token=creds_dict["refresh_token"],
                token_uri=creds_dict["token_uri"],
                client_id=creds_dict["client_id"],
                client_secret=creds_dict["client_secret"],
                scopes=creds_dict["scopes"]
            )

            # Refresh if expired
            if credentials.expired and credentials.refresh_token:
                credentials.refresh(Request())
                cls.save_credentials(credentials)

            return credentials

        except Exception as e:
            st.error(f"Error loading credentials: {str(e)}")
            return None

    @classmethod
    def clear_credentials(cls) -> None:
        """Clear credentials from session state"""
        if cls.CREDENTIALS_KEY in st.session_state:
            del st.session_state[cls.CREDENTIALS_KEY]

    @classmethod
    def has_valid_credentials(cls) -> bool:
        """Check if valid credentials exist"""
        credentials = cls.load_credentials()
        return credentials is not None and credentials.valid
