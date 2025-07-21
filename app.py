import streamlit as st
import requests
import os

# --- Custom CSS for styling ---
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .recommendation-box {
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
        font-size: 1.2rem;
        font-weight: bold;
        background-color: #e3f2fd;
        border: 2px solid #2196f3;
    }
    .error-box {
        background-color: #ffebee;
        color: #c62828;
        border: 2px solid #ef5350;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
    }
    .info-box {
        background-color: #e8f5e9;
        color: #2e7d32;
        border: 2px solid #66bb6a;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


api_url = os.getenv("API_URL", "http://localhost:8000")
api_key = os.getenv("API_KEY", "changeme123")

# --- Main header ---
st.markdown('<h1 class="main-header">🛒 Product Recommendation System</h1>', unsafe_allow_html=True)

# --- API status check ---
def check_api_status():
    try:
        resp = requests.get(f"{api_url}/", headers={"X-API-Key": api_key}, timeout=5)
        if resp.status_code == 200:
            return True, resp.json()
        else:
            return False, resp.text
    except Exception as e:
        return False, str(e)

api_ok, api_info = check_api_status()
# Place API status
st.markdown(f'''
    <div style="position: absolute; top: 1.5rem; right: 2rem; min-width: 180px; max-width: 250px; z-index: 9999;">
        <div style="font-size: 0.95rem; padding: 0.5rem 1rem; border-radius: 8px; text-align: center; {'background-color: #e8f5e9; color: #2e7d32; border: 1.5px solid #66bb6a;' if api_ok else 'background-color: #ffebee; color: #c62828; border: 1.5px solid #ef5350;'}">
            {'API Online' if api_ok else 'API Offline'}
        </div>
    </div>
''', unsafe_allow_html=True)
if not api_ok:
    st.stop()

# --- Get available users ---
def get_users():
    try:
        resp = requests.get(f"{api_url}/users", headers={"X-API-Key": api_key}, timeout=10)
        if resp.status_code == 200:
            return resp.json().get("users", [])
        else:
            return []
    except Exception:
        return []

user_ids = get_users()

# --- User input ---
st.header("🔍 Get Recommendations")
user_id = st.selectbox("Select a User ID", user_ids) if user_ids else st.text_input("Enter User ID")

if st.button("Get Recommendations", type="primary"):
    with st.spinner("Fetching recommendations..."):
        try:
            resp = requests.post(
                f"{api_url}/recommend",
                headers={"X-API-Key": api_key},
                json={"user_id": user_id},
                timeout=15
            )
            if resp.status_code == 200:
                recs = resp.json().get("recommendations", [])
                if recs:
                    st.subheader(f"Top Recommendations for User: {user_id}")
                    for rec in recs:
                        col1, col2 = st.columns([1, 4])
                        with col1:
                            st.markdown(f"**Product ID:**")
                            st.code(rec['product_id'])
                        with col2:
                            st.markdown(f"**Summary:** {rec['summary']}")
                        st.markdown("---")
                else:
                    st.info("No recommendations found for this user.")
            else:
                st.markdown(f'<div class="error-box">❌ Error: {resp.text}</div>', unsafe_allow_html=True)
        except Exception as e:
            st.markdown(f'<div class="error-box">❌ Exception: {str(e)}</div>', unsafe_allow_html=True)

# --- About section ---
st.markdown("---")
with st.expander("ℹ️ About This App", expanded=False):
    st.markdown("""
    This web app provides personalized product recommendations using a machine learning model trained on Amazon Fine Food Reviews. Enter a user ID to get product suggestions based on similar users' preferences.

    **Features:**
    - FastAPI backend with API
    - Streamlit UI
    - Docker and DVC
    """)
    st.info("Built with ❤️ using Machine Learning")

# --- User ID note ---
st.markdown("""
<div style='margin-top:2rem; font-size:1.05rem; color:#444;'>
<b>Note:</b> You can enter <b>any user ID</b> that exists in the model's data, not just those shown in the dropdown.
</div>
""", unsafe_allow_html=True) 