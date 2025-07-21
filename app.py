import streamlit as st
import pickle
import pandas as pd
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
</style>
""", unsafe_allow_html=True)

# --- Main header ---
st.markdown('<h1 class="main-header">🛒 Product Recommendation System</h1>', unsafe_allow_html=True)

# --- Load model and data ---
@st.cache_resource(show_spinner=True)
def load_model():
    with open("models/knn_model.pkl", "rb") as f:
        model, user_product_matrix, top_summaries = pickle.load(f)
    return model, user_product_matrix, top_summaries

model, user_product_matrix, top_summaries = load_model()

# --- Get available users ---
def get_users():
    return list(user_product_matrix.index[:10])

user_ids = get_users()

# --- User input ---
st.header("🔍 Get Recommendations")
user_id = st.selectbox("Select a User ID", user_ids) if user_ids else st.text_input("Enter User ID")

if st.button("Get Recommendations", type="primary"):
    if user_id not in user_product_matrix.index:
        st.error("User not found. Please enter a valid user ID.")
    else:
        with st.spinner("Generating recommendations..."):
            user_vector = user_product_matrix.loc[user_id].values.reshape(1, -1)
            distances, indices = model.kneighbors(user_vector, n_neighbors=6)
            recommended_items = set()
            for i in indices[0]:
                sim_user = user_product_matrix.index[i]
                if sim_user != user_id:
                    top_items = user_product_matrix.loc[sim_user].sort_values(ascending=False).head(5).index
                    recommended_items.update(top_items)
            recs = list(recommended_items)[:5]
            if recs:
                st.subheader(f"Top Recommendations for User: {user_id}")
                for pid in recs:
                    col1, col2 = st.columns([1, 4])
                    with col1:
                        st.markdown(f"**Product ID:**")
                        st.code(pid)
                    with col2:
                        st.markdown(f"**Summary:** {top_summaries.get(pid, 'No summary available')}")
                    st.markdown("---")
            else:
                st.info("No recommendations found for this user.")

# --- About section ---
st.markdown("---")
with st.expander("ℹ️ About This App", expanded=False):
    st.markdown("""
    This web app provides personalized product recommendations using a machine learning model trained on Amazon Fine Food Reviews. Enter a user ID to get product suggestions based on similar users' preferences.

    **Features:**
    - Standalone Streamlit app (no API required)
    - Modern, responsive UI
    - DVC and Git for data and model management
    """)
    st.info("Built with ❤️ using Streamlit and Machine Learning")

# --- User ID note ---
st.markdown("""
<div style='margin-top:2rem; font-size:1.05rem; color:#444;'>
<b>Note:</b> You can enter <b>any user ID</b> that exists in the model's data, not just those shown in the dropdown. If you enter a user ID that is not present in the model, you will see an error message.
</div>
""", unsafe_allow_html=True) 