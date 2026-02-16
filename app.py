import streamlit as st
import pandas as pd
import joblib
import numpy as np
from llm_reasoning import explain_verdict

# Load model
import os
try:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, 'model.pkl')
    model = joblib.load(model_path)
except:
    st.error(f"Model not found at {model_path}! Please run 'python fake_profile_detector/model_trainer.py' first.")
    st.stop()

from scraper import get_profile_data
import validators

st.set_page_config(page_title="Fake Profile Detector", page_icon="🕵️‍♀️")

st.title("🕵️‍♀️ Fake Social Media Profile Detector")
st.markdown("Enter an **Instagram URL** or manually input details to check if a profile is likely **Real** or **Fake**.")

# --- Platform Selection ---
platform = st.selectbox("Select Social Media Platform", ["Instagram", "Facebook", "LinkedIn", "Twitter (X)", "TikTok"])

# --- URL Scraper Section ---
with st.expander(f"🔗 **Import from {platform} URL**", expanded=True):
    col1, col2 = st.columns([3, 1])
    with col1:
        profile_url = st.text_input(f"Paste {platform} Profile Link", placeholder=f"https://{platform.lower()}.com/username")
    with col2:
        scrape_btn = st.button("✨ Fetch Info", disabled=(platform != "Instagram"), help="Auto-fetch currently only supports Instagram")
    
    if platform != "Instagram":
        st.info(f"💡 Auto-fetch is currently optimized for Instagram. For {platform}, please enter details manually below.")

    if scrape_btn and profile_url and platform == "Instagram":
        if validators.url(profile_url) and "instagram.com" in profile_url:
            with st.spinner("Fetching profile data..."):
                scraped_data = get_profile_data(profile_url)
                if scraped_data:
                    st.session_state['scraped_data'] = scraped_data
                    st.success(f"Data loaded!")
                    st.rerun()
                else:
                    st.error("Could not fetch data. The profile might be private.")
        else:
            st.warning("Please enter a valid Instagram URL.")
            
    # --- Helper Buttons ---
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("🔄 Reset Inputs"):
            st.session_state['scraped_data'] = {}
            st.rerun()
    with col_b:
        if st.button("🎲 Load Demo"):
            st.session_state['scraped_data'] = {
                'username': 'demo_user',
                'num_followers': 1500,
                'num_following': 400,
                'num_posts': 50,
                'len_desc': 120,
                'profile_pic': 1,
                'extern_url': 1,
                'private': 0,
                'ratio_numlen_username': 0.1,
                'len_fullname': 15,
                'ratio_numlen_fullname': 0.0,
                'sim_name_username': 1
            }
            st.rerun()

# Defaults from scraping or standard
default_data = st.session_state.get('scraped_data', {})

# --- Sidebar: Profile Metadata ---
with st.sidebar:
    st.header("📝 Profile Details")
    
    with st.expander("Basics", expanded=True):
        # Auto-fill if data exists
        val_pic = "Yes" if default_data.get('profile_pic', 1) == 1 else "No"
        val_url = "Yes" if default_data.get('extern_url', 0) == 1 else "No"
        val_priv = "Yes" if default_data.get('private', 0) == 1 else "No"
        
        profile_pic = st.radio("Profile Picture", ["Yes", "No"], index=0 if val_pic=="Yes" else 1, horizontal=True)
        extern_url = st.radio("Has Bio Link?", ["Yes", "No"], index=0 if val_url=="Yes" else 1, horizontal=True)
        private = st.radio("Account Privacy", ["Public", "Private"], index=0 if val_priv=="No" else 1, horizontal=True)
        
        len_desc = st.slider("Bio Length (chars)", 0, 150, default_data.get('len_desc', 50))

    with st.expander("Engagement Stats", expanded=True):
        num_posts = st.number_input("Posts", min_value=0, value=default_data.get('num_posts', 10))
        num_followers = st.number_input("Followers", min_value=0, value=default_data.get('num_followers', 100))
        num_following = st.number_input("Following", min_value=0, value=default_data.get('num_following', 100))
    
    with st.expander("Advanced Metrics (Optional)", expanded=False):
        ratio_numlen_username = st.slider("Digits in Username (%)", 0.0, 1.0, float(default_data.get('ratio_numlen_username', 0.0)))
        len_fullname = st.number_input("Full Name Length", min_value=0, value=default_data.get('len_fullname', 10))
        ratio_numlen_fullname = st.slider("Digits in Name (%)", 0.0, 1.0, float(default_data.get('ratio_numlen_fullname', 0.0)))
        
        # Map back 0/1/2 to index
        sim_idx = default_data.get('sim_name_username', 0)
        sim_name_username = st.selectbox("Username vs Name Match", ["No match", "Partial match", "Full match"], index=sim_idx)

# Prepare input data
input_data = {
    'profile_pic': 1 if profile_pic == "Yes" else 0,
    'ratio_numlen_username': ratio_numlen_username,
    'len_fullname': len_fullname,
    'ratio_numlen_fullname': ratio_numlen_fullname,
    'sim_name_username': 0 if sim_name_username == "No match" else (1 if sim_name_username == "Partial match" else 2),
    'len_desc': len_desc,
    'extern_url': 1 if extern_url == "Yes" else 0,
    'private': 1 if private == "Private" else 0, # Note: changed prompt to Public/Private
    'num_posts': num_posts,
    'num_followers': num_followers,
    'num_following': num_following
}

# Convert to DataFrame
df_input = pd.DataFrame([input_data])

# Button
if st.button("Analyze Profile", type="primary"):
    with st.spinner("Analyzing patterns..."):
        # Prediction
        prediction = model.predict(df_input)[0]
        probability = model.predict_proba(df_input)[0]
        
        # LLM Explanation
        explanation = explain_verdict(input_data, prediction, probability)
        
        # Display
        st.markdown("### Verdict")
        if prediction == 1:
            st.error(f"🚨 **FAKE ACCOUNT DETECTED** ({probability[1]:.1%} confidence)")
        else:
            st.success(f"✅ **Likely Real Account** ({probability[0]:.1%} confidence)")
            
        st.markdown("### AI Analysis")
        st.info(explanation)

        # Raw Data Expander
        with st.expander("🔍 View Detailed Analysis"):
            # Create a more readable format
            display_data = {
                "Profile Picture": "Yes" if input_data['profile_pic'] == 1 else "No",
                "Has Bio Link": "Yes" if input_data['extern_url'] == 1 else "No",
                "Account Privacy": "Private" if input_data['private'] == 1 else "Public",
                "Posts": input_data['num_posts'],
                "Followers": input_data['num_followers'],
                "Following": input_data['num_following'],
                "Bio Length": f"{input_data['len_desc']} characters",
                "Username/Name Match": ["No", "Partial", "Full"][input_data['sim_name_username']],
                "Digits in Username": f"{input_data['ratio_numlen_username']:.0%}",
                "Digits in Name": f"{input_data['ratio_numlen_fullname']:.0%}",
                "Full Name Length": f"{input_data['len_fullname']} characters"
            }
            st.table(pd.DataFrame(display_data.items(), columns=["Feature", "Value"]).set_index("Feature"))
