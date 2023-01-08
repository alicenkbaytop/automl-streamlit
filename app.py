import streamlit as st
import pandas as pd
import os
import pandas_profiling
from streamlit_pandas_profiling import st_profile_report
from pycaret.classification import *

st.set_page_config(page_title="AutoML App", page_icon="icon.png")
st.title("AutoML Application")

with st.sidebar:
    st.image("https://images.squarespace-cdn.com/content/v1/58090c87d1758ec5d1815f6f/1541610301581-R6EX1PXA2U9N3DQDE6QH/machine-learning-cybersecurity-applications.png")
    choice = st.sidebar.selectbox("Options", ["Upload Dataset", "Data Analysis", "Modelling", "Download Model", "Test"])
    st.sidebar.info("This is 'Automated Machine Learning' development application.")
    
if os.path.exists("sourcedata.csv"):
    df = pd.read_csv("sourcedata.csv", index_col=None)
    
if choice == "Upload Dataset":
    file = st.file_uploader("Upload your dataset here")
    if file:
        df = pd.read_csv(file, index_col=None)
        df.to_csv("sourcedata.csv", index=None)
        st.dataframe(df)
        
if choice == "Data Analysis":
    st.title("Exploratory Data Analysis")
    profile_report = df.profile_report()
    st_profile_report(profile_report)
    
if choice == "Modelling":
    target = st.selectbox("Choose the target column", df.columns)
    
    if st.button("Run model"):
        setup(df, target=target)
        setup_df = pull()
        st.info("Settings: ")
        st.dataframe(setup_df)
        best_model = compare_models()
        result = pull()
        st.info("Models: ")
        st.dataframe(result)
        save_model(best_model, "best_model")
        
if choice == "Download Model":
    with open("best_model.pkl", "rb") as f:
        st.download_button("Download model", f, file_name="trained_model")
        
if choice == "Test":
    st.info("Fixing...")