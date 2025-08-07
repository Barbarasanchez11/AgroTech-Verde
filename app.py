import streamlit as st
import sys
import os


sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from app import main
    main()
except Exception as e:
    st.error(f"Error loading app: {e}")
    st.write("Please check the logs for more details.") 