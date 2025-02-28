import streamlit.web.cli as stcli
import sys
import os
from src import dashboard

if __name__ == "__main__":
    # If run directly with python app.py, this might not start streamlit server correctly
    # Use 'streamlit run app.py'
    try:
        dashboard.main()
    except Exception as e:
        print(f"Error running dashboard: {e}")