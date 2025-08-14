import os
import sys
import subprocess

def main():
    # Get the project root directory
    project_root = os.path.dirname(os.path.abspath(__file__))
    
    # Check if Streamlit is installed
    try:
        import streamlit
        print("Streamlit is installed. Starting dashboard...")
    except ImportError:
        print("Streamlit is not installed. Installing requirements...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", os.path.join(project_root, "requirements.txt")])
    
    # Start the Streamlit dashboard
    dashboard_path = os.path.join(project_root, "src", "dashboard", "app.py")
    
    # Check if the dashboard file exists
    if not os.path.exists(dashboard_path):
        print(f"Error: Dashboard file not found at {dashboard_path}")
        return
    
    print(f"Starting dashboard from {dashboard_path}")
    subprocess.call([sys.executable, "-m", "streamlit", "run", dashboard_path])

if __name__ == "__main__":
    main()