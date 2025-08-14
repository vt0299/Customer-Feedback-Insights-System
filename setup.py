import os
import sys
import subprocess
import shutil
from pathlib import Path

def main():
    # Get the project root directory
    project_root = os.path.dirname(os.path.abspath(__file__))
    
    # Create .env file from .env.example if it doesn't exist
    env_file = os.path.join(project_root, ".env")
    env_example_file = os.path.join(project_root, ".env.example")
    
    if not os.path.exists(env_file) and os.path.exists(env_example_file):
        print("Creating .env file from .env.example...")
        shutil.copy(env_example_file, env_file)
        print("Please edit the .env file to add your API keys and configuration.")
    
    # Install requirements
    print("Installing requirements...")
    requirements_file = os.path.join(project_root, "requirements.txt")
    if os.path.exists(requirements_file):
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", requirements_file])
    else:
        print(f"Error: Requirements file not found at {requirements_file}")
        return
    
    # Create necessary directories if they don't exist
    directories = [
        os.path.join(project_root, "data"),
        os.path.join(project_root, "data", "output")
    ]
    
    for directory in directories:
        if not os.path.exists(directory):
            print(f"Creating directory: {directory}")
            os.makedirs(directory)
    
    print("\nSetup completed successfully!")
    print("\nNext steps:")
    print("1. Edit the .env file to add your OpenAI API key and other configurations")
    print("2. Run the pipeline: python run_pipeline.py")
    print("3. Start the dashboard: python run_dashboard.py")

if __name__ == "__main__":
    main()