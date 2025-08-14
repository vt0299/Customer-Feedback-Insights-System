import os
import sys
import subprocess
import argparse

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run the Customer Feedback Insights pipeline')
    parser.add_argument('--input', help='Input CSV file with raw feedback data')
    parser.add_argument('--output-dir', help='Output directory for all pipeline artifacts')
    parser.add_argument('--skip', nargs='+', choices=['ingest', 'embed', 'cluster', 'label', 'rag', 'evaluate'],
                        help='Steps to skip in the pipeline')
    args = parser.parse_args()
    
    # Get the project root directory
    project_root = os.path.dirname(os.path.abspath(__file__))
    
    # Build the command
    cmd = [sys.executable, os.path.join(project_root, "src", "pipeline", "run_pipeline.py")]
    
    # Add arguments if provided
    if args.input:
        cmd.extend(["--input", args.input])
    
    if args.output_dir:
        cmd.extend(["--output-dir", args.output_dir])
    
    if args.skip:
        cmd.extend(["--skip"] + args.skip)
    
    # Run the pipeline
    print("Starting Customer Feedback Insights pipeline...")
    subprocess.call(cmd)

if __name__ == "__main__":
    main()