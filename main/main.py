import sys
import subprocess

if __name__ == "__main__":
    input_name = sys.argv[1] if len(sys.argv) > 1 else "housing.csv"
    output_name = sys.argv[2] if len(sys.argv) > 2 else "housing_output.csv"
    
    subprocess.run(["python", "../orchestration/prediction.py"], check=True)
    subprocess.run([
        "python", "../deployment/batch.py",
        "--input_name", input_name,
        "--output_name", output_name
        ], check=True)
    subprocess.run([
        "python", "../monitoring/monitoring.py", 
        "--output_name", output_name
        ], check=True)