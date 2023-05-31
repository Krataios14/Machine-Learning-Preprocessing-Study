import subprocess
import pandas as pd

def main():
    print("Starting the pipeline...")
     
    print("Generating simulated data...")
    subprocess.run(["python", "./src/brownian_simulation/Brownian_transformation.py"])
    print("Collecting data...")
    subprocess.run(["python", "./src/data_collection/collect_data.py"])

    print("Extracting Fourier features...")
    subprocess.run(["python", "./src/feature_extraction/extract_Fourier.py"])

    print("Extracting other features...")
    subprocess.run(["python", "./src/feature_extraction/extract_others.py"])

    print("Training model...")
    subprocess.run(["python", "./src/model/train.py"])

    print("Testing model...")
    subprocess.run(["python", "./src/model/test.py"])

    print("Generating predictions...")
    subprocess.run(["python", "./src/model/predict.py"])

    print("All tasks completed successfully.")   

if __name__ == "__main__":
    main()
