import os
from module1_preprocessor.pipeline import run_pipeline

def load_dataset(data_path):
    X = []
    y = []

    # Loop through each label folder (happy, sad, etc.)
    for label in os.listdir(data_path):
        label_path = os.path.join(data_path, label)

        if not os.path.isdir(label_path):
            continue

        # Loop through audio files
        for file in os.listdir(label_path):
            file_path = os.path.join(label_path, file)

            try:
                features = run_pipeline(file_path)
                X.append(features)
                y.append(label)
            except Exception as e:
                print(f"Error processing {file_path}: {e}")

    return X, y