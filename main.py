from pathlib import Path

from dataset_loader import load_dataset
from module1_preprocessor.pipeline import process_audio
from module3_model.model import AudioModel


def _pick_test_file(data_dir: Path, preferred_label: str = "happy") -> Path:
	preferred_dir = data_dir / preferred_label
	if preferred_dir.exists() and preferred_dir.is_dir():
		preferred_files = sorted([p for p in preferred_dir.iterdir() if p.is_file()])
		if preferred_files:
			return preferred_files[0]

	for label_dir in sorted([p for p in data_dir.iterdir() if p.is_dir()]):
		files = sorted([p for p in label_dir.iterdir() if p.is_file()])
		if files:
			return files[0]

	raise FileNotFoundError(f"No audio files found in '{data_dir}'.")


def main():
	project_root = Path(__file__).resolve().parent
	data_path = project_root / "data"

	if not data_path.exists():
		raise FileNotFoundError(f"Data folder not found: {data_path}")

	X, y = load_dataset(str(data_path))
	if not X:
		raise ValueError("No training samples found in the data folder.")

	model = AudioModel()
	model.train(X, y)

	test_file = _pick_test_file(data_path)
	test_features = process_audio(str(test_file))
	prediction = model.predict(test_features)

	print("\n=== RUN SUMMARY ===")
	print(f"Training samples: {len(X)}")
	print(f"Classes: {sorted(set(y))}")
	print(f"Test file: {test_file}")
	print(f"Prediction: {prediction}")


if __name__ == "__main__":
	main()