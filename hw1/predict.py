"""Script for generating predictions on test data using trained model."""

import os
import zipfile
import torch
from torch.utils.data import DataLoader
import pandas as pd
from train import CONFIG, ImageDataset, get_transforms, ImageClassifier


def predict():
    """Generate predictions for test data."""
    # Load the original prediction.csv to maintain order
    original_df = pd.read_csv("prediction.csv")
    # Load test data
    test_transform = get_transforms(is_train=False)
    test_dataset = ImageDataset(
        os.path.join(CONFIG["DATA_DIR"], "test"),
        transform=test_transform,
        is_test=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=CONFIG["BATCH_SIZE"],
        shuffle=False,
        num_workers=CONFIG["NUM_WORKERS"],
        pin_memory=True,
    )
    # Load model
    model = ImageClassifier().to(CONFIG["DEVICE"])
    checkpoint = torch.load("models/resnet50_best_model.pth")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    # Create a dictionary to store predictions
    predictions_dict = {}
    # Make predictions
    with torch.no_grad():
        for inputs, batch_filenames in test_loader:
            inputs = inputs.to(CONFIG["DEVICE"])
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            # Store predictions with filename (without extension) as key
            for filename, pred in zip(
                batch_filenames, predicted.cpu().numpy()
            ):
                predictions_dict[os.path.splitext(filename)[0]] = int(pred)
    # Update only the pred_label column while maintaining the original order
    original_df["pred_label"] = original_df["image_name"].map(predictions_dict)
    # Verify all data has labels
    missing_labels = original_df["pred_label"].isna().sum()
    if missing_labels > 0:
        print(f"Warning: {missing_labels} images are missing predictions!")
        # Fill missing values with 0 (or any other default value if needed)
        original_df["pred_label"] = (
            original_df["pred_label"].fillna(0).astype(int)
        )
    else:
        # Convert to int even if no missing values
        original_df["pred_label"] = original_df["pred_label"].astype(int)
    # Save predictions maintaining the original order
    original_df.to_csv("prediction.csv", index=False)
    print("Predictions saved to prediction.csv")
    print(f"Total predictions: {len(original_df)}")
    print(f"Unique labels: {original_df['pred_label'].nunique()}")
    print(
        f"Label distribution:\n"
        f"{original_df['pred_label'].value_counts().sort_index()}"
    )
    # Create solution.zip
    with zipfile.ZipFile("solution.zip", "w", zipfile.ZIP_DEFLATED) as zipf:
        zipf.write("prediction.csv")
    print("Created solution.zip containing prediction.csv")


if __name__ == "__main__":
    predict()
