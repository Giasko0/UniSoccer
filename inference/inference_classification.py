import sys, os
sys.path.append('..')
from dataset.video_dataset import VideoCaptionDataset, VideoCaptionDataset_Balanced
from model.MatchVision_classifier import MatchVision_Classifier
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import importlib.util

def load_config(path):
    spec = importlib.util.spec_from_file_location("config", path)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    return config.config

def main():
    ############## Configs ################
    parser = argparse.ArgumentParser(description="Load a Python config file.")
    parser.add_argument('--config_path', type=str, default="config/pretrain_classification.py", help='The path to the Python config file')
    parser.add_argument('--checkpoint_path', type=str, default="pretrained_classification.pth", help='The path to the checkpoint file')
    args = parser.parse_args()

    config = load_config(args.config_path)
    checkpoint_path = args.checkpoint_path

    # Dataset configuration
    config_dataset = config["dataset"]
    config_test_dataset = config_dataset["test"]

    # Model configuration
    config_training_settings = config["training_settings"]
    device_ids = config_training_settings["device_ids"]
    classifier_transformer_type = config_training_settings["classifier_transformer_type"]
    encoder_type = config_training_settings["encoder_type"]
    use_transformer = config_training_settings["use_transformer"]

    # Set up the device
    devices = [torch.device(f'cuda:{i}') for i in device_ids]

    ############## Dataset ################
    test_dataset_type = None
    if config_test_dataset["balanced_or_not"] == "balanced":
        test_dataset_type = VideoCaptionDataset_Balanced
    else:
        test_dataset_type = VideoCaptionDataset

    test_dataset = test_dataset_type(
        json_file=config_test_dataset["json"],
        video_base_dir=config_test_dataset["video_base"],
        sample=config_test_dataset["sample"],
        keywords=config_test_dataset["keywords"],
    )

    test_data_loader = DataLoader(
        test_dataset,
        batch_size=config_test_dataset["batch_size"],
        num_workers=config_test_dataset["num_workers"],
        shuffle=False,
        pin_memory=True,
        persistent_workers=True
    )

    ############## Model ################
    classifier = MatchVision_Classifier(
        keywords=config_test_dataset["keywords"],
        classifier_transformer_type=classifier_transformer_type,
        vision_encoder_type=encoder_type,
        use_transformer=use_transformer
    ).eval()

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    new_state_dict = {key.replace('module.', ''): value for key, value in checkpoint['state_dict'].items()}
    classifier.load_state_dict(new_state_dict)

    # Move model to device and wrap with DataParallel
    classifier = classifier.to(devices[0])
    classifier = torch.nn.DataParallel(classifier, device_ids=device_ids)

    ############## Inference ################
    all_predictions = []
    test_progress_bar = tqdm(enumerate(test_data_loader), total=len(test_data_loader), desc="Inference")

    with torch.no_grad():
        for batch_idx, (frames, _) in test_progress_bar:  # Captions not needed for inference
            frames = frames.to(devices[0])

            # Forward pass
            logits = classifier.module.get_logits(frames)
            predictions = classifier.module.get_types(logits)
            all_predictions.append(predictions.cpu())

    # Combine all predictions into a single tensor
    all_predictions = torch.cat(all_predictions, dim=0)
    # print(all_predictions)
    test_keywords = config_test_dataset["keywords"]
    for pred in all_predictions:
        print(*(test_keywords[event] for event in pred), sep=", ")    
    # # Save predictions to a file or process further
    # output_file = "predictions.pt"
    # torch.save(all_predictions, output_file)
    # print(f"Predictions saved to {output_file}")

if __name__ == "__main__":
    main()