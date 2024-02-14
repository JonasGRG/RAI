import argparse
from torch.utils.data import DataLoader
from torchvision import transforms
from pytorch_lightning import Trainer
from model.model import DermatologyModel  # Adjust this import path as necessary
from data.dataset import get_dataset_splits  # Adjust this import path as necessary

def main(args):
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    # Get dataset splits
    train_dataset, val_dataset = get_dataset_splits(args.csv_file, args.image_dir, transform, split_ratio=args.split_ratio)

    # Setup DataLoader
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Instantiate the model
    model = DermatologyModel(num_classes=args.num_classes)

    # Initialize the PyTorch Lightning Trainer
    trainer = Trainer(max_epochs=args.epochs, gpus=args.gpus)

    # Train the model
    trainer.fit(model, train_loader, val_loader)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model on Diverse Dermatology Dataset")
    parser.add_argument("--csv_file", type=str, required=True, help="Path to the CSV file containing annotations")
    parser.add_argument("--image_dir", type=str, required=True, help="Directory containing the images")
    parser.add_argument("--split_ratio", type=float, default=0.8, help="Train/Validation split ratio")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training and validation")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs to train")
    parser.add_argument("--num_classes", type=int, default=2, help="Number of classes in the dataset")
    parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs to use for training")

    args = parser.parse_args()

    main(args)



