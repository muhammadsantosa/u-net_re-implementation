from torchvision import transforms
from dataset import BrainTumorSegmentationDataset
from pathlib import Path
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F

def data_loaders(root_dir, train_test_ratio=0.8, train_val_ratio=0.25, target_size=(512, 512), batch_size=16, random_state=42):

    image_dir = Path(root_dir) / "images"
    mask_dir = Path(root_dir) / "masks"

    image_paths = list(image_dir.glob("*.png"))
    mask_paths = list(mask_dir.glob("*.png"))

    train_image_paths, test_image_paths, train_mask_paths, test_mask_paths = train_test_split(
        image_paths, mask_paths, train_size=train_test_ratio, random_state=random_state
    )
    
    train_image_paths, val_image_paths, train_mask_paths, val_mask_paths = train_test_split(
        train_image_paths, train_mask_paths, train_size=train_val_ratio, random_state=random_state
    )

    data_transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
    ])

    train_dataset = BrainTumorSegmentationDataset(
        train_image_paths, 
        train_mask_paths,
        transform=data_transform,
    )
    val_dataset = BrainTumorSegmentationDataset(
        val_image_paths, 
        val_mask_paths,
        transform=data_transform,
    )

    test_dataset = BrainTumorSegmentationDataset(
        test_image_paths, 
        test_mask_paths,
        transform=data_transform,
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

def check_accuracy_and_visualize(loader, model, device="cuda", num_samples=10):
    num_correct = 0
    num_pixels = 0
    cumulative_dice_score = 0
    model.eval()

    samples_visualized = 0

    with torch.no_grad():
        for img, mask in loader:
            img = img.to(device)
            mask = mask.to(device)  

            preds = model(img)
            probs = torch.sigmoid(preds)
            preds = (probs > 0.5).float()  

            num_correct += (preds == mask).sum().item()
            num_pixels += torch.numel(preds)

            intersection = (preds * mask).sum().item()
            dice_score = (2 * intersection) / ((preds + mask).sum().item() + 1e-6)
            cumulative_dice_score += dice_score

            if samples_visualized < num_samples:
                batch_size = img.size(0)
                for i in range(batch_size):
                    if samples_visualized >= num_samples:
                        break
                    original_img = F.to_pil_image(img[i].cpu().squeeze(0))
                    ground_truth_mask = F.to_pil_image(mask[i].cpu().squeeze(0))
                    predicted_mask = F.to_pil_image(preds[i].cpu().squeeze(0))

                    plt.figure(figsize=(12, 4))
                    plt.subplot(1, 3, 1)
                    plt.imshow(original_img, cmap="gray")
                    plt.title("Original Image")
                    plt.axis("off")

                    plt.subplot(1, 3, 2)
                    plt.imshow(ground_truth_mask, cmap="gray")
                    plt.title("Ground Truth Mask")
                    plt.axis("off")

                    plt.subplot(1, 3, 3)
                    plt.imshow(predicted_mask, cmap="gray")
                    plt.title("Predicted Mask")
                    plt.axis("off")

                    plt.tight_layout()
                    plt.show()

                    samples_visualized += 1

    accuracy = num_correct / num_pixels * 100
    avg_dice_score = cumulative_dice_score / len(loader)

    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Average Dice Score: {avg_dice_score*100:.2f}%")
    model.train()