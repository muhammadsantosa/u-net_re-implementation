from torchvision import transforms
from dataset import BrainTumorSegmentationDataset
from pathlib import Path
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

def data_loaders(root_dir, train_test_ratio=0.8, train_val_ratio=0.25, target_size=(512, 512), random_state=42):

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

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    return train_loader, val_loader, test_loader