from torch.utils.data import Dataset
from PIL import Image

class MultiScaleImageDataset(Dataset):
    def __init__(self, samples, size_to_transform):
        """
        samples: list of (path, target)
        size_to_transform: dict[int, torchvision.transforms.Compose]
        """
        self.samples = samples
        self.size_to_transform = size_to_transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, key):
        idx, size = key
        path, target = self.samples[idx]

        with open(path, "rb") as f:
            img = Image.open(f).convert("RGB")

        transform = self.size_to_transform[size]
        img = transform(img)

        return img, target
        
class CustomImageDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def set_transform(self, transform):
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, target = self.samples[idx]
        with open(path, 'rb') as f:
            sample = Image.open(f).convert('RGB')
        if self.transform:
            sample = self.transform(sample)
        return sample, target