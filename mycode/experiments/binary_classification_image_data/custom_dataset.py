from torch.utils.data import Dataset
import os
import torchvision.transforms as transforms
from torchvision.transforms import v2
from torchvision.io import read_image
import torch


class ChestXRayDataset(Dataset):
    """
    Custom Dataset for Chest X-Ray images.

    Args:
        dataset_type (str): The type of dataset to load ('train', 'test', etc.).
    """
    def __init__(self, dataset_type):
        self.paths_labels = []
        prefix = "datasets/chest_xray/" + dataset_type + "/"

        for item in os.listdir(prefix):
            for img_path in os.listdir(prefix + item):
                if 'bacteria' in img_path:
                    label = 1
                elif 'virus' in img_path:
                    label = 2
                else:
                    label = 0
                self.paths_labels.append((prefix + item + "/" + img_path, label))

        self.size = len(self.paths_labels)
        self.train = False

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        img_path, label = self.paths_labels[idx]
        feature = read_image(img_path)
        label = torch.tensor(label, dtype=torch.long)
        feature = transforms.Resize((224, 224), antialias=True)(feature)
        feature = v2.Grayscale()(feature)
        feature = feature / 255.
        return feature, label
