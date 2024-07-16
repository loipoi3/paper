from torch.utils.data import Dataset
import os
import torchvision.transforms as transforms
from torchvision.transforms import v2
from torchvision.io import read_image
import torch


class ChestXRayDataset(Dataset):
    def __init__(self, dataset_type):
        """
        Initialize the dataset by loading image paths and labels.

        Args:
            dataset_type: The type of dataset ('train', 'test', etc.).
        """
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
        """
        Get the size of the dataset.

        Returns:
            The number of items in the dataset.
        """
        return self.size

    def __getitem__(self, idx):
        """
        Get an item from the dataset at the specified index.

        Args:
            idx: The index of the item to retrieve.

        Returns:
            A tuple containing the image tensor and its corresponding label tensor.
        """
        img_path, label = self.paths_labels[idx]
        feature = read_image(img_path)
        label = torch.tensor(label, dtype=torch.long)
        feature = transforms.Resize((224, 224), antialias=True)(feature)
        feature = v2.Grayscale()(feature)
        feature = feature / 255.
        return feature, label
