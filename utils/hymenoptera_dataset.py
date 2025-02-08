import os
import torch
from PIL import Image
from torch.utils.data import Dataset

class BeeAntDataset(Dataset):
    """Dataset containing bee vs ants images."""

    def __init__(self, root_dir, transform=None):
        """
        Arguments:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir  = root_dir
        self.transform = transform

        self.images = []
        self.labels = []

        class_names = os.listdir(self.root_dir)

        for class_name in class_names:
            class_folder = os.path.join(root_dir, class_name)

            # Set our labels for class
            # - 0 : Bee
            # - 1 : Ant
            label = 0.
            if ("ant" in class_name):
                label = 1.

            for file in os.listdir(class_folder):
                image_path = os.path.join(class_folder, file)
                self.images.append(image_path)
                self.labels.append(label)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.images[idx]
        target   = self.labels[idx]

        image = Image.open(img_name)

        if self.transform:
            image = self.transform(image)

        sample = (image, target)

        return sample