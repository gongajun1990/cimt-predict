import os
from PIL import Image
import torch
import torch.nn as nn
import torchvision.models as models

from torch.utils.data import Dataset

class EyeDataset(Dataset):
    """
    Custom dataset class for loading paired images of left and right eyes and their labels.
    """

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.pairs = [os.path.join(root_dir, label, pair) for label in os.listdir(root_dir) for pair in
                      os.listdir(os.path.join(root_dir, label))]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair_path = self.pairs[idx]
        left_eye_path = os.path.join(pair_path, "左眼.jpg")
        right_eye_path = os.path.join(pair_path, "右眼.jpg")
        label = int(os.path.basename(os.path.dirname(pair_path)))

        left_eye = Image.open(left_eye_path)
        right_eye = Image.open(right_eye_path)

        if self.transform:
            left_eye = self.transform(left_eye)
            right_eye = self.transform(right_eye)

        return left_eye, right_eye, label



class SiameseResNet(nn.Module):
    def __init__(self):
        super(SiameseResNet, self).__init__()
        base_model = models.resnet50(pretrained=True)
        self.base_layers = nn.Sequential(*list(base_model.children())[:-1])
        print(base_model.fc.in_features * 2)
        self.fc = nn.Linear(base_model.fc.in_features * 2, 2)

    def forward(self, left_eye, right_eye):
        left_features = self.base_layers(left_eye).flatten(1)
        right_features = self.base_layers(right_eye).flatten(1)
        combined = torch.cat((left_features, right_features), dim=1)
        output = self.fc(combined)
        return output



class SiameseResNeXt(nn.Module):

    def __init__(self):
        super(SiameseResNeXt, self).__init__()
        base_model = models.resnext50_32x4d(pretrained=True)  # ResNeXt50
        self.base_layers = nn.Sequential(*list(base_model.children())[:-1])
        self.fc = nn.Linear(base_model.fc.in_features * 2, 2)

    def forward(self, left_eye, right_eye):
        left_features = self.base_layers(left_eye).flatten(1)
        right_features = self.base_layers(right_eye).flatten(1)
        combined = torch.cat((left_features, right_features), dim=1)
        output = self.fc(combined)
        return output
