import torch
import pandas as pd
import torch.nn as nn
from PIL import Image
import torchvision.models as models
from torch.utils.data import Dataset

import os






class EyeDataset(Dataset):
    def __init__(self, root_dir, age_path, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.pairs = [os.path.join(root_dir, label, pair) for label in os.listdir(root_dir) for pair in
                      os.listdir(os.path.join(root_dir, label))]
        self.age_df = pd.read_excel(age_path)

        self.age_dict = dict(zip(self.age_df['id'], self.age_df['age']))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair_path = self.pairs[idx]
        left_eye_path = os.path.join(pair_path, "左眼.jpg")
        right_eye_path = os.path.join(pair_path, "右眼.jpg")
        label = int(os.path.basename(os.path.dirname(pair_path)))
        id = os.path.basename(pair_path)
        age = self.age_dict.get(id, 0)  # 默认年龄为0，您可以根据实际情况进行修改
        if not isinstance(age, float):
            age = float(age)  # 如果age不是浮点数，将其转换为浮点数
        age = torch.tensor(age, dtype=torch.float32)  # 转换为PyTorch张量
        age = age.view(1)  # 修改形状为[1]

        left_eye = Image.open(left_eye_path)
        right_eye = Image.open(right_eye_path)

        if self.transform:
            left_eye = self.transform(left_eye)
            right_eye = self.transform(right_eye)

        return left_eye, right_eye, label, age



class SiameseResNeXtWithAge(nn.Module):
    """
    孪生网络模型，使用两个相同的ResNeXt50_32x4d子网络处理左眼和右眼的图像。
    """

    def __init__(self):
        super(SiameseResNeXtWithAge, self).__init__()
        base_model = models.resnext50_32x4d(pretrained=True)  # 使用ResNeXt50_32x4d
        self.base_layers = nn.Sequential(*list(base_model.children())[:-1])
        # 新增全连接层专门用于处理年龄信息
        self.age_fc = nn.Linear(1, 128)
        # 更新分类器的输入维度以包括处理过的年龄信息
        self.fc_class = nn.Linear(base_model.fc.in_features * 2 + 128, 2)

    def forward(self, left_eye, right_eye, age):
        left_features = self.base_layers(left_eye).flatten(1)
        right_features = self.base_layers(right_eye).flatten(1)

        # 处理年龄特征
        age_features = self.age_fc(age)
        combined = torch.cat((left_features, right_features, age_features), dim=1)
        output_class = self.fc_class(combined)
        return output_class



