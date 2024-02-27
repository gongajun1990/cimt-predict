import os
import torch
import torch.nn as nn
import torchvision.models as models

from torchvision import transforms
import matplotlib.cm as cm
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

class SiameseResNeXtCAM(nn.Module):
    def __init__(self):
        super(SiameseResNeXtCAM, self).__init__()
        base_model = models.resnext50_32x4d(pretrained=True)
        self.base_layers = nn.Sequential(*list(base_model.children())[:-2])
        self.avgpool = list(base_model.children())[-2]
        self.fc = nn.Linear(base_model.fc.in_features * 2, 2)
        self.gradients = None

    def activations_hook(self, grad):
        self.gradients = grad

    def forward(self, left_eye, right_eye):
        left_features = self.base_layers(left_eye)
        right_features = self.base_layers(right_eye)

        if left_features.requires_grad:
            left_features.register_hook(self.activations_hook)
        if right_features.requires_grad:
            right_features.register_hook(self.activations_hook)

        left_pooled = self.avgpool(left_features).flatten(1)
        right_pooled = self.avgpool(right_features).flatten(1)

        combined = torch.cat((left_pooled, right_pooled), dim=1)
        output = self.fc(combined)

        return output, left_features, right_features

    def get_activations_gradient(self):
        return self.gradients

    def get_activations(self, eye):
        return self.base_layers(eye)

# 图像预处理
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img = Image.open(image_path).convert('RGB')
    img = transform(img).unsqueeze(0)
    return img

# 生成Grad-CAM热图
def generate_grad_cam(model, eye, target_category=None):
    # 如果模型是DataParallel实例，则需要通过module属性访问原始模型
    if isinstance(model, nn.DataParallel):
        model = model.module

    model.eval()
    eye_activations = model.get_activations(eye)
    eye_output, _, _ = model(eye, eye)  # 只需使用一个眼睛的激活

    if target_category is None:
        target_category = eye_output.argmax(dim=1)

    # 反向传播以获取梯度
    model.zero_grad()
    one_hot_output = torch.FloatTensor(1, eye_output.size()[-1]).zero_()
    one_hot_output[0][target_category] = 1
    one_hot_output = one_hot_output.to(eye.device)
    eye_output.backward(gradient=one_hot_output)

    # 获取梯度和激活
    gradients = model.get_activations_gradient()
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])

    # 权重激活并生成Grad-CAM
    for i in range(eye_activations.shape[1]):
        eye_activations[:, i, :, :] *= pooled_gradients[i]

    heatmap = torch.mean(eye_activations, dim=1).squeeze().detach().cpu().numpy()
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    return heatmap

# 可视化Grad-CAM
def apply_color_map(heatmap):
    # 将热图转换为PIL图像，并应用颜色映射
    colormap = plt.get_cmap("Purples")
    heatmap_color = colormap(heatmap)
    heatmap_color = (heatmap_color[:, :, :3] * 255).astype(np.uint8)
    return Image.fromarray(heatmap_color)

def resize_and_crop_image(img, resize_size, crop_size, final_size):
    # Resize the image
    img = img.resize(resize_size)

    # Crop the image
    left = (img.width - crop_size[0]) // 2
    top = (img.height - crop_size[1]) // 2
    right = (img.width + crop_size[0]) // 2
    bottom = (img.height + crop_size[1]) // 2
    img = img.crop((left, top, right, bottom))

    # Resize to final size
    img = img.resize(final_size)
    return img


def process_heatmap(heatmap, img):
    # Normalize the heatmap and set values below average to zero
    heatmap = np.uint8(255 * heatmap)
    # average_value = np.mean(heatmap)
    # heatmap[heatmap < average_value] = 0  # Set values below average to zero

    # Apply color map to the heatmap
    heatmap_colored = apply_color_map(heatmap)
    heatmap_colored = heatmap_colored.resize(img.size)

    # Overlay the heatmap on the original image
    # return Image.blend(img, heatmap_colored, alpha=0.6)
    return Image.blend(heatmap_colored, img, alpha=0.4)

def show_grad_cam(heatmap_left, heatmap_right, left_image_path, right_image_path, save_path):
    # Read and process the original images

    left_img = Image.open(left_image_path).convert('RGB')
    left_img = resize_and_crop_image(left_img, (256, 256), (224, 224), (2048, 2048))

    right_img = Image.open(right_image_path).convert('RGB')
    right_img = resize_and_crop_image(right_img, (256, 256), (224, 224), (2048, 2048))

    # Process each heatmap
    superimposed_img_left = process_heatmap(heatmap_left, left_img)
    superimposed_img_right = process_heatmap(heatmap_right, right_img)

    heatmap_img_left = apply_color_map(heatmap_left).resize(left_img.size)
    heatmap_img_right = apply_color_map(heatmap_right).resize(right_img.size)

    # Combine images horizontally
    def combine_horizontally(*images):
        total_width = sum(img.width for img in images)
        max_height = max(img.height for img in images)
        combined = Image.new('RGB', (total_width, max_height))
        x_offset = 0
        for img in images:
            combined.paste(img, (x_offset, 0))
            x_offset += img.width
        return combined

    # Combine images for each row
    original_row = combine_horizontally(left_img, right_img)
    heatmap_row = combine_horizontally(heatmap_img_left, heatmap_img_right)
    superimposed_row = combine_horizontally(superimposed_img_left, superimposed_img_right)

    # Combine rows vertically
    total_height = original_row.height + heatmap_row.height + superimposed_row.height
    combined_img = Image.new('RGB', (original_row.width, total_height))
    combined_img.paste(original_row, (0, 0))
    combined_img.paste(heatmap_row, (0, original_row.height))
    combined_img.paste(superimposed_row, (0, original_row.height + heatmap_row.height))

    # Display the combined image
    plt.imshow(combined_img)
    plt.axis('off')
    # plt.show()

    # Save the combined image
    combined_img.save(save_path)


# 图像预处理函数
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)
    return image.to(device)

def process_subdirectory(subdir, model, save_dir_base):
    img_left_path = os.path.join(subdir, '左眼.jpg')
    img_right_path = os.path.join(subdir, '右眼.jpg')

    if not os.path.exists(img_left_path) or not os.path.exists(img_right_path):
        print(f"Images not found in {subdir}")
        return

    # 预处理图像
    img_left = preprocess_image(img_left_path).to(device)
    img_right = preprocess_image(img_right_path).to(device)

    # 进行预测并生成Grad-CAM
    with torch.no_grad():
        output, left_features, right_features = model(img_left, img_right)
        prediction = output.argmax(dim=1).item()  # 获取预测类别

    # 根据预测结果选择保存目录
    save_dir = os.path.join(save_dir_base, str(prediction))
    os.makedirs(save_dir, exist_ok=True)  # 确保保存目录存在

    # 生成Grad-CAM热图
    left_heatmap = generate_grad_cam(model, img_left)
    right_heatmap = generate_grad_cam(model, img_right)

    # 处理并保存热图
    subdir_name = os.path.basename(subdir)
    save_heatmap_path = os.path.join(save_dir, f'{subdir_name}_grad_cam.png')

    show_grad_cam(left_heatmap, right_heatmap, img_left_path, img_right_path, save_heatmap_path)





if __name__ == '__main__':
    # 加载模型
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 加载模型
    model = SiameseResNeXtCAM().to(device)

    # 如果有多个 GPU，则使用 DataParallel
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    # 加载模型权重
    model.load_state_dict(torch.load(r'model_state_dict_0.8800.pth', map_location=device))

    # 将模型设置为评估模式
    model.eval()

    data_dir = r'H:\eye-cmit\train\1'
    base_save_dir = r'H:\data\1'  # 设置基础保存目录

    # 确保基础保存目录存在
    os.makedirs(base_save_dir, exist_ok=True)

    # 遍历数据目录
    for subdir, dirs, files in os.walk(data_dir):
        if subdir == data_dir:  # Skip the root directory
            continue
        process_subdirectory(subdir, model, base_save_dir)