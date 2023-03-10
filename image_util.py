from PIL import Image
from torchvision import transforms
from config import imsize, device
import torch
from pathlib import Path
from matplotlib import pyplot as plt

loaders = transforms.Compose([
    transforms.Resize(imsize),  # 将图片resize到给定大小
    transforms.ToTensor()]  # 将图片转换为Tensor,归一化至[0,1]
)

unloader = transforms.ToPILImage()  # 可以把Tensor转换回Image


def transform_image(image) -> torch.Tensor:
    image = loaders(image).unsqueeze(0)  # 添加一维，变成batch_size=1
    return image.to(device, torch.float)


def load_content_and_style_image(path_content_image, path_style_image):
    image_content = Image.open(Path("image", path_content_image))
    image_style = Image.open(Path("image", path_style_image)).resize(image_content.size, Image.LANCZOS)
    content_image = transform_image(image_content)
    style_image = transform_image(image_style)
    return content_image, style_image


def imshow(tensor, title=None):
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = unloader(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)
