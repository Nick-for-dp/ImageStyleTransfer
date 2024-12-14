import torch
import numpy as np
import torchvision.transforms as transforms
from PIL import Image


transform_method = transforms.Resize([512, 512])


def load_image(img_path):
    img = Image.open(img_path)  # shape H*W*C
    # img.show()
    img_arr = np.array(img)  # shape H*W*C
    img_t = transform_method(torch.from_numpy(img_arr).permute(2, 0, 1))  # shape C*H*W
    img_t = img_t.unsqueeze(0).to(dtype=torch.float32)  # 增加batch维度 shape N*C*H*W
    img_t = img_t / 255.0  # 归一化
    return img_t


if __name__ == '__main__':
    content_img_path = '../train_data/content/01.jpg'
    content_img = load_image(content_img_path)
