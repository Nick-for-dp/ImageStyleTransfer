import torch
import random
import torch.nn as nn
from model.VGG16 import VGG16
from data.classifiy_dataset import CIFARDataset
from torch.utils.data import DataLoader


def test(model, data_loader, device):
    model.eval()  # 推理时不做梯度回传
    total_item = 0
    pred_right_item = 0
    for test_data in data_loader:
        img_data = test_data['img_data'].permute(0, 3, 1, 2).to(device)  # N*H*W*C -> N*C*H*W
        label = test_data['label'].to(device)  # N*1
        net_out = model(img_data)

        total_item += len(label)
        pred_right_item += (torch.argmax(net_out, dim=-1) == label).sum().item()

    print(f'Accuracy: {pred_right_item / total_item: .4f}')


if __name__ == '__main__':
    test_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = VGG16(class_num=10).to(test_device)
    # 读取完成训练的模型
    ckpt_path = '../save_ckpt/vgg16_epoch2.pth'
    checkpoint = torch.load(ckpt_path)
    net.load_state_dict(checkpoint)
    test_dataset = CIFARDataset(root_path='../train_data/cifar_10', is_train=False)
    test_dataloader = DataLoader(test_dataset, batch_size=8)
    test(net, data_loader=test_dataloader, device=test_device)
