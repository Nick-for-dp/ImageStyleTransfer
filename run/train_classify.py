import os
import torch
import random
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transform
from model.VGG16 import VGG16
from data.classifiy_dataset import CIFARDataset
from torch.utils.data import DataLoader


def train(model, epochs, device, save_root_path, batch_size=8, learning_rate=1e-3):
    if not os.path.exists(save_root_path):
        os.makedirs(save_root_path, exist_ok=True)
    train_dataset = CIFARDataset(root_path='../train_data/cifar_10', is_train=True)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    lr = learning_rate  # 学习率
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    total_loss = 0.0
    pred_right_item = 0
    total_item = 0
    for i in range(epochs):
        for idx, data in enumerate(train_dataloader):
            img_data = data['img_data'].permute(0, 3, 1, 2).to(device)  # N*H*W*C -> N*C*H*W
            label = data['label'].to(device)  # N*1
            net_out = model(img_data)
            loss = criterion(net_out, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_item += len(label)
            pred_right_item += (torch.argmax(net_out, dim=-1) == label).sum().item()

            if idx % 100 == 0:
                print(f'Index {idx}, total loss: {total_loss / total_item: .4f}')

        if (i + 1) % 2 == 0:
            accuracy = pred_right_item / total_item
            model_name = 'vgg16_epoch{}.pth'.format(i+1)
            save_file_path = os.path.join(save_root_path, model_name)
            torch.save(model.state_dict(), save_file_path)
            print(f'Epoch {i+1}, total loss: {total_loss / total_item: .4f}, accuracy: {accuracy: .3f}')

    print("=====Train Finish.=====")


if __name__ == '__main__':
    calc_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = VGG16(class_num=10).to(calc_device)  # 声明模型
    save_path = '../save_ckpt'
    train(net, epochs=10, device=calc_device, save_root_path=save_path)
