import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transform
from model.VGG16 import VGG16
from data.classifiy_dataset import CIFARDataset
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CyclicLR


def train(model, epochs, device, save_root_path, pre_train_model_path, batch_size=8, learning_rate=1e-2):
    if not os.path.exists(save_root_path):
        os.makedirs(save_root_path, exist_ok=True)

    if pre_train_model_path is not None:
        pretrain_model_param = torch.load(pre_train_model_path)
        translate_pre_train_model_param = {}
        for k in pretrain_model_param.keys():
            if k.startswith('features'):
                new_k = k.replace('features', 'conv_net')
                translate_pre_train_model_param[new_k] = pretrain_model_param.get(k, None)
            elif k.startswith('classifier'):
                continue
            else:
                translate_pre_train_model_param[k] = pretrain_model_param.get(k, None)
        model_dict = model.state_dict()  # 原模型参数
        model_dict.update(translate_pre_train_model_param)  # 更新原模型的部分参数，除去classifier那部分
        model.load_state_dict(model_dict)

    train_dataset = CIFARDataset(root_path='../train_data/cifar_10', is_train=True)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    lr = learning_rate  # 学习率
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # 学习率随着训练进行不断调整
    scheduler = CyclicLR(optimizer, base_lr=1e-4, max_lr=learning_rate, cycle_momentum=False)

    transform_for_train = transform.Compose([
        transform.RandomHorizontalFlip(p=0.5),
        transform.RandomVerticalFlip(p=0.5),
        transform.ColorJitter(brightness=0.1, contrast=0.5, saturation=0.5, hue=0.5),
    ])
    # 方差与均值统计来自ImageNet
    normalize_for_train = transform.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])

    total_loss = 0.0
    pred_right_item = 0
    total_item = 0
    for i in range(epochs):
        for idx, data in enumerate(train_dataloader):
            img_data = data['img_data'].permute(0, 3, 1, 2).to(device)  # N*H*W*C -> N*C*H*W
            label = data['label'].to(device)  # N*1

            # 图像取值约束在[0, 1]之间
            img_data = img_data / img_data.max()
            # 按一定概率做数据增广（旋转和颜色变化）
            img_data = transform_for_train(img_data)
            img_data = normalize_for_train(img_data)

            net_out = model(img_data)
            loss = criterion(net_out, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            total_item += len(label)
            pred_right_item += (torch.argmax(net_out, dim=-1) == label).sum().item()

            if idx % 100 == 0:
                print(f'Index {idx}, total loss: {total_loss / total_item: .4f}')

        if (i + 1) % 1 == 0:
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
    pre_train_model = '../pretrain_model/vgg16-397923af.pth'
    train(net, epochs=10, device=calc_device,
          save_root_path=save_path, pre_train_model_path=pre_train_model, learning_rate=1e-3)
