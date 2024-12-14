import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import CIFAR10


class CIFARDataset(Dataset):
    def __init__(self, root_path, is_train):
        super(CIFARDataset, self).__init__()
        self.dataset = CIFAR10(root=root_path, train=is_train, download=False)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data = self.dataset[index]
        img = data[0]  # Image类型
        label = data[1]  # int
        img_upper_scale = img.resize([224, 224], 2)  # 将数据从(32, 32)上采样到(224, 224), 2 -> 'bilinear'
        img_arr = np.asarray(img_upper_scale, dtype=np.float32)  # shape: H*W*C, 224*224*3
        label_arr = np.asarray(label, dtype=np.int64)  # cross entropy loss的label需要torch.Long类型
        sample = {
            'img_data': img_arr,
            'label': label_arr
        }
        return sample


if __name__ == '__main__':
    train_dataset = CIFARDataset(root_path='../train_data/cifar_10', is_train=True)
    train_dataloader = DataLoader(train_dataset, batch_size=2)
    for train_data in train_dataloader:
        img_data = train_data.get('img_data', None)
        img_label = train_data.get('label', None)
        print(f'img data shape: {img_data.shape}, label shape: {img_label.shape}')
        break
