import torch
import torch.nn as nn


class VGG16(nn.Module):
    def __init__(self, class_num):
        super(VGG16, self).__init__()
        if class_num > 1:
            self.class_num = class_num
        else:
            raise ValueError('[ERROR] The number of class must bigger than 1.')

        blocks = [
            # block 1
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1, stride=1), nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # block 2
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, stride=1), nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, stride=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # block 3
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1, stride=1), nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, stride=1), nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, stride=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # block 4
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1, stride=1), nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1), nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # block 5
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1), nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1), nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)]

        self.conv_net = nn.Sequential(*blocks)
        self.classifier = nn.Sequential(
            nn.Linear(in_features=512 * 7 * 7, out_features=4096),  # 这里7是根据输入尺寸224计算得到的
            nn.ReLU(),  # inplace=True 可以实现地址传递 而非值传递 节约内存 加快运算效率
            nn.Dropout(p=0.5),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=4096, out_features=self.class_num)
        )

    def forward(self, x):
        # input shape N*C*H*W, H=W=224
        x = self.conv_net(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x


if __name__ == '__main__':
    img = torch.rand([16, 3, 224, 224])
    vgg_model = VGG16(class_num=3)
    out = vgg_model(img)
    print(f'mode out is {out}, model out shape is {out.size()}')
