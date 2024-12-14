import copy
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from data.transfer_dataset import load_image
from loss.StyleTransferLoss import ContentLoss, StyleLoss
from torchvision import models

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# 载入预训练模型
vgg16_model = models.vgg16(weights=True).features
# 选取模型指定的卷积层来获取内容特征和风格特征
content_layer = ['Conv_13']
style_layer = ['Conv_1', 'Conv_3', 'Conv_5', 'Conv_8', 'Conv_11']

# 输入内容图像和风格图像
content_img = load_image('../train_data/content/06.jpg').to(device=device)
style_img = load_image('../train_data/vangogh-style/00008.jpg').to(device=device)

content_losses = []
style_losses = []
# 重构model 在适当位置加入内容损失和风格损失模块
model = nn.Sequential().to(device=device)
vgg16 = copy.deepcopy(vgg16_model)
index = 1  # 对卷积层进行计数
for layer in list(vgg16):
    if isinstance(layer, nn.Conv2d):
        name = "Conv_" + str(index)
        model.append(layer.to(device=device))
        if name in content_layer:
            target = model(content_img)  # 计算图像的内容特征
            content_loss = ContentLoss(target=target).to(device=device)  # 添加内容损失模块
            model.append(content_loss)
            content_losses.append(content_loss)

        if name in style_layer:
            target = model(style_img)  # 计算图像的风格特征
            style_loss = StyleLoss(target=target).to(device=device)  # 添加风格损失模块
            model.append(style_loss)
            style_losses.append(style_loss)

    if isinstance(layer, nn.ReLU):
        # model.append(layer.to(device=device))  # 原模型的RELU存在inplace操作
        model.append(nn.ReLU().to(device=device))
        index += 1

    if isinstance(layer, nn.MaxPool2d):
        model.append(layer.to(device=device))

# 训练
epochs = 50
learning_rate = 0.05
lbd = 1e6

input_img = load_image('../train_data/content/06.jpg').to(device=device)  # 复制内容图像作为输入
param = nn.Parameter(input_img.data)  # 输入数据看作模型参数进行更新
optimizer = optim.Adam([param], lr=learning_rate)

loss = 0.0
for i in range(epochs):
    style_loss_value = 0.0
    content_loss_value = 0.0
    model(param)
    for _content_loss in content_losses:
        content_loss_value = content_loss_value + _content_loss.backward()
    for _style_loss in style_losses:
        style_loss_value = style_loss_value + _style_loss.backward()
    loss = content_loss_value + lbd * style_loss_value
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    # 对输入图像更新，可能会导致图像值域超过0-1之间
    param.data.clamp_(0, 1)

    if (i+1) % 10 == 0:
        print(f'Epoch: {i+1}, style loss: {style_loss_value.item(): .4f}, content loss: {content_loss_value.item(): .4f}')
        input_img_arr = input_img[0].permute(1, 2, 0).cpu().numpy()
        input_img_arr = (input_img_arr * 255).clip(0, 255)
        input_image = Image.fromarray(input_img_arr.astype('uint8')).convert('RGB')
        input_image.show()

print("Train finish.")
