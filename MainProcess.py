import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
from torchvision.transforms import ToTensor
import time
import einops
# import cv2
import numpy as np
import os


class MaskConv2d(nn.Module):
    """
        掩码卷积的实现思路：
            在卷积核组上设置一个mask，在前向传播的时候，先让卷积核组乘mask，再做普通的卷积
    """
    def __init__(self, conv_type, *args, **kwags):
        super().__init__()
        assert conv_type in ('A', 'B')
        self.conv = nn.Conv2d(*args, **kwags)
        H, W = self.conv.weight.shape[-2:]
        # 由于输入输出都是单通道图像，我们只需要在卷积核的h, w两个维度设置掩码
        mask = torch.zeros((H, W), dtype=torch.float32)
        mask[0:H // 2] = 1
        mask[H // 2, 0:W // 2] = 1
        if conv_type == 'B':
            mask[H // 2, W // 2] = 1
        # 为了保证掩码能正确广播到4维的卷积核组上，我们做一个reshape操作
        mask = mask.reshape((1, 1, H, W))
        # register_buffer可以把一个变量加入成员变量的同时，记录到PyTorch的Module中
        # 每当执行model.to(device)把模型中所有参数转到某个设备上时，被注册的变量会跟着转。
        # 第三个参数表示被注册的变量是否要加入state_dict中以保存下来
        self.register_buffer(name='mask', tensor=mask, persistent=False)
        
    def forward(self, x):
        self.conv.weight.data *= self.mask
        conv_res = self.conv(x)
        return conv_res

class ResidualBlock(nn.Module):
    """
        残差块ResidualBlock
    """
    def __init__(self, h, bn=True):
        super().__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(2 * h, h, 1)
        self.bn1 = nn.BatchNorm2d(h) if bn else nn.Identity()
        self.conv2 = MaskConv2d('B', h, h, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(h) if bn else nn.Identity()
        self.conv3 = nn.Conv2d(h, 2 * h, 1)
        self.bn3 = nn.BatchNorm2d(2 * h) if bn else nn.Identity()

    def forward(self, x):
        # 1、ReLU + 1×1 Conv + bn
        y = self.relu(x)
        y = self.conv1(y)
        y = self.bn1(y)
        # 2、ReLU + 3×3 Conv(mask B) + bn
        y = self.relu(y)
        y = self.conv2(y)
        y = self.bn2(y)
        # 3、ReLU + 1×1 Conv + bn
        y = self.relu(y)
        y = self.conv3(y)
        y = self.bn3(y)
        # 4、残差连接
        y = y + x
        return y

class PixelCNN(nn.Module):
    def __init__(self, n_blocks, h, linear_dim, bn=True, color_level=256):
        super().__init__()
        self.conv1 = MaskConv2d('A', 1, 2 * h, 7, 1, 3)
        self.bn1 = nn.BatchNorm2d(2 * h) if bn else nn.Identity()
        self.residual_blocks = nn.ModuleList()
        for _ in range(n_blocks):
            self.residual_blocks.append(ResidualBlock(h, bn))
        self.relu = nn.ReLU()
        self.linear1 = nn.Conv2d(2 * h, linear_dim, 1)
        self.linear2 = nn.Conv2d(linear_dim, linear_dim, 1)
        self.out = nn.Conv2d(linear_dim, color_level, 1)

    def forward(self, x):
        # 1、7 × 7 conv(mask A)
        x = self.conv1(x)
        x = self.bn1(x)
        # 2、Multiple residual blocks
        for block in self.residual_blocks:
            x = block(x)
        x = self.relu(x)
        # 3、1 × 1 conv
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.out(x)
        return x

def get_dataloader(batch_size: int):
    dataset = torchvision.datasets.MNIST(root='./data/minist',
                                         train=True,
                                         transform=ToTensor(),
                                         download=True
                                         )
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def train(model, device, model_path, batch_size=128, color_level=8, n_epochs=1):
    """训练过程"""
    dataloader = get_dataloader(batch_size)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), 1e-3)
    loss_fn = nn.CrossEntropyLoss()

    tic = time.time()
    for e in range(n_epochs):
        total_loss = 0
        for x, _ in dataloader:
            current_batch_size = x.shape[0]
            x = x.to(device)
            # 把训练集的浮点颜色值转换成[0, color_level-1]之间的整型标签
            y = torch.ceil(x * (color_level - 1)).long()
            y = y.squeeze(1)
            predict_y = model(x)
            loss = loss_fn(predict_y, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * current_batch_size
        total_loss /= len(dataloader.dataset)
        toc = time.time()
        torch.save(model.state_dict(), model_path)
        print(f'epoch {e} loss: {total_loss} elapsed {(toc - tic):.2f}s')

def get_img_shape():
    return (1, 28, 28)

def sample(model, device, model_path, output_path, n_sample=1):
    """
        把x初始化成一个0张量。
        循环遍历每一个像素，输入x，把预测出的下一个像素填入x
    """
    model.eval()
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    C, H, W = get_img_shape()  # (1, 28, 28)
    x = torch.zeros((n_sample, C, H, W)).to(device)
    with torch.no_grad():
        for i in range(H):
            for j in range(W):
                # 我们先获取模型的输出，再用softmax转换成概率分布
                output = model(x)
                prob_dist = F.softmax(output[:, :, i, j], -1)
                # 再用torch.multinomial从概率分布里采样出【1】个[0, color_level-1]的离散颜色值
                # 再除以(color_level - 1)把离散颜色转换成浮点[0, 1]
                pixel = torch.multinomial(input=prob_dist, num_samples=1).float() / (color_level - 1)
                # 最后把新像素填入到生成图像中
                x[:, :, i, j] = pixel
    # 乘255变成一个用8位字节表示的图像
    imgs = x * 255
    imgs = imgs.clamp(0, 255)
    imgs = einops.rearrange(imgs, '(b1 b2) c h w -> (b1 h) (b2 w) c', b1=int(n_sample**0.5))

    imgs = imgs.detach().cpu().numpy().astype(np.uint8)
    torch.save(output_path, imgs)


if __name__ == '__main__':
    os.makedirs('work_dirs', exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # 需要注意的是：MNIST数据集的大部分像素都是0和255
    color_level = 8  # or 256
    # 1、创建PixelCNN模型
    model = PixelCNN(n_blocks=15, h=128, linear_dim=32, bn=True, color_level=color_level)
    # 2、模型训练
    model_path = f'work_dirs/model_pixelcnn_{color_level}.pth'
    train(model, device, model_path)
    # 3、采样
    sample(model, device, model_path, f'work_dirs/pixelcnn_{color_level}.jpg')
