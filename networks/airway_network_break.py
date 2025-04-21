import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np


def normalize_CT(image: np.ndarray) -> np.ndarray:
    min = np.min(image)
    max = np.max(image)
    image_normalized = (image - min) / (max - min)
    return image_normalized


def lumTrans(image):
    lungwin = np.array([-1200., 600.])
    newimg = (image - lungwin[0]) / (lungwin[1] - lungwin[0])
    newimg[newimg < 0] = 0
    newimg[newimg > 1] = 1
    newimg = (newimg * 255).astype('uint8')
    return newimg


def create_feature_maps(init_channel_number, number_of_fmaps):
    return [init_channel_number * 2 ** k for k in range(number_of_fmaps)]


class SSEConv(nn.Module):
    def __init__(self, in_channel=1, out_channel1=1, out_channel2=2, stride=1, kernel_size=3,
                 padding=1, dilation=1, down_sample=1, bias=True):
        self.in_channel = in_channel
        self.out_channel = out_channel1
        super(SSEConv, self).__init__()
        self.conv1 = nn.Conv3d(in_channel, out_channel1, kernel_size, stride=stride, padding=padding * dilation,
                               bias=bias, dilation=dilation)
        self.conv2 = nn.Conv3d(out_channel1, out_channel2,
                               kernel_size=1, stride=1, padding=0, bias=bias)
        self.norm = nn.InstanceNorm3d(out_channel1)
        self.act = nn.LeakyReLU(inplace=True)
        self.up_sample = nn.Upsample(
            scale_factor=down_sample, mode='trilinear', align_corners=True)
        self.conv_se = nn.Conv3d(
            out_channel1, 1, kernel_size=1, stride=1, padding=0, bias=False)
        self.norm_se = nn.Sigmoid()

    def forward(self, x):
        e0 = self.conv1(x)
        e0 = self.norm(e0)
        e0 = self.act(e0)
        e_se = self.conv_se(e0)
        e_se = self.norm_se(e_se)
        e0 = e0 * e_se
        e1 = self.conv2(e0)
        e1 = self.up_sample(e1)
        return e0, e1


class SSEConv2(nn.Module):
    def __init__(self, in_channel=1, out_channel1=1, out_channel2=2, stride=1, kernel_size=3,
                 padding=1, dilation=1, down_sample=1, bias=True):
        self.in_channel = in_channel
        self.out_channel = out_channel1
        super(SSEConv2, self).__init__()
        self.conv1 = nn.Conv3d(in_channel, out_channel1, kernel_size, stride=stride, padding=padding * dilation,
                               bias=bias, dilation=dilation)
        self.conv2 = nn.Conv3d(out_channel1, out_channel2,
                               kernel_size=1, stride=1, padding=0, bias=bias)
        self.norm = nn.InstanceNorm3d(out_channel1)
        self.act = nn.LeakyReLU(inplace=True)
        self.up_sample = nn.Upsample(
            scale_factor=down_sample, mode='trilinear', align_corners=True)
        self.conv_se = nn.Conv3d(
            out_channel1, 1, kernel_size=1, stride=1, padding=0, bias=False)
        self.norm_se = nn.Sigmoid()
        self.conv_se2 = nn.Conv3d(
            out_channel1, 1, kernel_size=1, stride=1, padding=0, bias=False)
        self.norm_se2 = nn.Sigmoid()

    def forward(self, x):
        e0 = self.conv1(x)
        e0 = self.norm(e0)
        e0 = self.act(e0)
        e_se = self.conv_se(e0)
        e_se = self.norm_se(e_se)
        e0 = e0 * e_se
        e_se = self.conv_se2(e0)
        e_se = self.norm_se2(e_se)
        e0 = e0 * e_se
        e1 = self.conv2(e0)
        e1 = self.up_sample(e1)
        return e0, e1


class droplayer(nn.Module):
    def __init__(self, channel_num=1, thr=0.3):
        super(droplayer, self).__init__()
        self.channel_num = channel_num
        self.threshold = thr

    def forward(self, x):
        if self.training:
            r = torch.rand(x.shape[0], self.channel_num, 1, 1, 1).cuda()
            r[r < self.threshold] = 0
            r[r >= self.threshold] = 1
            r = r * self.channel_num / (r.sum() + 0.01)
            return x * r
        else:
            return x


class WingsNet_v3(nn.Module):
    def __init__(self, in_channel=1, n_classes=1):
        self.in_channel = in_channel
        self.n_classes = n_classes
        self.batchnorm = False
        self.bias = True
        self.out_channel2 = 2
        super(WingsNet_v3, self).__init__()
        self.ec1 = SSEConv(self.in_channel, 8,
                           self.out_channel2, bias=self.bias)
        self.ec2 = SSEConv(8, 16, self.out_channel2, bias=self.bias)
        self.ec3 = SSEConv(16, 32, self.out_channel2,
                           bias=self.bias, dilation=2)

        self.ec4 = SSEConv2(32, 32, self.out_channel2,
                            bias=self.bias, down_sample=2)
        self.ec5 = SSEConv2(32, 32, self.out_channel2,
                            bias=self.bias, dilation=2, down_sample=2)
        self.ec6 = SSEConv2(32, 64, self.out_channel2,
                            bias=self.bias, dilation=2, down_sample=2)

        self.ec7 = SSEConv2(64, 64, self.out_channel2,
                            bias=self.bias, down_sample=4)
        self.ec8 = SSEConv2(64, 64, self.out_channel2,
                            bias=self.bias, dilation=2, down_sample=4)
        self.ec9 = SSEConv2(64, 64, self.out_channel2,
                            bias=self.bias, dilation=2, down_sample=4)

        self.ec10 = SSEConv2(64, 64, self.out_channel2,
                             bias=self.bias, down_sample=8)
        self.ec11 = SSEConv2(64, 64, self.out_channel2,
                             bias=self.bias, down_sample=8)
        self.ec12 = SSEConv2(64, 64, self.out_channel2,
                             bias=self.bias, down_sample=8)

        self.pool0 = nn.MaxPool3d(kernel_size=[2, 2, 2], stride=[
                                  2, 2, 2], return_indices=False)
        self.pool1 = nn.MaxPool3d(kernel_size=[2, 2, 2], stride=[
                                  2, 2, 2], return_indices=False)
        self.pool2 = nn.MaxPool3d(kernel_size=[2, 2, 2], stride=[
                                  2, 2, 2], return_indices=False)

        self.up_sample0 = nn.Upsample(
            scale_factor=2, mode='trilinear', align_corners=True)
        self.up_sample1 = nn.Upsample(
            scale_factor=2, mode='trilinear', align_corners=True)
        self.up_sample2 = nn.Upsample(
            scale_factor=2, mode='trilinear', align_corners=True)

        self.dc1 = SSEConv2(128, 64, self.out_channel2,
                            bias=self.bias, down_sample=4)
        self.dc2 = SSEConv2(64, 64, self.out_channel2,
                            bias=self.bias, down_sample=4)
        self.dc3 = SSEConv2(128, 64, self.out_channel2,
                            bias=self.bias, down_sample=2)
        self.dc4 = SSEConv2(64, 32, self.out_channel2,
                            bias=self.bias, down_sample=2)
        self.dc5 = SSEConv(64, 32, self.out_channel2,
                           bias=self.bias, down_sample=1)
        self.dc6 = SSEConv(32, 16, self.out_channel2,
                           bias=self.bias, down_sample=1)

        self.dc0_0 = nn.Sequential(
            nn.Conv3d(24, n_classes, kernel_size=1, stride=1, padding=0, bias=self.bias))
        self.dc0_1 = nn.Sequential(
            nn.Conv3d(12, n_classes, kernel_size=1, stride=1, padding=0, bias=self.bias))
        self.dropout1 = droplayer(channel_num=24, thr=0.3)
        self.dropout2 = droplayer(channel_num=12, thr=0.3)

    def encoder(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                bias=True, batchnorm=True, dilation=1):
        if batchnorm:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding * dilation, bias=bias,
                          dilation=dilation),
                nn.BatchNorm3d(out_channels),
                nn.LeakyReLU(inplace=True))
        else:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding * dilation, bias=bias,
                          dilation=dilation),
                nn.InstanceNorm3d(out_channels),
                nn.LeakyReLU(inplace=True))
        return layer

    def decoder(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                output_padding=0, bias=True):
        layer = nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride,
                               padding=padding, output_padding=output_padding, bias=bias),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(inplace=True))
        return layer

    def forward(self, x):
        e0, s0 = self.ec1(x)
        e1, s1 = self.ec2(e0)
        e1, s2 = self.ec3(e1)  # 64 64 64

        e2 = self.pool0(e1)
        e2, s3 = self.ec4(e2)
        e3, s4 = self.ec5(e2)
        e3, s5 = self.ec6(e3)  # 32 32 32

        e4 = self.pool1(e3)
        e4, s6 = self.ec7(e4)  # 16 16 16
        e5, s7 = self.ec8(e4)
        e5, s8 = self.ec9(e5)
        e6 = self.pool2(e5)
        # print("e6: ", e6.shape)
        e6, s9 = self.ec10(e6)
        e7, s10 = self.ec11(e6)  # 8 8 8
        e7, s11 = self.ec12(e7)

        e8 = self.up_sample0(e7)
        d0, s12 = self.dc1(torch.cat((e8, e5), 1))
        d0, s13 = self.dc2(d0)

        d1 = self.up_sample1(d0)
        d1, s14 = self.dc3(torch.cat((d1, e3), 1))
        d1, s15 = self.dc4(d1)

        d2 = self.up_sample2(d1)
        # print(d2.shape, e1.shape)
        if d2.shape[2] == e1.shape[2]:
            xxx = torch.cat((d2, e1), 1)
        else:
            x2 = torch.zeros((1, 32, 1, 128, 128), dtype=torch.float32).cuda()
            d2 = torch.cat((x2, d2), 2)
            xxx = torch.cat((d2, e1), 1)
        d2, s16 = self.dc5(xxx)
        d2, s17 = self.dc6(d2)
        # print(s12.shape, s13.shape,s14.shape, s15.shape,s16.shape, s17.shape)
        if d2.shape[2] == e1.shape[2]:
            s12 = s12
            s13 = s13
            s14 = s14
            s15 = s15
        else:
            y2 = torch.zeros((1, 2, 1, 128, 128), dtype=torch.float32).cuda()
            s12 = torch.cat((y2, s12), 2)
            z2 = torch.zeros((1, 2, 1, 128, 128), dtype=torch.float32).cuda()
            s13 = torch.cat((z2, s13), 2)
            y4 = torch.zeros((1, 2, 1, 128, 128), dtype=torch.float32).cuda()
            s14 = torch.cat((y4, s14), 2)
            z4 = torch.zeros((1, 2, 1, 128, 128), dtype=torch.float32).cuda()
            s15 = torch.cat((z4, s15), 2)
        pred1 = self.dc0_1(self.dropout2(
            torch.cat((s12, s13, s14, s15, s16, s17), 1)))

        # return pred1
        return torch.sigmoid(pred1)
