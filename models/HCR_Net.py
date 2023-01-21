#import _init_paths
import sys
import os.path as osp
import torch
import torch.nn as nn
#from utils import init_weights, count_param
import torch.nn.init as init

def count_param(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        init.kaiming_normal_(m.weight.data)

def init_weights(m):
    if type(m) == nn.Linear:
        init.kaiming_normal_(m.weight.data)

#def add_path(path):
    #if path not in sys.path:
        #sys.path.insert(0, path)
#this_dir = osp.dirname(__file__)
#add_path(osp.join(this_dir, '..'))

class unetConv2(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm, n=2, ks=3, stride=1, padding=1):
        super(unetConv2, self).__init__()
        self.n = n
        self.ks = ks
        self.stride = stride
        self.padding = padding
        s = stride
        p = padding
        if is_batchnorm:
            for i in range(1, n + 1):
                conv = nn.Sequential(nn.Conv2d(in_size, out_size, ks, s, p),
                                     nn.BatchNorm2d(out_size),
                                     nn.ReLU(inplace=True), )
                setattr(self, 'conv%d' % i, conv)
                in_size = out_size

        else:
            for i in range(1, n + 1):
                conv = nn.Sequential(nn.Conv2d(in_size, out_size, ks, s, p),
                                     nn.ReLU(inplace=True), )
                setattr(self, 'conv%d' % i, conv)
                in_size = out_size

        # initialise the blocks
        for m in self.children():
            init_weights(m)

    def forward(self, inputs):
        x = inputs
        for i in range(1, self.n + 1):
            conv = getattr(self, 'conv%d' % i)
            x = conv(x)

        return x


class unetUp(nn.Module):
    def __init__(self, in_size, out_size, is_deconv, n_concat=2):
        super(unetUp, self).__init__()
        self.conv = unetConv2(in_size + (n_concat - 2) * out_size, out_size, False)
        if is_deconv:
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2, padding=0)
        else:
            self.up = nn.Sequential(
                nn.UpsamplingBilinear2d(scale_factor=2),
                nn.Conv2d(in_size, out_size, 1))

        # initialise the blocks
        for m in self.children():
            if m.__class__.__name__.find('unetConv2') != -1: continue
            init_weights(m)

    def forward(self, high_feature, *low_feature):
        outputs0 = self.up(high_feature)
        for feature in low_feature:
            outputs0 = torch.cat([outputs0, feature], 1)
        return self.conv(outputs0)


class HCR_Net(nn.Module):

    def __init__(self, in_channels=13, feature_scale=2, is_deconv=True, is_batchnorm=True):
        super(HCR_Net, self).__init__()
        self.in_channels = in_channels
        self.feature_scale = feature_scale
        self.is_deconv = is_deconv
        self.is_batchnorm = is_batchnorm

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]

        # downsampling
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.conv1 = unetConv2(self.in_channels, filters[0], self.is_batchnorm)
        self.conv2 = unetConv2(filters[0], filters[1], self.is_batchnorm)
        self.conv3 = unetConv2(filters[1], filters[2], self.is_batchnorm)
        self.conv4 = unetConv2(filters[2], filters[3], self.is_batchnorm)
        self.center = unetConv2(filters[3], filters[4], self.is_batchnorm)
        # upsampling
        self.up_concat4 = unetUp(filters[4], filters[3], self.is_deconv)
        self.up_concat3 = unetUp(filters[3], filters[2], self.is_deconv)
        self.up_concat2 = unetUp(filters[2], filters[1], self.is_deconv)
        self.up_concat1 = unetUp(filters[1], filters[0], self.is_deconv)
        # final conv (without any concat)-decloud
        self.final_decloud = nn.Conv2d(filters[0], 13, 1)

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m)
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m)

    def forward(self, inputs):
        conv1 = self.conv1(inputs)  # 16*512*512   32*256*256
        maxpool1 = self.maxpool(conv1)  # 16*256*256   32*128*128

        conv2 = self.conv2(maxpool1)  # 32*256*256   64*128*128
        maxpool2 = self.maxpool(conv2)  # 32*128*128   64*64*64

        conv3 = self.conv3(maxpool2)  # 64*128*128     128*64*64
        maxpool3 = self.maxpool(conv3)  # 64*64*64     128*32*32

        conv4 = self.conv4(maxpool3)  # 128*64*64      256*32*32
        maxpool4 = self.maxpool(conv4)  # 128*32*32    256*16*16

        center = self.center(maxpool4)  # 256*32*32    512*16*16

        up4 = self.up_concat4(center, conv4)  # 128*64*64 256*32*32
        up3 = self.up_concat3(up4, conv3)  # 64*128*128 128*64*64
        up2 = self.up_concat2(up3, conv2)  # 32*256*256 64*128*128
        up1 = self.up_concat1(up2, conv1)  # 16*512*512 32*256*256

        decloud = self.final_decloud(up1)   # 13*256*256

        return decloud


if __name__ == '__main__':
    print('#### Test Case ###')
    from torch.autograd import Variable

    #x = Variable(torch.rand(2, 1, 64, 64)).cuda()
    x = torch.randn((1, 15, 256, 256))
    #model = UNet_cloud().cuda()
    model = HCR_Net()
    param = count_param(model)
    print(param)
    #y = model(x)
    #print('Output shape:', y.shape)
    #print('UNet totoal parameters: %.2fM (%d)' % (param / 1e6, param))