import torch
import torch.nn as nn
import model.ops as ops


class Net(nn.Module):
    def __init__(self, **kwargs):
        super(Net, self).__init__()
        
        scale = kwargs.get("scale") #value of scale is scale. 
        multi_scale = kwargs.get("multi_scale") # value of multi_scale is multi_scale in args.
        group = kwargs.get("group", 1) #if valule of group isn't given, group is 1.
        kernel_size = 3 #tcw 201904091123
        kernel_size1 = 1 #tcw 201904091123
        padding1 = 0 #tcw 201904091124
        padding = 1     #tcw201904091123
        features = 64   #tcw201904091124
        groups = 1       #tcw201904091124
        channels = 3
        features1 = 64
        self.sub_mean = ops.MeanShift((0.4488, 0.4371, 0.4040), sub=True)
        self.add_mean = ops.MeanShift((0.4488, 0.4371, 0.4040), sub=False)
        '''
           in_channels, out_channels, kernel_size, stride, padding,dialation, groups,
        '''
        self.conv1_1 = nn.Sequential(nn.Conv2d(in_channels=channels,out_channels=features,kernel_size=(1,3),padding=(0,1),groups=1,bias=False))
        self.conv1_2 = nn.Sequential(nn.Conv2d(in_channels=channels,out_channels=features,kernel_size=(3,1),padding=(1,0),groups=1,bias=False))
        self.conv1_3 = nn.Sequential(nn.Conv2d(in_channels=channels,out_channels=features,kernel_size=(3,3),padding=(1,1),groups=1,bias=False))
        self.conv2_1 = nn.Sequential(nn.Conv2d(in_channels=features,out_channels=features,kernel_size=(1,3),padding=(0,1),groups=1,bias=False))
        self.conv2_2 = nn.Sequential(nn.Conv2d(in_channels=features,out_channels=features,kernel_size=(3,1),padding=(1,0),groups=1,bias=False))
        self.conv2_3 = nn.Sequential(nn.Conv2d(in_channels=features,out_channels=features,kernel_size=(3,3),padding=(1,1),groups=1,bias=False))
        self.conv3_1 = nn.Sequential(nn.Conv2d(in_channels=features,out_channels=features,kernel_size=(1,3),padding=(0,1),groups=1,bias=False))
        self.conv3_2 = nn.Sequential(nn.Conv2d(in_channels=features,out_channels=features,kernel_size=(3,1),padding=(1,0),groups=1,bias=False))
        self.conv3_3 = nn.Sequential(nn.Conv2d(in_channels=features,out_channels=features,kernel_size=(3,3),padding=(1,1),groups=1,bias=False))
        self.conv4_1 = nn.Sequential(nn.Conv2d(in_channels=features,out_channels=features,kernel_size=(1,3),padding=(0,1),groups=1,bias=False))
        self.conv4_2 = nn.Sequential(nn.Conv2d(in_channels=features,out_channels=features,kernel_size=(3,1),padding=(1,0),groups=1,bias=False))
        self.conv4_3 = nn.Sequential(nn.Conv2d(in_channels=features,out_channels=features,kernel_size=(3,3),padding=(1,1),groups=1,bias=False))
        self.conv5_1 = nn.Sequential(nn.Conv2d(in_channels=features,out_channels=features,kernel_size=(1,3),padding=(0,1),groups=1,bias=False))
        self.conv5_2 = nn.Sequential(nn.Conv2d(in_channels=features,out_channels=features,kernel_size=(3,1),padding=(1,0),groups=1,bias=False))
        self.conv5_3 = nn.Sequential(nn.Conv2d(in_channels=features,out_channels=features,kernel_size=(3,3),padding=(1,1),groups=1,bias=False))
        self.conv6_1 = nn.Sequential(nn.Conv2d(in_channels=features,out_channels=features,kernel_size=(1,3),padding=(0,1),groups=1,bias=False))
        self.conv6_2 = nn.Sequential(nn.Conv2d(in_channels=features,out_channels=features,kernel_size=(3,1),padding=(1,0),groups=1,bias=False))
        self.conv6_3 = nn.Sequential(nn.Conv2d(in_channels=features,out_channels=features,kernel_size=(3,3),padding=(1,1),groups=1,bias=False))
        self.conv7_1 = nn.Sequential(nn.Conv2d(in_channels=features,out_channels=features,kernel_size=(1,3),padding=(0,1),groups=1,bias=False))
        self.conv7_2 = nn.Sequential(nn.Conv2d(in_channels=features,out_channels=features,kernel_size=(3,1),padding=(1,0),groups=1,bias=False))
        self.conv7_3 = nn.Sequential(nn.Conv2d(in_channels=features,out_channels=features,kernel_size=(3,3),padding=(1,1),groups=1,bias=False))
        self.conv8_1 = nn.Sequential(nn.Conv2d(in_channels=features,out_channels=features,kernel_size=(1,3),padding=(0,1),groups=1,bias=False))
        self.conv8_2 = nn.Sequential(nn.Conv2d(in_channels=features,out_channels=features,kernel_size=(3,1),padding=(1,0),groups=1,bias=False))
        self.conv8_3 = nn.Sequential(nn.Conv2d(in_channels=features,out_channels=features,kernel_size=(3,3),padding=(1,1),groups=1,bias=False))
        self.conv9_1 = nn.Sequential(nn.Conv2d(in_channels=features,out_channels=features,kernel_size=(1,3),padding=(0,1),groups=1,bias=False))
        self.conv9_2 = nn.Sequential(nn.Conv2d(in_channels=features,out_channels=features,kernel_size=(3,1),padding=(1,0),groups=1,bias=False))
        self.conv9_3 = nn.Sequential(nn.Conv2d(in_channels=features,out_channels=features,kernel_size=(3,3),padding=(1,1),groups=1,bias=False))
        self.conv10_1 = nn.Sequential(nn.Conv2d(in_channels=features,out_channels=features,kernel_size=(1,3),padding=(0,1),groups=1,bias=False))
        self.conv10_2 = nn.Sequential(nn.Conv2d(in_channels=features,out_channels=features,kernel_size=(3,1),padding=(1,0),groups=1,bias=False))
        self.conv10_3 = nn.Sequential(nn.Conv2d(in_channels=features,out_channels=features,kernel_size=(3,3),padding=(1,1),groups=1,bias=False))
        self.conv11_1 = nn.Sequential(nn.Conv2d(in_channels=features,out_channels=features,kernel_size=(1,3),padding=(0,1),groups=1,bias=False))
        self.conv11_2 = nn.Sequential(nn.Conv2d(in_channels=features,out_channels=features,kernel_size=(3,1),padding=(1,0),groups=1,bias=False))
        self.conv11_3 = nn.Sequential(nn.Conv2d(in_channels=features,out_channels=features,kernel_size=(3,3),padding=(1,1),groups=1,bias=False))
        self.conv12_1 = nn.Sequential(nn.Conv2d(in_channels=features,out_channels=features,kernel_size=(1,3),padding=(0,1),groups=1,bias=False))
        self.conv12_2 = nn.Sequential(nn.Conv2d(in_channels=features,out_channels=features,kernel_size=(3,1),padding=(1,0),groups=1,bias=False))
        self.conv12_3 = nn.Sequential(nn.Conv2d(in_channels=features,out_channels=features,kernel_size=(3,3),padding=(1,1),groups=1,bias=False))
        self.conv13_1 = nn.Sequential(nn.Conv2d(in_channels=features,out_channels=features,kernel_size=(1,3),padding=(0,1),groups=1,bias=False))
        self.conv13_2 = nn.Sequential(nn.Conv2d(in_channels=features,out_channels=features,kernel_size=(3,1),padding=(1,0),groups=1,bias=False))
        self.conv13_3 = nn.Sequential(nn.Conv2d(in_channels=features,out_channels=features,kernel_size=(3,3),padding=(1,1),groups=1,bias=False))
        self.conv14_1 = nn.Sequential(nn.Conv2d(in_channels=features,out_channels=features,kernel_size=(1,3),padding=(0,1),groups=1,bias=False))
        self.conv14_2 = nn.Sequential(nn.Conv2d(in_channels=features,out_channels=features,kernel_size=(3,1),padding=(1,0),groups=1,bias=False))
        self.conv14_3 = nn.Sequential(nn.Conv2d(in_channels=features,out_channels=features,kernel_size=(3,3),padding=(1,1),groups=1,bias=False))
        self.conv15_1 = nn.Sequential(nn.Conv2d(in_channels=features,out_channels=features,kernel_size=(1,3),padding=(0,1),groups=1,bias=False))
        self.conv15_2 = nn.Sequential(nn.Conv2d(in_channels=features,out_channels=features,kernel_size=(3,1),padding=(1,0),groups=1,bias=False))
        self.conv15_3 = nn.Sequential(nn.Conv2d(in_channels=features,out_channels=features,kernel_size=(3,3),padding=(1,1),groups=1,bias=False))
        self.conv16_1 = nn.Sequential(nn.Conv2d(in_channels=features,out_channels=features,kernel_size=(1,3),padding=(0,1),groups=1,bias=False))
        self.conv16_2 = nn.Sequential(nn.Conv2d(in_channels=features,out_channels=features,kernel_size=(3,1),padding=(1,0),groups=1,bias=False))
        self.conv16_3 = nn.Sequential(nn.Conv2d(in_channels=features,out_channels=features,kernel_size=(3,3),padding=(1,1),groups=1,bias=False))
        self.conv17_1 = nn.Sequential(nn.Conv2d(in_channels=features,out_channels=features,kernel_size=(1,3),padding=(0,1),groups=1,bias=False))
        self.conv17_2 = nn.Sequential(nn.Conv2d(in_channels=features,out_channels=features,kernel_size=(3,1),padding=(1,0),groups=1,bias=False))
        self.conv17_3 = nn.Sequential(nn.Conv2d(in_channels=features,out_channels=features,kernel_size=(3,3),padding=(1,1),groups=1,bias=False))
        self.conv18_1 = nn.Sequential(nn.Conv2d(in_channels=features,out_channels=features,kernel_size=(3,3),padding=(1,1),groups=1,bias=False),nn.ReLU(inplace=True))
        self.conv18_2 = nn.Sequential(nn.Conv2d(in_channels=features,out_channels=features,kernel_size=(3,3),padding=(1,1),groups=1,bias=False),nn.ReLU(inplace=True))
        self.conv19_1 = nn.Sequential(nn.Conv2d(in_channels=features,out_channels=features,kernel_size=(3,3),padding=(1,1),groups=1,bias=False))
        self.conv19_2 = nn.Sequential(nn.Conv2d(in_channels=features,out_channels=features,kernel_size=(3,3),padding=(1,1),groups=1,bias=False))
        self.conv20 = nn.Sequential(nn.Conv2d(in_channels=features,out_channels=features,kernel_size=(3,3),padding=(1,1),groups=1,bias=False),nn.ReLU(inplace=True))
        self.conv21 = nn.Sequential(nn.Conv2d(in_channels=features,out_channels=features,kernel_size=(3,3),padding=(1,1),groups=1,bias=False),nn.ReLU(inplace=True))
        self.conv22 = nn.Sequential(nn.Conv2d(in_channels=features,out_channels=3,kernel_size=kernel_size,padding=padding,groups=groups,bias=False))
        self.ReLU=nn.ReLU(inplace=True)
        #self.BN = nn.BatchNorm2d(features)
        self.upsample = ops.UpsampleBlock(64, scale=scale, multi_scale=multi_scale,group=1)
    def forward(self, x, scale):
        x0 = self.sub_mean(x)
        x1_1 = self.conv1_1(x0)
        x1_2 = self.conv1_2(x0)
        x1_3 = self.conv1_3(x0)
        x1 = x1_1+x1_2+x1_3
        x1_tcw = self.ReLU(x1)
        x2_1 = self.conv2_1(x1_tcw)
        x2_2 = self.conv2_2(x1_tcw)
        x2_3 = self.conv2_3(x1_tcw)
        x2 = x2_1+x2_2+x2_3
        x2_tcw = self.ReLU(x2)
        x3_1 = self.conv3_1(x2_tcw)
        x3_2 = self.conv3_2(x2_tcw)
        x3_3 = self.conv3_3(x2_tcw)
        x3 = x3_1+x3_2+x3_3
        x3_tcw = self.ReLU(x3)
        x4_1 = self.conv4_1(x3_tcw)
        x4_2 = self.conv4_2(x3_tcw)
        x4_3 = self.conv4_3(x3_tcw)
        x4 = x4_1+x4_2+x4_3
        x4_tcw = self.ReLU(x4)
        x5_1 = self.conv5_1(x4_tcw)
        x5_2 = self.conv5_2(x4_tcw)
        x5_3 = self.conv5_3(x4_tcw)
        x5 = x5_1+x5_2+x5_3
        x5_tcw = self.ReLU(x5)
        x6_1 = self.conv6_1(x5_tcw)
        x6_2 = self.conv6_2(x5_tcw)
        x6_3 = self.conv6_3(x5_tcw)
        x6 = x6_1+x6_2+x6_3
        x6_tcw = self.ReLU(x6)
        x7_1 = self.conv7_1(x6_tcw)
        x7_2 = self.conv7_2(x6_tcw)
        x7_3 = self.conv7_3(x6_tcw)
        x7 = x7_1+x7_2+x7_3
        x7_tcw = self.ReLU(x7)
        x8_1 = self.conv8_1(x7_tcw)
        x8_2 = self.conv8_2(x7_tcw)
        x8_3 = self.conv8_3(x7_tcw)
        x8 = x8_1+x8_2+x8_3
        x8_tcw = self.ReLU(x8)
        x9_1 = self.conv9_1(x8_tcw)
        x9_2 = self.conv9_2(x8_tcw)
        x9_3 = self.conv9_3(x8_tcw)
        x9 = x9_1+x9_2+x9_3
        x9_tcw = self.ReLU(x9)
        x10_1 = self.conv10_1(x9_tcw)
        x10_2 = self.conv10_2(x9_tcw)
        x10_3 = self.conv10_3(x9_tcw)
        x10 = x10_1+x10_2+x10_3
        x10_tcw = self.ReLU(x10)
        x11_1 = self.conv11_1(x10_tcw)
        x11_2 = self.conv11_2(x10_tcw)
        x11_3 = self.conv11_3(x10_tcw)
        x11 = x11_1+x11_2+x11_3
        x11_tcw = self.ReLU(x11)
        x12_1 = self.conv12_1(x11_tcw)
        x12_2 = self.conv12_2(x11_tcw)
        x12_3 = self.conv12_3(x11_tcw)
        x12 = x12_1+x12_2+x12_3
        x12_tcw = self.ReLU(x12)
        x13_1 = self.conv13_1(x12_tcw)
        x13_2 = self.conv13_2(x12_tcw)
        x13_3 = self.conv13_3(x12_tcw)
        x13 = x13_1+x13_2+x13_3
        x13_tcw = self.ReLU(x13)
        x14_1 = self.conv14_1(x13_tcw)
        x14_2 = self.conv14_2(x13_tcw)
        x14_3 = self.conv14_3(x13_tcw)
        x14 = x14_1+x14_2+x14_3
        x14_tcw = self.ReLU(x14)
        x15_1 = self.conv15_1(x14_tcw)
        x15_2 = self.conv15_2(x14_tcw)
        x15_3 = self.conv15_3(x14_tcw)
        x15 = x15_1+x15_2+x15_3
        x15_tcw = self.ReLU(x15)
        x16_1 = self.conv16_1(x15_tcw)
        x16_2 = self.conv16_2(x15_tcw)
        x16_3 = self.conv16_3(x15_tcw)
        x16 = x16_1+x16_2+x16_3
        x16_tcw = self.ReLU(x16)
        x17_1 = self.conv17_1(x16_tcw)
        x17_2 = self.conv17_2(x16_tcw)
        x17_3 = self.conv17_3(x16_tcw)
        x17 = x17_1+x17_2+x17_3
        x17 = x1+x2+x3+x4+x5+x6+x7+x8+x9+x10+x11+x12+x13+x14+x15+x16+x17 #tcw
        x17_tcw = self.ReLU(x17)
        temp = self.upsample(x17_tcw, scale=scale)
        temp1 = self.ReLU(temp)
        x111 = self.upsample(x1_tcw,scale=scale)
        temp2 = self.ReLU(x111)
        x18_1 = self.conv18_1(temp1)
        x18_2 = self.conv18_2(temp2)
        x19_1 = self.conv19_1(x18_1)
        x19_2 = self.conv19_2(x18_2)
        x19 = x19_1 +x19_2
        temp3 = self.ReLU(x19)
        x20 = self.conv20(temp3)
        x21 = self.conv21(x20)
        x22 = self.conv22(x21)
        out = self.add_mean(x22)
        return out
