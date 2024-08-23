import torch
import torch.nn as nn
from model.frangi import get_eig,get_derivative,Frangi,SEmoudle,Frangi2,SEmoudle2,Frangi_one,SEmoudle_channel,get_derivative_conv,get_eig2
import matplotlib.pyplot as plt
import torch.nn.functional as F
from model.frangi import SEmoudle_channel
from torch.nn.init import kaiming_normal_
from frangi2 import DF,CBAMLayer,DF2
from frangi3 import DF as DF_final
from frangi4 import DF as Frangi516



class conv(nn.Module):
    def __init__(self,weight,bias,stride,padding,dilation):
        super(conv, self).__init__()
        self.weight = weight
        self.bias = bias
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
    def forward(self,x):
        out = F.conv2d(x,self.weight,self.bias,self.stride,self.padding,self.dilation)
        return out

class DeepFrangi(nn.Module):  # 3.6 code bug V = self.SE(V)*x bu gai cheng x
    def __init__(self,inchannel,F = True,shrare = False,addchannel=False,scale=9):
        super(DeepFrangi, self).__init__()
        # self.der = get_derivative()
        self.der = get_derivative_conv(inchannel*9)
        # self.eig = get_eig()
        self.eig = get_eig2()
        # self.frangi = Frangi()
        self.frangi = Frangi_one(inchannel,9)
        self.F = F
        if addchannel:
            self.SE = SEmoudle_channel(inchannel,scale=scale)
        else:
            self.SE = SEmoudle(inchannel*scale)
        if shrare:
            self.weight = torch.nn.Parameter(torch.randn(inchannel,inchannel,3,3),requires_grad=True)
            self.bias = None
            torch.nn.init.xavier_normal_(self.weight,gain=nn.init.calculate_gain('relu'))
            self.conv3_3 = conv(weight=self.weight,bias=self.bias,stride=1,padding=1,dilation=1)
            self.conv3_3_2 = conv(weight=self.weight,bias=self.bias,stride=1,padding=2,dilation=2)
            self.conv3_3_3 = conv(weight=self.weight,bias=self.bias,stride=1,padding=3,dilation=3)
            self.conv3_3_4 = conv(weight=self.weight,bias=self.bias,stride=1,padding=4,dilation=4)
            self.conv3_3_5 = conv(weight=self.weight,bias=self.bias,stride=1,padding=5,dilation=5)
            self.conv3_3_6 = conv(weight=self.weight,bias=self.bias,stride=1,padding=6,dilation=6)
            self.conv3_3_7 = conv(weight=self.weight,bias=self.bias,stride=1,padding=7,dilation=7)
            self.conv3_3_8 = conv(weight=self.weight,bias=self.bias,stride=1,padding=8,dilation=8)
            self.conv3_3_9 = conv(weight=self.weight,bias=self.bias,stride=1,padding=9,dilation=9)
        else:
            # self.conv1_1 = nn.Conv2d(inchannel,inchannel,1,1)
            self.conv3_3 = nn.Conv2d(inchannel,inchannel,3,1,1,bias=False)
            self.conv3_3_2 = nn.Conv2d(inchannel,inchannel,3,1,2,2,bias=False)
            self.conv3_3_3 = nn.Conv2d(inchannel, inchannel, 3, 1, 3, 3,bias=False)
            self.conv3_3_4 = nn.Conv2d(inchannel, inchannel, 3, 1, 4, 4,bias=False)
            self.conv3_3_5 = nn.Conv2d(inchannel, inchannel, 3, 1, 5, 5,bias=False)
            self.conv3_3_6 = nn.Conv2d(inchannel, inchannel, 3, 1, 6, 6,bias=False)
            self.conv3_3_7 = nn.Conv2d(inchannel, inchannel, 3, 1, 7, 7,bias=False)
            self.conv3_3_8 = nn.Conv2d(inchannel, inchannel, 3, 1, 8, 8,bias=False)
            self.conv3_3_9 = nn.Conv2d(inchannel,inchannel,3,1,9,9,bias=False)

    def forward(self,x):
        if self.F:
            x1 = self.conv3_3_9(x)
            x2 = self.conv3_3(x)
            x3 = self.conv3_3_2(x)
            x4 = self.conv3_3_3(x)
            x5 = self.conv3_3_4(x)
            x6 = self.conv3_3_5(x)
            x7 = self.conv3_3_6(x)
            x8 = self.conv3_3_7(x)
            x9 = self.conv3_3_8(x)
            x = torch.cat([x1,x2,x3,x4,x5,x6,x7,x8,x9],dim=1)
            mulit = x.cpu().detach()
            Dxx,Dxy,Dyy = self.der(x)
            L1,L2 = self.eig(Dxx,Dxy,Dyy)
            V = self.frangi(L1,L2)
            f_feature = V.cpu().detach()
            V = self.SE(V)
            t = V.cpu().detach()
            # V = self.SE(V)*x
        else:V = x
        return V    #,mulit,f_feature,t

class DeepFrangi_1125(nn.Module):  # 3.6 code bug V = self.SE(V)*x bu gai cheng x   1125 testUnet
    def __init__(self,inchannel,F = True):
        super(DeepFrangi_1125, self).__init__()
        self.der = get_derivative()
        # self.der = get_derivative_conv(inchannel*9)
        self.eig = get_eig()
        # self.eig = get_eig2()
        self.frangi = Frangi()
        # self.frangi = Frangi_one(inchannel,9)
        self.F = F
        self.SE = SEmoudle(inchannel*9)
        self.weight = torch.nn.Parameter(torch.randn(inchannel,inchannel,3,3))
        self.bias = None
        torch.nn.init.xavier_normal_(self.weight,gain=nn.init.calculate_gain('relu'))
        self.conv3_3 = conv(weight=self.weight,bias=self.bias,stride=1,padding=1,dilation=1)
        self.conv3_3_2 = conv(weight=self.weight,bias=self.bias,stride=1,padding=2,dilation=2)
        self.conv3_3_3 = conv(weight=self.weight,bias=self.bias,stride=1,padding=3,dilation=3)
        self.conv3_3_4 = conv(weight=self.weight,bias=self.bias,stride=1,padding=4,dilation=4)
        self.conv3_3_5 = conv(weight=self.weight,bias=self.bias,stride=1,padding=5,dilation=5)
        self.conv3_3_6 = conv(weight=self.weight,bias=self.bias,stride=1,padding=6,dilation=6)
        self.conv3_3_7 = conv(weight=self.weight,bias=self.bias,stride=1,padding=7,dilation=7)
        self.conv3_3_8 = conv(weight=self.weight,bias=self.bias,stride=1,padding=8,dilation=8)
        self.conv3_3_9 = conv(weight=self.weight,bias=self.bias,stride=1,padding=9,dilation=9)
        # self.conv1_1 = nn.Conv2d(inchannel,inchannel,1,1)
        # self.conv3_3 = nn.Conv2d(inchannel,inchannel,3,1,1)
        # self.conv3_3_2 = nn.Conv2d(inchannel,inchannel,3,1,2,2)
        # self.conv3_3_3 = nn.Conv2d(inchannel, inchannel, 3, 1, 3, 3)
        # self.conv3_3_4 = nn.Conv2d(inchannel, inchannel, 3, 1, 4, 4)
        # self.conv3_3_5 = nn.Conv2d(inchannel, inchannel, 3, 1, 5, 5)
        # self.conv3_3_6 = nn.Conv2d(inchannel, inchannel, 3, 1, 6, 6)
        # self.conv3_3_7 = nn.Conv2d(inchannel, inchannel, 3, 1, 7, 7)
        # self.conv3_3_8 = nn.Conv2d(inchannel, inchannel, 3, 1, 8, 8)

    def forward(self,x):
        if self.F:
            x1 = self.conv3_3_9(x)
            x2 = self.conv3_3(x)
            x3 = self.conv3_3_2(x)
            x4 = self.conv3_3_3(x)
            x5 = self.conv3_3_4(x)
            x6 = self.conv3_3_5(x)
            x7 = self.conv3_3_6(x)
            x8 = self.conv3_3_7(x)
            x9 = self.conv3_3_8(x)
            x = torch.cat([x1,x2,x3,x4,x5,x6,x7,x8,x9],dim=1)
            Dxx,Dxy,Dyy = self.der(x)
            L1,L2 = self.eig(Dxx,Dxy,Dyy)
            V = self.frangi(L1,L2)
            V = self.SE(V)
            # V = self.SE(V)*x
        else:V = x
        return V

class DeepFrangi2(nn.Module):  # 3.6 remove weight_share in multi-scale
    def __init__(self,inchannel,F = True):
        super(DeepFrangi2, self).__init__()
        self.der = get_derivative()
        self.eig = get_eig()
        self.frangi = Frangi2(feature_channel=inchannel)
        self.F = F
        self.SE = SEmoudle2(inchannel*9)
        # self.weight = torch.nn.Parameter(torch.randn(inchannel,inchannel,3,3))
        # self.bias = None
        # torch.nn.init.xavier_normal_(self.weight,gain=nn.init.calculate_gain('relu'))
        # self.conv3_3 = conv(weight=self.weight,bias=self.bias,stride=1,padding=1,dilation=1)
        # self.conv3_3_2 = conv(weight=self.weight,bias=self.bias,stride=1,padding=2,dilation=2)
        # self.conv3_3_3 = conv(weight=self.weight,bias=self.bias,stride=1,padding=3,dilation=3)
        # self.conv3_3_4 = conv(weight=self.weight,bias=self.bias,stride=1,padding=4,dilation=4)
        # self.conv3_3_5 = conv(weight=self.weight,bias=self.bias,stride=1,padding=5,dilation=5)
        # self.conv3_3_6 = conv(weight=self.weight,bias=self.bias,stride=1,padding=6,dilation=6)
        # self.conv3_3_7 = conv(weight=self.weight,bias=self.bias,stride=1,padding=7,dilation=7)
        # self.conv3_3_8 = conv(weight=self.weight,bias=self.bias,stride=1,padding=8,dilation=8)
        # self.conv3_3_9 = conv(weight=self.weight,bias=self.bias,stride=1,padding=9,dilation=9)
        self.conv1_1 = nn.Conv2d(inchannel,inchannel,1,1, bias=False)
        self.conv3_3 = nn.Conv2d(inchannel,inchannel,3,1,1, bias=False)
        self.conv3_3_2 = nn.Conv2d(inchannel,inchannel,3,1,2,2, bias=False)
        self.conv3_3_3 = nn.Conv2d(inchannel, inchannel, 3, 1, 3, 3, bias=False)
        self.conv3_3_4 = nn.Conv2d(inchannel, inchannel, 3, 1, 4, 4, bias=False)
        self.conv3_3_5 = nn.Conv2d(inchannel, inchannel, 3, 1, 5, 5, bias=False)
        self.conv3_3_6 = nn.Conv2d(inchannel, inchannel, 3, 1, 6, 6, bias=False)
        self.conv3_3_7 = nn.Conv2d(inchannel, inchannel, 3, 1, 7, 7, bias=False)
        self.conv3_3_8 = nn.Conv2d(inchannel, inchannel, 3, 1, 8, 8, bias=False)

    def forward(self,x):
        if self.F:
            x1 = self.conv1_1(x)
            x2 = self.conv3_3(x)
            x3 = self.conv3_3_2(x)
            x4 = self.conv3_3_3(x)
            x5 = self.conv3_3_4(x)
            x6 = self.conv3_3_5(x)
            x7 = self.conv3_3_6(x)
            x8 = self.conv3_3_7(x)
            x9 = self.conv3_3_8(x)
            x = torch.cat([x1,x2,x3,x4,x5,x6,x7,x8,x9],dim=1)
            Dxx,Dxy,Dyy = self.der(x)
            L1,L2 = self.eig(Dxx,Dxy,Dyy)
            V = self.frangi(L1,L2)
            V = self.SE(V)
        else:V = x
        return V

class DF_share(nn.Module):
    def __init__(self,feature_channel=64,scale=9,share = False,F = True,reduction=2):
        super(DF_share, self).__init__()
        self.F = F
        self.getder = get_derivative_conv(feature_channel*scale)
        self.geteig = get_eig2()
        self.frangi = Frangi_one(feature_channel=feature_channel,sacle=scale)
        self.SE = SEmoudle(feature_channel*scale,reduction=2)
        self.conv = nn.ModuleList()
        self.weight = torch.nn.Parameter(torch.randn(feature_channel, feature_channel, 3, 3))
        self.bias = None
        torch.nn.init.xavier_normal_(self.weight,gain=nn.init.calculate_gain('relu'))
        for i in range(1,scale+1):
            if share:
                self.conv.append(conv(weight=self.weight,bias=self.bias,stride=1,padding=i,dilation=i))
            else:
                self.conv.append(nn.Conv2d(feature_channel, feature_channel, 3, 1, i, i, bias=False))
    def forward(self,x):
        if self.F:
            temp = []
            for i in self.conv:
                temp.append(i(x))
            x = torch.cat(temp,dim=1)
            Dxx,Dxy,Dyy = self.getder(x)
            L1,L2 = self.geteig(Dxx,Dxy,Dyy)
            V = self.frangi(L1,L2)
            V = self.SE(V)
        else:V = x
        return V



class DeepFrangi3(nn.Module):  # 7.4 change multi-scale
    def __init__(self,inchannel,F = True,scale = 9):
        super(DeepFrangi3, self).__init__()
        self.der = get_derivative()
        self.eig = get_eig()
        self.frangi = Frangi2(feature_channel=inchannel,sacle=scale)
        self.F = F
        self.SE = SEmoudle2(inchannel*scale,scale)
        # self.weight = torch.nn.Parameter(torch.randn(inchannel,inchannel,3,3))
        # self.bias = None
        # torch.nn.init.xavier_normal_(self.weight,gain=nn.init.calculate_gain('relu'))
        # self.conv3_3 = conv(weight=self.weight,bias=self.bias,stride=1,padding=1,dilation=1)
        # self.conv3_3_2 = conv(weight=self.weight,bias=self.bias,stride=1,padding=2,dilation=2)
        # self.conv3_3_3 = conv(weight=self.weight,bias=self.bias,stride=1,padding=3,dilation=3)
        # self.conv3_3_4 = conv(weight=self.weight,bias=self.bias,stride=1,padding=4,dilation=4)
        # self.conv3_3_5 = conv(weight=self.weight,bias=self.bias,stride=1,padding=5,dilation=5)
        # self.conv3_3_6 = conv(weight=self.weight,bias=self.bias,stride=1,padding=6,dilation=6)
        # self.conv3_3_7 = conv(weight=self.weight,bias=self.bias,stride=1,padding=7,dilation=7)
        # self.conv3_3_8 = conv(weight=self.weight,bias=self.bias,stride=1,padding=8,dilation=8)
        # self.conv3_3_9 = conv(weight=self.weight,bias=self.bias,stride=1,padding=9,dilation=9)
        self.conv = nn.ModuleList()
        for i in range(scale):
            if i==0:
                self.conv.append(nn.Conv2d(inchannel,inchannel,1,1, bias=False))
            else:
                self.conv.append(nn.Conv2d(inchannel,inchannel,3,1,i,i, bias=False))


    def forward(self,x):
        if self.F:
            temp = []
            for i in self.conv:
                temp.append(i(x))
            x = torch.cat(temp,dim=1)
            Dxx,Dxy,Dyy = self.der(x)
            L1,L2 = self.eig(Dxx,Dxy,Dyy)
            V = self.frangi(L1,L2)
            V = self.SE(V)
        else:V = x
        return V


class DeepFrangi_channel(nn.Module):  # 7.16 Frangi_single,share_paras
    def __init__(self,inchannel,F = True,scale = 9,share = False):
        super(DeepFrangi_channel, self).__init__()
        self.der = get_derivative()
        self.eig = get_eig()
        self.frangi = Frangi_one(feature_channel=inchannel,sacle=scale)
        self.F = F
        self.SE = SEmoudle_channel(inchannel*scale,scale)
        # self.weight = torch.nn.Parameter(torch.randn(inchannel,inchannel,3,3))
        # self.bias = None
        # torch.nn.init.xavier_normal_(self.weight,gain=nn.init.calculate_gain('relu'))
        # self.conv3_3 = conv(weight=self.weight,bias=self.bias,stride=1,padding=1,dilation=1)
        # self.conv3_3_2 = conv(weight=self.weight,bias=self.bias,stride=1,padding=2,dilation=2)
        # self.conv3_3_3 = conv(weight=self.weight,bias=self.bias,stride=1,padding=3,dilation=3)
        # self.conv3_3_4 = conv(weight=self.weight,bias=self.bias,stride=1,padding=4,dilation=4)
        # self.conv3_3_5 = conv(weight=self.weight,bias=self.bias,stride=1,padding=5,dilation=5)
        # self.conv3_3_6 = conv(weight=self.weight,bias=self.bias,stride=1,padding=6,dilation=6)
        # self.conv3_3_7 = conv(weight=self.weight,bias=self.bias,stride=1,padding=7,dilation=7)
        # self.conv3_3_8 = conv(weight=self.weight,bias=self.bias,stride=1,padding=8,dilation=8)
        # self.conv3_3_9 = conv(weight=self.weight,bias=self.bias,stride=1,padding=9,dilation=9)
        self.conv = nn.ModuleList()
        if share:
            self.weight = torch.nn.Parameter(torch.randn(inchannel, inchannel, 3, 3))
            self.bias = None
            torch.nn.init.xavier_normal_(self.weight,gain=nn.init.calculate_gain('relu'))
            for i in range(scale):
                self.conv.append(conv(self.weight,self.bias,stride=1,padding=i+1,dilation=i+1))
        else:
            for i in range(scale):
                if i==0:
                    self.conv.append(nn.Conv2d(inchannel,inchannel,1,1, bias=False))
                else:
                    self.conv.append(nn.Conv2d(inchannel,inchannel,3,1,i,i, bias=False))
    def forward(self,x):
        if self.F:
            temp = []
            for i in self.conv:
                temp.append(i(x))
            x = torch.cat(temp,dim=1)
            Dxx,Dxy,Dyy = self.der(x)
            L1,L2 = self.eig(Dxx,Dxy,Dyy)
            V = self.frangi(L1,L2)
            V = self.SE(V)
        else:V = x
        return V


class DeepFrangi_channel2(nn.Module):  # 7.17 add s**2
    def __init__(self,inchannel,outchannel=None,F = True,scale = 9,share = False,SEnorm = True,mulits=True):
        super(DeepFrangi_channel2, self).__init__()
        if not(isinstance(outchannel,int) and outchannel>0):
            outchannel = inchannel

        self.der = get_derivative()
        self.eig = get_eig()
        self.frangi = Frangi_one(feature_channel=outchannel,sacle=scale)
        self.F = F
        if SEnorm:
            self.SE = SEmoudle_channel(outchannel*scale,scale)
        else:
            self.SE = SEmoudle2(outchannel*scale,scale)
        self.mulit_s = mulits
        # self.weight = torch.nn.Parameter(torch.randn(inchannel,inchannel,3,3))
        # self.bias = None
        # torch.nn.init.xavier_normal_(self.weight,gain=nn.init.calculate_gain('relu'))
        # self.conv3_3 = conv(weight=self.weight,bias=self.bias,stride=1,padding=1,dilation=1)
        # self.conv3_3_2 = conv(weight=self.weight,bias=self.bias,stride=1,padding=2,dilation=2)
        # self.conv3_3_3 = conv(weight=self.weight,bias=self.bias,stride=1,padding=3,dilation=3)
        # self.conv3_3_4 = conv(weight=self.weight,bias=self.bias,stride=1,padding=4,dilation=4)
        # self.conv3_3_5 = conv(weight=self.weight,bias=self.bias,stride=1,padding=5,dilation=5)
        # self.conv3_3_6 = conv(weight=self.weight,bias=self.bias,stride=1,padding=6,dilation=6)
        # self.conv3_3_7 = conv(weight=self.weight,bias=self.bias,stride=1,padding=7,dilation=7)
        # self.conv3_3_8 = conv(weight=self.weight,bias=self.bias,stride=1,padding=8,dilation=8)
        # self.conv3_3_9 = conv(weight=self.weight,bias=self.bias,stride=1,padding=9,dilation=9)
        self.conv = nn.ModuleList()
        if share:
            self.weight = torch.nn.Parameter(torch.randn(outchannel, inchannel, 3, 3))
            self.bias = None
            torch.nn.init.kaiming_normal_(self.weight,nonlinearity='relu')
            for i in range(scale):
                self.conv.append(conv(self.weight,self.bias,stride=1,padding=i+1,dilation=i+1))
        else:
            for i in range(scale):
                if i==0:
                    self.conv.append(nn.Conv2d(inchannel,outchannel,1,1, bias=False))
                else:
                    self.conv.append(nn.Conv2d(inchannel,outchannel,3,1,i,i, bias=False))
    def forward(self,x):
        if self.F:
            tempDxx,tempDxy,tempDyy = [],[],[]
            for index,i in enumerate(self.conv):
                Dxx,Dxy,Dyy = self.der(i(x))
                if self.mulit_s:
                    tempDxx.append(Dxx*(index+1)/3)
                    tempDxy.append(Dxy*(index+1)/3)
                    tempDyy.append(Dyy * (index + 1) / 3)
                else:
                    tempDxx.append(Dxx)
                    tempDxy.append(Dxy)
                    tempDyy.append(Dyy)
            L1,L2 = self.eig(torch.cat(tempDxx,dim=1),torch.cat(tempDxy,dim=1),torch.cat(tempDyy,dim=1))
            V = self.frangi(L1,L2)
            V = self.SE(V)
        else:V = x
        return V
    def normal(self,x):
        return (x-torch.min(x))/(torch.max(x)-torch.min(x))*255



class UF_net(nn.Module):
    def __init__(self,inchannel=[32,64,128,256,512,1024],deepfrangi=False,share = False,addchannel=False):
        super(UF_net, self).__init__()
        if addchannel:
            temp=1
        else:
            temp=9
        self.deepfrangi = deepfrangi
        self.preconv_0 = nn.Sequential(
            nn.Conv2d(3,inchannel[0],1),)
        self.DF =nn.Sequential(DeepFrangi(inchannel[0],deepfrangi,shrare=share,addchannel=addchannel),
            nn.BatchNorm2d(inchannel[0]*temp),
            nn.ReLU(True),
            nn.Conv2d(inchannel[0]*temp,inchannel[0],3,1,1),
            nn.BatchNorm2d(inchannel[0]),
            nn.ReLU(True)
                                     )
        self.down1 = nn.Sequential(
            nn.Conv2d(inchannel[0],inchannel[1],3,1,1),
            nn.ReLU(True),
            nn.Conv2d(inchannel[1],inchannel[1],3,1,1),
            nn.BatchNorm2d(inchannel[1]),
            nn.ReLU(True)
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(inchannel[1], inchannel[2], 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(inchannel[2], inchannel[2], 3, 1, 1),
            nn.BatchNorm2d(inchannel[2]),
            nn.ReLU(True)
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(inchannel[2], inchannel[3], 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(inchannel[3], inchannel[3], 3, 1, 1),
            nn.BatchNorm2d(inchannel[3]),
            nn.ReLU(True)
        )
        self.down4 = nn.Sequential(
            nn.Conv2d(inchannel[3], inchannel[4], 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(inchannel[4], inchannel[4], 3, 1, 1),
            nn.BatchNorm2d(inchannel[4]),
            nn.ReLU(True)
        )
        self.down5 = nn.Sequential(
            nn.Conv2d(inchannel[4], inchannel[5], 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(inchannel[5], inchannel[5], 3, 1, 1),
            nn.BatchNorm2d(inchannel[5]),
            nn.ReLU(True)
        )
        self.pool = nn.MaxPool2d(2,2)
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        self.up4 = nn.Sequential(
            nn.Conv2d(inchannel[5]+inchannel[4],inchannel[4],3,1,1),
            nn.ReLU(True),
            nn.Conv2d(inchannel[4],inchannel[4],3,1,1),
            nn.BatchNorm2d(inchannel[4]),
            nn.ReLU(True)
        )
        self.up3 = nn.Sequential(
            nn.Conv2d(inchannel[4] + inchannel[3], inchannel[3], 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(inchannel[3], inchannel[3], 3, 1, 1),
            nn.BatchNorm2d(inchannel[3]),
            nn.ReLU(True)
        )
        self.up2 = nn.Sequential(
            nn.Conv2d(inchannel[3] + inchannel[2], inchannel[2], 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(inchannel[2], inchannel[2], 3, 1, 1),
            nn.BatchNorm2d(inchannel[2]),
            nn.ReLU(True)
        )
        self.up1 = nn.Sequential(
            nn.Conv2d(inchannel[2] + inchannel[1], inchannel[1], 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(inchannel[1], inchannel[1], 3, 1, 1),
            nn.BatchNorm2d(inchannel[1]),
            nn.ReLU(True)
        )
        self.out = nn.Sequential(
            nn.Conv2d(inchannel[1],1,1,bias=False)
        )
    def forward(self,x):
        x = self.preconv_0(x)
        if self.deepfrangi:
            x = self.DF(x)
        # x = self.preconv_1(x)
        x1_skip = self.down1(x)
        x1 = self.pool(x1_skip)
        x2_skip = self.down2(x1)
        x2 = self.pool(x2_skip)
        x3_skip = self.down3(x2)
        x3 = self.pool(x3_skip)
        x4_skip = self.down4(x3)
        x4 = self.pool(x4_skip)
        x5_skip = self.down5(x4)
        x = self.up4(torch.cat([x4_skip,self.upsample(x5_skip)],dim=1))
        x = self.up3(torch.cat([x3_skip,self.upsample(x)],dim=1))
        x = self.up2(torch.cat([x2_skip, self.upsample(x)], dim=1))
        x = self.up1(torch.cat([x1_skip, self.upsample(x)], dim=1))
        x = self.out(x)
        return x


class UF_net_con_2(nn.Module):
    def __init__(self,inchannel=[32,64,128,256,512,1024],deepfrangi=False,share = False,addchannel=False):
        super(UF_net_con_2, self).__init__()
        if addchannel:
            temp = 1
        else:
            temp=9

        self.deepfrangi = deepfrangi
        self.preconv_0 = nn.Sequential(
            nn.Conv2d(3,inchannel[0],1),)
        self.DF =nn.Sequential(DeepFrangi(inchannel[0],deepfrangi,shrare=share,addchannel=addchannel),
            nn.BatchNorm2d(inchannel[0]*temp),
            nn.ReLU(True),
            nn.Conv2d(inchannel[0]*temp,inchannel[0],3,1,1),
            nn.BatchNorm2d(inchannel[0]),
            nn.ReLU(True)
                                     )
        self.down1 = nn.Sequential(
            nn.Conv2d(inchannel[0]*2,inchannel[1],3,1,1),
            nn.ReLU(True),
            nn.Conv2d(inchannel[1],inchannel[1],3,1,1),
            nn.BatchNorm2d(inchannel[1]),
            nn.ReLU(True)
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(inchannel[1], inchannel[2], 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(inchannel[2], inchannel[2], 3, 1, 1),
            nn.BatchNorm2d(inchannel[2]),
            nn.ReLU(True)
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(inchannel[2], inchannel[3], 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(inchannel[3], inchannel[3], 3, 1, 1),
            nn.BatchNorm2d(inchannel[3]),
            nn.ReLU(True)
        )
        self.down4 = nn.Sequential(
            nn.Conv2d(inchannel[3], inchannel[4], 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(inchannel[4], inchannel[4], 3, 1, 1),
            nn.BatchNorm2d(inchannel[4]),
            nn.ReLU(True)
        )
        self.down5 = nn.Sequential(
            nn.Conv2d(inchannel[4], inchannel[5], 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(inchannel[5], inchannel[5], 3, 1, 1),
            nn.BatchNorm2d(inchannel[5]),
            nn.ReLU(True)
        )
        self.pool = nn.MaxPool2d(2,2)
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        self.up4 = nn.Sequential(
            nn.Conv2d(inchannel[5]+inchannel[4],inchannel[4],3,1,1),
            nn.ReLU(True),
            nn.Conv2d(inchannel[4],inchannel[4],3,1,1),
            nn.BatchNorm2d(inchannel[4]),
            nn.ReLU(True)
        )
        self.up3 = nn.Sequential(
            nn.Conv2d(inchannel[4] + inchannel[3], inchannel[3], 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(inchannel[3], inchannel[3], 3, 1, 1),
            nn.BatchNorm2d(inchannel[3]),
            nn.ReLU(True)
        )
        self.up2 = nn.Sequential(
            nn.Conv2d(inchannel[3] + inchannel[2], inchannel[2], 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(inchannel[2], inchannel[2], 3, 1, 1),
            nn.BatchNorm2d(inchannel[2]),
            nn.ReLU(True)
        )
        self.up1 = nn.Sequential(
            nn.Conv2d(inchannel[2] + inchannel[1], inchannel[1], 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(inchannel[1], inchannel[1], 3, 1, 1),
            nn.BatchNorm2d(inchannel[1]),
            nn.ReLU(True)
        )
        self.out = nn.Sequential(
            nn.Conv2d(inchannel[1],1,1,bias=False)
        )
    def forward(self,x):
        x0 = self.preconv_0(x)
        if self.deepfrangi:
            for i in range(len(self.DF)):
                if i==0:
                    x,mulit,f_feature,attention = self.DF[i](x0)
                else:
                    x = self.DF[i](x)
            df = x.cpu().detach()
            x = torch.cat([x0, x], dim=1)
        else:
            x = x0
        # x = self.preconv_1(x)

        x1_skip = self.down1(x)
        x1 = self.pool(x1_skip)
        x2_skip = self.down2(x1)
        x2 = self.pool(x2_skip)
        x3_skip = self.down3(x2)
        x3 = self.pool(x3_skip)
        x4_skip = self.down4(x3)
        x4 = self.pool(x4_skip)
        x5_skip = self.down5(x4)
        x = self.up4(torch.cat([x4_skip,self.upsample(x5_skip)],dim=1))
        x = self.up3(torch.cat([x3_skip,self.upsample(x)],dim=1))
        x = self.up2(torch.cat([x2_skip, self.upsample(x)], dim=1))
        x = self.up1(torch.cat([x1_skip, self.upsample(x)], dim=1))
        x = self.out(x)
        return x,mulit,f_feature,attention,df


class UF_net_con_54(nn.Module):
    def __init__(self,inchannel=[32,64,128,256,512,1024],inch=3,outch = 32,num=6):
        super(UF_net_con_54, self).__init__()
        self.DF = DF(inch,outch,num)
        self.attention = CBAMLayer(outch*num)
        self.preconv_0 = nn.Sequential(
            nn.Conv2d(3, inchannel[0], 3, 1, 1),
            nn.BatchNorm2d(inchannel[0]),
            nn.ReLU(True))
        self.DFconv =nn.Sequential(
            nn.Conv2d(outch*num,inchannel[0],3,1,1),
            nn.BatchNorm2d(inchannel[0]),
            nn.ReLU(True))
        self.down1 = nn.Sequential(
            nn.Conv2d(inchannel[0]*2,inchannel[1],3,1,1),
            nn.ReLU(True),
            nn.Conv2d(inchannel[1],inchannel[1],3,1,1),
            nn.BatchNorm2d(inchannel[1]),
            nn.ReLU(True)
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(inchannel[1], inchannel[2], 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(inchannel[2], inchannel[2], 3, 1, 1),
            nn.BatchNorm2d(inchannel[2]),
            nn.ReLU(True)
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(inchannel[2], inchannel[3], 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(inchannel[3], inchannel[3], 3, 1, 1),
            nn.BatchNorm2d(inchannel[3]),
            nn.ReLU(True)
        )
        self.down4 = nn.Sequential(
            nn.Conv2d(inchannel[3], inchannel[4], 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(inchannel[4], inchannel[4], 3, 1, 1),
            nn.BatchNorm2d(inchannel[4]),
            nn.ReLU(True)
        )
        self.down5 = nn.Sequential(
            nn.Conv2d(inchannel[4], inchannel[5], 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(inchannel[5], inchannel[5], 3, 1, 1),
            nn.BatchNorm2d(inchannel[5]),
            nn.ReLU(True)
        )
        self.pool = nn.MaxPool2d(2,2)
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        self.up4 = nn.Sequential(
            nn.Conv2d(inchannel[5]+inchannel[4],inchannel[4],3,1,1),
            nn.ReLU(True),
            nn.Conv2d(inchannel[4],inchannel[4],3,1,1),
            nn.BatchNorm2d(inchannel[4]),
            nn.ReLU(True)
        )
        self.up3 = nn.Sequential(
            nn.Conv2d(inchannel[4] + inchannel[3], inchannel[3], 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(inchannel[3], inchannel[3], 3, 1, 1),
            nn.BatchNorm2d(inchannel[3]),
            nn.ReLU(True)
        )
        self.up2 = nn.Sequential(
            nn.Conv2d(inchannel[3] + inchannel[2], inchannel[2], 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(inchannel[2], inchannel[2], 3, 1, 1),
            nn.BatchNorm2d(inchannel[2]),
            nn.ReLU(True)
        )
        self.up1 = nn.Sequential(
            nn.Conv2d(inchannel[2] + inchannel[1], inchannel[1], 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(inchannel[1], inchannel[1], 3, 1, 1),
            nn.BatchNorm2d(inchannel[1]),
            nn.ReLU(True)
        )
        self.out = nn.Sequential(
            nn.Conv2d(inchannel[1],1,1,bias=False)
        )
    def forward(self,x):
        df,rgb,mulit = self.DF(x)
        pre_x = self.preconv_0(x/x.max())
        xattention = self.attention(df)
        xDF = self.DFconv(xattention)
        x = torch.cat([pre_x,xDF],dim=1)

        x1_skip = self.down1(x)
        x1 = self.pool(x1_skip)
        x2_skip = self.down2(x1)
        x2 = self.pool(x2_skip)
        x3_skip = self.down3(x2)
        x3 = self.pool(x3_skip)
        x4_skip = self.down4(x3)
        x4 = self.pool(x4_skip)
        x5_skip = self.down5(x4)
        x = self.up4(torch.cat([x4_skip,self.upsample(x5_skip)],dim=1))
        x = self.up3(torch.cat([x3_skip,self.upsample(x)],dim=1))
        x = self.up2(torch.cat([x2_skip, self.upsample(x)], dim=1))
        x = self.up1(torch.cat([x1_skip, self.upsample(x)], dim=1))
        x = self.out(x)
        return x,df.cpu().detach(),rgb,mulit,xattention.cpu().detach(),xDF.cpu().detach()


class UF_net_con_54_2(nn.Module):
    def __init__(self,inchannel=[32,64,128,256,512,1024],inch=3,outch = 32,num=6):
        super(UF_net_con_54_2, self).__init__()
        self.DF = DF2(num)
        self.attention = CBAMLayer(num,reduction=2)
        self.preconv_0 = nn.Sequential(
            nn.Conv2d(3, inchannel[0], 3, 1, 1),
            nn.BatchNorm2d(inchannel[0]),
            nn.ReLU(True))
        # self.DFconv =nn.Sequential(
        #     nn.Conv2d(outch*num,inchannel[0],3,1,1),
        #     nn.BatchNorm2d(inchannel[0]),
        #     nn.ReLU(True))
        self.down1 = nn.Sequential(
            nn.Conv2d(inchannel[0]+num,inchannel[1],3,1,1),
            nn.ReLU(True),
            nn.Conv2d(inchannel[1],inchannel[1],3,1,1),
            nn.BatchNorm2d(inchannel[1]),
            nn.ReLU(True)
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(inchannel[1], inchannel[2], 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(inchannel[2], inchannel[2], 3, 1, 1),
            nn.BatchNorm2d(inchannel[2]),
            nn.ReLU(True)
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(inchannel[2], inchannel[3], 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(inchannel[3], inchannel[3], 3, 1, 1),
            nn.BatchNorm2d(inchannel[3]),
            nn.ReLU(True)
        )
        self.down4 = nn.Sequential(
            nn.Conv2d(inchannel[3], inchannel[4], 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(inchannel[4], inchannel[4], 3, 1, 1),
            nn.BatchNorm2d(inchannel[4]),
            nn.ReLU(True)
        )
        self.down5 = nn.Sequential(
            nn.Conv2d(inchannel[4], inchannel[5], 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(inchannel[5], inchannel[5], 3, 1, 1),
            nn.BatchNorm2d(inchannel[5]),
            nn.ReLU(True)
        )
        self.pool = nn.MaxPool2d(2,2)
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        self.up4 = nn.Sequential(
            nn.Conv2d(inchannel[5]+inchannel[4],inchannel[4],3,1,1),
            nn.ReLU(True),
            nn.Conv2d(inchannel[4],inchannel[4],3,1,1),
            nn.BatchNorm2d(inchannel[4]),
            nn.ReLU(True)
        )
        self.up3 = nn.Sequential(
            nn.Conv2d(inchannel[4] + inchannel[3], inchannel[3], 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(inchannel[3], inchannel[3], 3, 1, 1),
            nn.BatchNorm2d(inchannel[3]),
            nn.ReLU(True)
        )
        self.up2 = nn.Sequential(
            nn.Conv2d(inchannel[3] + inchannel[2], inchannel[2], 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(inchannel[2], inchannel[2], 3, 1, 1),
            nn.BatchNorm2d(inchannel[2]),
            nn.ReLU(True)
        )
        self.up1 = nn.Sequential(
            nn.Conv2d(inchannel[2] + inchannel[1], inchannel[1], 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(inchannel[1], inchannel[1], 3, 1, 1),
            nn.BatchNorm2d(inchannel[1]),
            nn.ReLU(True)
        )
        self.out = nn.Sequential(
            nn.Conv2d(inchannel[1],1,1,bias=False)
        )
    def forward(self,x):
        df,rgb,mulit = self.DF(x)
        pre_x = self.preconv_0(x/x.max())
        xattention = self.attention(df)
        # xDF = self.DFconv(xattention)
        x = torch.cat([pre_x,xattention],dim=1)

        x1_skip = self.down1(x)
        x1 = self.pool(x1_skip)
        x2_skip = self.down2(x1)
        x2 = self.pool(x2_skip)
        x3_skip = self.down3(x2)
        x3 = self.pool(x3_skip)
        x4_skip = self.down4(x3)
        x4 = self.pool(x4_skip)
        x5_skip = self.down5(x4)
        x = self.up4(torch.cat([x4_skip,self.upsample(x5_skip)],dim=1))
        x = self.up3(torch.cat([x3_skip,self.upsample(x)],dim=1))
        x = self.up2(torch.cat([x2_skip, self.upsample(x)], dim=1))
        x = self.up1(torch.cat([x1_skip, self.upsample(x)], dim=1))
        x = self.out(x)
        return x,df.cpu().detach(),rgb,mulit,xattention.cpu().detach()


class UF_net_con(nn.Module):
    def __init__(self,inchannel=[32,64,128,256,512,1024],deepfrangi=True):
        super(UF_net_con, self).__init__()
        self.deepfrangi = deepfrangi
        self.preconv_0 = nn.Sequential(
            nn.Conv2d(3,inchannel[0],1),)
        self.DF =nn.Sequential(DeepFrangi(inchannel[0],deepfrangi),
            nn.BatchNorm2d(inchannel[0]*9),
            nn.ReLU(True),
            nn.Conv2d(inchannel[0]*9,inchannel[0],3,1,1),
            nn.BatchNorm2d(inchannel[0]),
            nn.ReLU(True)
                                     )
        self.down1_con = nn.Sequential(
            nn.Conv2d(inchannel[0]*2,inchannel[1],3,1,1),
            nn.ReLU(True),
            nn.Conv2d(inchannel[1],inchannel[1],3,1,1),
            nn.BatchNorm2d(inchannel[1]),
            nn.ReLU(True)
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(inchannel[1], inchannel[2], 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(inchannel[2], inchannel[2], 3, 1, 1),
            nn.BatchNorm2d(inchannel[2]),
            nn.ReLU(True)
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(inchannel[2], inchannel[3], 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(inchannel[3], inchannel[3], 3, 1, 1),
            nn.BatchNorm2d(inchannel[3]),
            nn.ReLU(True)
        )
        self.down4 = nn.Sequential(
            nn.Conv2d(inchannel[3], inchannel[4], 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(inchannel[4], inchannel[4], 3, 1, 1),
            nn.BatchNorm2d(inchannel[4]),
            nn.ReLU(True)
        )
        self.down5 = nn.Sequential(
            nn.Conv2d(inchannel[4], inchannel[5], 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(inchannel[5], inchannel[5], 3, 1, 1),
            nn.BatchNorm2d(inchannel[5]),
            nn.ReLU(True)
        )
        self.pool = nn.MaxPool2d(2,2)
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        self.up4 = nn.Sequential(
            nn.Conv2d(inchannel[5]+inchannel[4],inchannel[4],3,1,1),
            nn.ReLU(True),
            nn.Conv2d(inchannel[4],inchannel[4],3,1,1),
            nn.BatchNorm2d(inchannel[4]),
            nn.ReLU(True)
        )
        self.up3 = nn.Sequential(
            nn.Conv2d(inchannel[4] + inchannel[3], inchannel[3], 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(inchannel[3], inchannel[3], 3, 1, 1),
            nn.BatchNorm2d(inchannel[3]),
            nn.ReLU(True)
        )
        self.up2 = nn.Sequential(
            nn.Conv2d(inchannel[3] + inchannel[2], inchannel[2], 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(inchannel[2], inchannel[2], 3, 1, 1),
            nn.BatchNorm2d(inchannel[2]),
            nn.ReLU(True)
        )
        self.up1 = nn.Sequential(
            nn.Conv2d(inchannel[2] + inchannel[1], inchannel[1], 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(inchannel[1], inchannel[1], 3, 1, 1),
            nn.BatchNorm2d(inchannel[1]),
            nn.ReLU(True)
        )
        self.out = nn.Sequential(
            nn.Conv2d(inchannel[1],1,1,bias=False)
        )
    def forward(self,x):
        x = self.preconv_0(x)
        if self.deepfrangi:
            vx = self.DF(x)
        # x = self.preconv_1(x)
        x = torch.cat([vx,x],dim=1)
        x1_skip = self.down1_con(x)
        x1 = self.pool(x1_skip)
        x2_skip = self.down2(x1)
        x2 = self.pool(x2_skip)
        x3_skip = self.down3(x2)
        x3 = self.pool(x3_skip)
        x4_skip = self.down4(x3)
        x4 = self.pool(x4_skip)
        x5_skip = self.down5(x4)
        x = self.up4(torch.cat([x4_skip,self.upsample(x5_skip)],dim=1))
        x = self.up3(torch.cat([x3_skip,self.upsample(x)],dim=1))
        x = self.up2(torch.cat([x2_skip, self.upsample(x)], dim=1))
        x = self.up1(torch.cat([x1_skip, self.upsample(x)], dim=1))
        x = self.out(x)
        return x

class UF_net_1125(nn.Module): # 1125 testUnet
    def __init__(self,inchannel=[32,64,128,256,512,1024],deepfrangi=False):
        super(UF_net_1125, self).__init__()
        self.deepfrangi = deepfrangi
        self.preconv_0 = nn.Sequential(
            nn.Conv2d(3,inchannel[0],1),)
        self.DF =nn.Sequential(DeepFrangi_1125(inchannel[0],deepfrangi),
            nn.BatchNorm2d(inchannel[0]*9),
            nn.ReLU(True),
            nn.Conv2d(inchannel[0]*9,inchannel[0],3,1,1),
            nn.BatchNorm2d(inchannel[0]),
            nn.ReLU(True)
                                     )
        self.down1 = nn.Sequential(
            nn.Conv2d(inchannel[0],inchannel[1],3,1,1),
            nn.ReLU(True),
            nn.Conv2d(inchannel[1],inchannel[1],3,1,1),
            nn.BatchNorm2d(inchannel[1]),
            nn.ReLU(True)
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(inchannel[1], inchannel[2], 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(inchannel[2], inchannel[2], 3, 1, 1),
            nn.BatchNorm2d(inchannel[2]),
            nn.ReLU(True)
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(inchannel[2], inchannel[3], 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(inchannel[3], inchannel[3], 3, 1, 1),
            nn.BatchNorm2d(inchannel[3]),
            nn.ReLU(True)
        )
        self.down4 = nn.Sequential(
            nn.Conv2d(inchannel[3], inchannel[4], 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(inchannel[4], inchannel[4], 3, 1, 1),
            nn.BatchNorm2d(inchannel[4]),
            nn.ReLU(True)
        )
        self.down5 = nn.Sequential(
            nn.Conv2d(inchannel[4], inchannel[5], 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(inchannel[5], inchannel[5], 3, 1, 1),
            nn.BatchNorm2d(inchannel[5]),
            nn.ReLU(True)
        )
        self.pool = nn.MaxPool2d(2,2)
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        self.up4 = nn.Sequential(
            nn.Conv2d(inchannel[5]+inchannel[4],inchannel[4],3,1,1),
            nn.ReLU(True),
            nn.Conv2d(inchannel[4],inchannel[4],3,1,1),
            nn.BatchNorm2d(inchannel[4]),
            nn.ReLU(True)
        )
        self.up3 = nn.Sequential(
            nn.Conv2d(inchannel[4] + inchannel[3], inchannel[3], 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(inchannel[3], inchannel[3], 3, 1, 1),
            nn.BatchNorm2d(inchannel[3]),
            nn.ReLU(True)
        )
        self.up2 = nn.Sequential(
            nn.Conv2d(inchannel[3] + inchannel[2], inchannel[2], 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(inchannel[2], inchannel[2], 3, 1, 1),
            nn.BatchNorm2d(inchannel[2]),
            nn.ReLU(True)
        )
        self.up1 = nn.Sequential(
            nn.Conv2d(inchannel[2] + inchannel[1], inchannel[1], 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(inchannel[1], inchannel[1], 3, 1, 1),
            nn.BatchNorm2d(inchannel[1]),
            nn.ReLU(True)
        )
        self.out = nn.Sequential(
            nn.Conv2d(inchannel[1],1,1,bias=False)
        )
    def forward(self,x):
        x = self.preconv_0(x)
        if self.deepfrangi:
            x = self.DF(x)
        # x = self.preconv_1(x)
        x1_skip = self.down1(x)
        x1 = self.pool(x1_skip)
        x2_skip = self.down2(x1)
        x2 = self.pool(x2_skip)
        x3_skip = self.down3(x2)
        x3 = self.pool(x3_skip)
        x4_skip = self.down4(x3)
        x4 = self.pool(x4_skip)
        x5_skip = self.down5(x4)
        x = self.up4(torch.cat([x4_skip,self.upsample(x5_skip)],dim=1))
        x = self.up3(torch.cat([x3_skip,self.upsample(x)],dim=1))
        x = self.up2(torch.cat([x2_skip, self.upsample(x)], dim=1))
        x = self.up1(torch.cat([x1_skip, self.upsample(x)], dim=1))
        x = self.out(x)
        return x
class UF_net2(nn.Module):
    def __init__(self,inchannel=[32,64,128,256,512,1024],deepfrangi=False):
        super(UF_net2, self).__init__()
        self.deepfrangi = deepfrangi
        self.preconv_0 = nn.Sequential(
            nn.Conv2d(3,inchannel[0],1),)
        self.DF =nn.Sequential(DeepFrangi2(inchannel[0],deepfrangi),
            nn.BatchNorm2d(inchannel[0]),
            nn.ReLU(True)
                                     )
        self.down1 = nn.Sequential(
            nn.Conv2d(inchannel[0],inchannel[1],3,1,1),
            nn.ReLU(True),
            nn.Conv2d(inchannel[1],inchannel[1],3,1,1),
            nn.BatchNorm2d(inchannel[1]),
            nn.ReLU(True)
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(inchannel[1], inchannel[2], 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(inchannel[2], inchannel[2], 3, 1, 1),
            nn.BatchNorm2d(inchannel[2]),
            nn.ReLU(True)
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(inchannel[2], inchannel[3], 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(inchannel[3], inchannel[3], 3, 1, 1),
            nn.BatchNorm2d(inchannel[3]),
            nn.ReLU(True)
        )
        self.down4 = nn.Sequential(
            nn.Conv2d(inchannel[3], inchannel[4], 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(inchannel[4], inchannel[4], 3, 1, 1),
            nn.BatchNorm2d(inchannel[4]),
            nn.ReLU(True)
        )
        self.down5 = nn.Sequential(
            nn.Conv2d(inchannel[4], inchannel[5], 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(inchannel[5], inchannel[5], 3, 1, 1),
            nn.BatchNorm2d(inchannel[5]),
            nn.ReLU(True)
        )
        self.pool = nn.MaxPool2d(2,2)
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        self.up4 = nn.Sequential(
            nn.Conv2d(inchannel[5]+inchannel[4],inchannel[4],3,1,1),
            nn.ReLU(True),
            nn.Conv2d(inchannel[4],inchannel[4],3,1,1),
            nn.BatchNorm2d(inchannel[4]),
            nn.ReLU(True)
        )
        self.up3 = nn.Sequential(
            nn.Conv2d(inchannel[4] + inchannel[3], inchannel[3], 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(inchannel[3], inchannel[3], 3, 1, 1),
            nn.BatchNorm2d(inchannel[3]),
            nn.ReLU(True)
        )
        self.up2 = nn.Sequential(
            nn.Conv2d(inchannel[3] + inchannel[2], inchannel[2], 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(inchannel[2], inchannel[2], 3, 1, 1),
            nn.BatchNorm2d(inchannel[2]),
            nn.ReLU(True)
        )
        self.up1 = nn.Sequential(
            nn.Conv2d(inchannel[2] + inchannel[1], inchannel[1], 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(inchannel[1], inchannel[1], 3, 1, 1),
            nn.BatchNorm2d(inchannel[1]),
            nn.ReLU(True)
        )
        self.out = nn.Sequential(
            nn.Conv2d(inchannel[1],1,1,bias=False)
        )
    def forward(self,x):
        x = self.preconv_0(x)
        if self.deepfrangi:
            x = self.DF(x)
        # x = self.preconv_1(x)
        x1_skip = self.down1(x)
        x1 = self.pool(x1_skip)
        x2_skip = self.down2(x1)
        x2 = self.pool(x2_skip)
        x3_skip = self.down3(x2)
        x3 = self.pool(x3_skip)
        x4_skip = self.down4(x3)
        x4 = self.pool(x4_skip)
        x5_skip = self.down5(x4)
        x = self.up4(torch.cat([x4_skip,self.upsample(x5_skip)],dim=1))
        x = self.up3(torch.cat([x3_skip,self.upsample(x)],dim=1))
        x = self.up2(torch.cat([x2_skip, self.upsample(x)], dim=1))
        x = self.up1(torch.cat([x1_skip, self.upsample(x)], dim=1))
        x = self.out(x)
        return x
class F_net(nn.Module):
    def __init__(self, inchannel=[8, 64, 32, 64, 32, 1024], deepfrangi=False):
    # def __init__(self,inchannel=[32,64,128,64,32,1024],deepfrangi=False):
        super(F_net, self).__init__()
        self.deepfrangi = deepfrangi
        self.preconv_0 = nn.Sequential(
            nn.Conv2d(3,inchannel[0],1),)
        self.DF =nn.Sequential(DeepFrangi(inchannel[0],deepfrangi),
            nn.BatchNorm2d(inchannel[0]*9),
            nn.ReLU(True),
            nn.Conv2d(inchannel[0]*9,inchannel[0],3,1,1),
            nn.BatchNorm2d(inchannel[0]),
            nn.ReLU(True)
                                     )
        self.down1 = nn.Sequential(
            nn.Conv2d(inchannel[0],inchannel[1],3,1,1),
            nn.ReLU(True),
            nn.Conv2d(inchannel[1],inchannel[1],3,1,1),
            nn.BatchNorm2d(inchannel[1]),
            nn.ReLU(True)
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(inchannel[1], inchannel[2], 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(inchannel[2], inchannel[2], 3, 1, 1),
            nn.BatchNorm2d(inchannel[2]),
            nn.ReLU(True)
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(inchannel[2], inchannel[3], 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(inchannel[3], inchannel[3], 3, 1, 1),
            nn.BatchNorm2d(inchannel[3]),
            nn.ReLU(True)
        )
        self.down4 = nn.Sequential(
            nn.Conv2d(inchannel[3], inchannel[4], 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(inchannel[4], inchannel[4], 3, 1, 1),
            nn.BatchNorm2d(inchannel[4]),
            nn.ReLU(True)
        )
        self.out = nn.Sequential(
            nn.Conv2d(inchannel[4],1,1,bias=False)
        )
    def forward(self,x):
        x = self.preconv_0(x)
        if self.deepfrangi:
            x = self.DF(x)
        x= self.down1(x)
        x = self.down2(x)
        # x = self.down3(x)
        # x = self.down4(x)
        x = self.out(x)
        return x
class DU_UF_net(nn.Module):
    def __init__(self,inchannel=[32,64,128,256,512,1024],deepfrangi=False):
        super(DU_UF_net, self).__init__()
        self.deepfrangi = deepfrangi
        self.preconv_0 = nn.Sequential(
            nn.Conv2d(3,inchannel[0],1),)
        self.DF =nn.Sequential(DeepFrangi(inchannel[0],deepfrangi),
            nn.BatchNorm2d(inchannel[0]*9),
            nn.ReLU(True),
            nn.Conv2d(inchannel[0]*9,inchannel[0],3,1,1),
            nn.BatchNorm2d(inchannel[0]),
            nn.ReLU(True)
                                     )
        self.down1 = nn.Sequential(
            nn.Conv2d(inchannel[0]*2,inchannel[1],3,1,1),
            nn.ReLU(True),
            nn.Conv2d(inchannel[1],inchannel[1],3,1,1),
            nn.BatchNorm2d(inchannel[1]),
            nn.ReLU(True)
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(inchannel[1], inchannel[2], 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(inchannel[2], inchannel[2], 3, 1, 1),
            nn.BatchNorm2d(inchannel[2]),
            nn.ReLU(True)
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(inchannel[2], inchannel[3], 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(inchannel[3], inchannel[3], 3, 1, 1),
            nn.BatchNorm2d(inchannel[3]),
            nn.ReLU(True)
        )
        self.down4 = nn.Sequential(
            nn.Conv2d(inchannel[3], inchannel[4], 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(inchannel[4], inchannel[4], 3, 1, 1),
            nn.BatchNorm2d(inchannel[4]),
            nn.ReLU(True)
        )
        self.down5 = nn.Sequential(
            nn.Conv2d(inchannel[4], inchannel[5], 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(inchannel[5], inchannel[5], 3, 1, 1),
            nn.BatchNorm2d(inchannel[5]),
            nn.ReLU(True)
        )
        self.pool = nn.MaxPool2d(2,2)
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        self.up4 = nn.Sequential(
            nn.Conv2d(inchannel[5]+inchannel[4],inchannel[4],3,1,1),
            nn.ReLU(True),
            nn.Conv2d(inchannel[4],inchannel[4],3,1,1),
            nn.BatchNorm2d(inchannel[4]),
            nn.ReLU(True)
        )
        self.up3 = nn.Sequential(
            nn.Conv2d(inchannel[4] + inchannel[3], inchannel[3], 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(inchannel[3], inchannel[3], 3, 1, 1),
            nn.BatchNorm2d(inchannel[3]),
            nn.ReLU(True)
        )
        self.up2 = nn.Sequential(
            nn.Conv2d(inchannel[3] + inchannel[2], inchannel[2], 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(inchannel[2], inchannel[2], 3, 1, 1),
            nn.BatchNorm2d(inchannel[2]),
            nn.ReLU(True)
        )
        self.up1 = nn.Sequential(
            nn.Conv2d(inchannel[2] + inchannel[1], inchannel[1], 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(inchannel[1], inchannel[1], 3, 1, 1),
            nn.BatchNorm2d(inchannel[1]),
            nn.ReLU(True)
        )
        self.out = nn.Sequential(
            nn.Conv2d(inchannel[1],1,1,bias=False)
        )
    def forward(self,x):
        x1 = self.preconv_0(x)
        if self.deepfrangi:
            x = self.DF(x1)
        # x = self.preconv_1(x)
        x1_skip = self.down1(torch.cat([x,x1],dim=1))
        x1 = self.pool(x1_skip)
        x2_skip = self.down2(x1)
        x2 = self.pool(x2_skip)
        x3_skip = self.down3(x2)
        x3 = self.pool(x3_skip)
        x4_skip = self.down4(x3)
        x4 = self.pool(x4_skip)
        x5_skip = self.down5(x4)
        x = self.up4(torch.cat([x4_skip,self.upsample(x5_skip)],dim=1))
        x = self.up3(torch.cat([x3_skip,self.upsample(x)],dim=1))
        x = self.up2(torch.cat([x2_skip, self.upsample(x)], dim=1))
        x = self.up1(torch.cat([x1_skip, self.upsample(x)], dim=1))
        x = self.out(x)
        return x



class UF_net_all(nn.Module):
    def __init__(self,inchannel=[32,64,128,256,512,1024],deepfrangi=False,share = False,concat = False):
        super(UF_net_all, self).__init__()
        self.deepfrangi = deepfrangi
        self.preconv_0 = nn.Sequential(
            nn.Conv2d(3,inchannel[0],1),)
        self.DF =nn.Sequential(DF_share(inchannel[0],scale=9,F=deepfrangi,share=share),
            nn.BatchNorm2d(inchannel[0]*9),
            nn.ReLU(True),
            nn.Conv2d(inchannel[0]*9,inchannel[0],3,1,1),
            nn.BatchNorm2d(inchannel[0]),
            nn.ReLU(True)
                                     )
        self.concat = concat
        if concat:
            self.down1 = nn.Sequential(
                nn.Conv2d(inchannel[0]+inchannel[0],inchannel[1],3,1,1),
                nn.ReLU(True),
                nn.Conv2d(inchannel[1],inchannel[1],3,1,1),
                nn.BatchNorm2d(inchannel[1]),
                nn.ReLU(True)
            )
        else:
            self.down1 = nn.Sequential(
                nn.Conv2d(inchannel[0], inchannel[1], 3, 1, 1),
                nn.ReLU(True),
                nn.Conv2d(inchannel[1], inchannel[1], 3, 1, 1),
                nn.BatchNorm2d(inchannel[1]),
                nn.ReLU(True)
            )
        self.down2 = nn.Sequential(
            nn.Conv2d(inchannel[1], inchannel[2], 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(inchannel[2], inchannel[2], 3, 1, 1),
            nn.BatchNorm2d(inchannel[2]),
            nn.ReLU(True)
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(inchannel[2], inchannel[3], 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(inchannel[3], inchannel[3], 3, 1, 1),
            nn.BatchNorm2d(inchannel[3]),
            nn.ReLU(True)
        )
        self.down4 = nn.Sequential(
            nn.Conv2d(inchannel[3], inchannel[4], 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(inchannel[4], inchannel[4], 3, 1, 1),
            nn.BatchNorm2d(inchannel[4]),
            nn.ReLU(True)
        )
        self.down5 = nn.Sequential(
            nn.Conv2d(inchannel[4], inchannel[5], 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(inchannel[5], inchannel[5], 3, 1, 1),
            nn.BatchNorm2d(inchannel[5]),
            nn.ReLU(True)
        )
        self.pool = nn.MaxPool2d(2,2)
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        self.up4 = nn.Sequential(
            nn.Conv2d(inchannel[5]+inchannel[4],inchannel[4],3,1,1),
            nn.ReLU(True),
            nn.Conv2d(inchannel[4],inchannel[4],3,1,1),
            nn.BatchNorm2d(inchannel[4]),
            nn.ReLU(True)
        )
        self.up3 = nn.Sequential(
            nn.Conv2d(inchannel[4] + inchannel[3], inchannel[3], 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(inchannel[3], inchannel[3], 3, 1, 1),
            nn.BatchNorm2d(inchannel[3]),
            nn.ReLU(True)
        )
        self.up2 = nn.Sequential(
            nn.Conv2d(inchannel[3] + inchannel[2], inchannel[2], 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(inchannel[2], inchannel[2], 3, 1, 1),
            nn.BatchNorm2d(inchannel[2]),
            nn.ReLU(True)
        )
        self.up1 = nn.Sequential(
            nn.Conv2d(inchannel[2] + inchannel[1], inchannel[1], 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(inchannel[1], inchannel[1], 3, 1, 1),
            nn.BatchNorm2d(inchannel[1]),
            nn.ReLU(True)
        )
        self.out = nn.Sequential(
            nn.Conv2d(inchannel[1],1,1,bias=False)
        )
    def forward(self,x):
        x = self.preconv_0(x)
        if self.deepfrangi:
            xdf = self.DF(x)
        # x = self.preconv_1(x)
        if self.concat:
            x = torch.cat([x,xdf],dim=1)
        else:
            x = xdf
        x1_skip = self.down1(x)
        x1 = self.pool(x1_skip)
        x2_skip = self.down2(x1)
        x2 = self.pool(x2_skip)
        x3_skip = self.down3(x2)
        x3 = self.pool(x3_skip)
        x4_skip = self.down4(x3)
        x4 = self.pool(x4_skip)
        x5_skip = self.down5(x4)
        x = self.up4(torch.cat([x4_skip,self.upsample(x5_skip)],dim=1))
        x = self.up3(torch.cat([x3_skip,self.upsample(x)],dim=1))
        x = self.up2(torch.cat([x2_skip, self.upsample(x)], dim=1))
        x = self.up1(torch.cat([x1_skip, self.upsample(x)], dim=1))
        x = self.out(x)
        return x





class UF_net_DF(nn.Module):
    def __init__(self,inchannel=[32,64,128,256,512,1024],inch=3,outch = 32,num=6,usedf = True):
        super(UF_net_DF, self).__init__()
        self.use = usedf
        self.DF = DF_final(inch,outch,num)
        self.attention = CBAMLayer(outch*num,reduction=2)
        self.preconv_0 = nn.Sequential(
            nn.Conv2d(3, inchannel[0], 3, 1, 1),
            nn.BatchNorm2d(inchannel[0]),
            nn.ReLU(True))
        self.DFconv =nn.Sequential(
            nn.Conv2d(outch*num,inchannel[0],3,1,1),
            nn.BatchNorm2d(inchannel[0]),
            nn.ReLU(True))
        if usedf:
            self.down1 = nn.Sequential(
                nn.Conv2d(inchannel[0]*2,inchannel[1],3,1,1),
                nn.ReLU(True),
                nn.Conv2d(inchannel[1],inchannel[1],3,1,1),
                nn.BatchNorm2d(inchannel[1]),
                nn.ReLU(True)
            )
        else:
            self.down1 = nn.Sequential(
                nn.Conv2d(inchannel[0], inchannel[1], 3, 1, 1),
                nn.ReLU(True),
                nn.Conv2d(inchannel[1], inchannel[1], 3, 1, 1),
                nn.BatchNorm2d(inchannel[1]),
                nn.ReLU(True)
            )
        self.down2 = nn.Sequential(
            nn.Conv2d(inchannel[1], inchannel[2], 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(inchannel[2], inchannel[2], 3, 1, 1),
            nn.BatchNorm2d(inchannel[2]),
            nn.ReLU(True)
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(inchannel[2], inchannel[3], 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(inchannel[3], inchannel[3], 3, 1, 1),
            nn.BatchNorm2d(inchannel[3]),
            nn.ReLU(True)
        )
        self.down4 = nn.Sequential(
            nn.Conv2d(inchannel[3], inchannel[4], 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(inchannel[4], inchannel[4], 3, 1, 1),
            nn.BatchNorm2d(inchannel[4]),
            nn.ReLU(True)
        )
        self.down5 = nn.Sequential(
            nn.Conv2d(inchannel[4], inchannel[5], 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(inchannel[5], inchannel[5], 3, 1, 1),
            nn.BatchNorm2d(inchannel[5]),
            nn.ReLU(True)
        )
        self.pool = nn.MaxPool2d(2,2)
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        self.up4 = nn.Sequential(
            nn.Conv2d(inchannel[5]+inchannel[4],inchannel[4],3,1,1),
            nn.ReLU(True),
            nn.Conv2d(inchannel[4],inchannel[4],3,1,1),
            nn.BatchNorm2d(inchannel[4]),
            nn.ReLU(True)
        )
        self.up3 = nn.Sequential(
            nn.Conv2d(inchannel[4] + inchannel[3], inchannel[3], 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(inchannel[3], inchannel[3], 3, 1, 1),
            nn.BatchNorm2d(inchannel[3]),
            nn.ReLU(True)
        )
        self.up2 = nn.Sequential(
            nn.Conv2d(inchannel[3] + inchannel[2], inchannel[2], 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(inchannel[2], inchannel[2], 3, 1, 1),
            nn.BatchNorm2d(inchannel[2]),
            nn.ReLU(True)
        )
        self.up1 = nn.Sequential(
            nn.Conv2d(inchannel[2] + inchannel[1], inchannel[1], 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(inchannel[1], inchannel[1], 3, 1, 1),
            nn.BatchNorm2d(inchannel[1]),
            nn.ReLU(True)
        )
        self.out = nn.Sequential(
            nn.Conv2d(inchannel[1],1,1,bias=False)
        )
    def forward(self,x):
        pre_x = self.preconv_0(x / x.max())
        if self.use:
            df,rgb,mulit = self.DF(x)
            xattention = self.attention(df)
            xDF = self.DFconv(xattention)
            x = torch.cat([pre_x,xDF],dim=1)
        else:
            x = pre_x
        x1_skip = self.down1(x)
        x1 = self.pool(x1_skip)
        x2_skip = self.down2(x1)
        x2 = self.pool(x2_skip)
        x3_skip = self.down3(x2)
        x3 = self.pool(x3_skip)
        x4_skip = self.down4(x3)
        x4 = self.pool(x4_skip)
        x5_skip = self.down5(x4)
        x = self.up4(torch.cat([x4_skip,self.upsample(x5_skip)],dim=1))
        x = self.up3(torch.cat([x3_skip,self.upsample(x)],dim=1))
        x = self.up2(torch.cat([x2_skip, self.upsample(x)], dim=1))
        x = self.up1(torch.cat([x1_skip, self.upsample(x)], dim=1))
        x = self.out(x)
        return x


class Feature_ex(nn.Module):
    def __init__(self,in_ch):
        super(Feature_ex, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch,in_ch//2,1,bias=False),nn.BatchNorm2d(in_ch//2), nn.ReLU(),
            nn.Conv2d(in_ch//2,in_ch//2,3,padding=1),nn.BatchNorm2d(in_ch//2), nn.ReLU(),
            nn.Conv2d(in_ch//2, in_ch, 1, bias=False), nn.BatchNorm2d(in_ch)
        )
    def forward(self, x):
        block = F.relu(self.block(x) + x, True)
        return block

class DSCBAMLayer(nn.Module):
    def __init__(self, channel, reduction=16, spatial_kernel=7):
        super(DSCBAMLayer, self).__init__()

        # channel attention 压缩H,W为1
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # shared MLP
        self.mlp = nn.Sequential(
            # Conv2d比Linear方便操作
            # nn.Linear(channel, channel // reduction, bias=False)
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            # inplace=True直接替换，节省内存
            nn.ReLU(inplace=True),
            # nn.Linear(channel // reduction, channel,bias=False)
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )

        # spatial attention
        self.conv = nn.Conv2d(2, 1, kernel_size=spatial_kernel,
                              padding=spatial_kernel // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.mlp(self.max_pool(x))
        avg_out = self.mlp(self.avg_pool(x))
        channel_out = self.sigmoid(max_out + avg_out)
        x = channel_out * x

        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        spatial_out = self.sigmoid(self.conv(torch.cat([max_out, avg_out], dim=1)))
        x = spatial_out * x
        return x
class pre_module(nn.Module):
    def __init__(self,in_ch):
        super(pre_module, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, in_ch // 2, 1, bias=False), nn.BatchNorm2d(in_ch // 2), nn.ReLU(),
            nn.Conv2d(in_ch // 2, in_ch // 4, 3, padding=1), nn.BatchNorm2d(in_ch // 4), nn.ReLU(),
            nn.Conv2d(in_ch // 4, 1, 1, bias=False)
        )
    def forward(self, x):
        return self.block(x)


class FP_FN_module(nn.Module):
    def __init__(self,in_ch):
        super(FP_FN_module, self).__init__()
        self.FP_ex = Feature_ex(in_ch)
        self.FN_ex = Feature_ex(in_ch)
        self.FN_att = DSCBAMLayer(in_ch)
        self.FP_att = DSCBAMLayer(in_ch)
        self.pre_FP = pre_module(in_ch)
        self.pre_FN = pre_module(in_ch)
        self.BL = nn.Sequential(nn.BatchNorm2d(in_ch),nn.ReLU())

    def forward(self, x):
        FP_feature = self.FP_att(self.FP_ex(x))
        FN_feature = self.FN_att(self.FN_ex(x))
        x = x+FN_feature-FP_feature
        x = self.BL(x)
        FPpre = self.pre_FP(FP_feature)
        FNpre = self.pre_FN(FN_feature)
        return x,FPpre,FNpre


class FrangiNet(nn.Module):
    def __init__(self,inch=3,outch=4,num=16,featurech = 64):
        super(FrangiNet, self).__init__()
        self.frangi = DF_final(inch,outch,num)
        self.attention  = CBAMLayer(channel=outch*num,reduction=4)
        self.conv = nn.Sequential(
            nn.Conv2d(outch*num,featurech,kernel_size=1,stride=1,padding=0,bias=False),
            nn.BatchNorm2d(featurech),
            nn.ReLU(),
            nn.Conv2d(featurech,featurech,kernel_size=3,stride=1,padding=1)
            # nn.BatchNorm2d(featurech),
            # nn.ReLU()

        )
        self.DS = FP_FN_module(featurech)
        self.pre = nn.Conv2d(featurech, 1, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        x, FPpre, FNpre = self.DS(self.conv(self.attention(self.frangi(x)[0])))
        return x, FPpre, FNpre, self.pre(x)


class FrangiNet_2(nn.Module):  #63
    def __init__(self,inch=3,outch=4,num=16,featurech = 64):
        super(FrangiNet_2, self).__init__()
        self.frangi = DF_final(inch,outch,num)
        self.attention  = CBAMLayer(channel=outch*num,reduction=4)
        self.conv = nn.Sequential(
            nn.Conv2d(outch*num,featurech,kernel_size=1,stride=1,padding=0,bias=False),
            nn.BatchNorm2d(featurech),
            nn.ReLU(),
            nn.Conv2d(featurech,featurech,kernel_size=3,stride=1,padding=1)
            # nn.BatchNorm2d(featurech),
            # nn.ReLU()

        )
        # self.DS = FP_FN_module(featurech)
        self.pre = nn.Conv2d(featurech, 1, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        x = self.conv(self.attention(self.frangi(x)[0]))
        return self.pre(x)




class FrangiNet_516(nn.Module):  # 516文件
    def __init__(self,inch=3,outch=4,num=16,featurech = 64,isgray = False):
        super(FrangiNet_516, self).__init__()
        self.frangi = Frangi516(inch,outch,num,isgray=isgray)
        self.attention  = CBAMLayer(outch*num,reduction=4)
        self.pre = nn.Sequential(
            nn.Conv2d(outch*num,featurech,kernel_size=1,stride=1,padding=0,bias=False),
            nn.BatchNorm2d(featurech),
            nn.ReLU(),
            nn.Conv2d(featurech,featurech,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(featurech),
            nn.ReLU(),
            nn.Conv2d(featurech, 1, kernel_size=1, stride=1, padding=0, bias=False)
        )


    def forward(self, x):
        x = self.frangi(x)
        x_attention = self.attention(x[0])
        x_pre = self.pre(x_attention)
        # return a
        # return x_pre,x_attention

        return  x_attention,x_pre

# if __name__ == '__main__':
#     d = UF_net_con_54()
#     x = torch.randn([2,3,128,128],dtype=torch.float32)
#     V = d(x)
#
#
    # a=0

