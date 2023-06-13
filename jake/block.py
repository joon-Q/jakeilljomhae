import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops


class LayerNormFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None

class LayerNorm2d(nn.Module):

    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)

class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

class NAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        
        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp

        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma

class DeformableConv2d(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 bias=False):

        super(DeformableConv2d, self).__init__()
        
        assert type(kernel_size) == tuple or type(kernel_size) == int

        kernel_size = kernel_size if type(kernel_size) == tuple else (kernel_size, kernel_size)
        self.stride = stride if type(stride) == tuple else (stride, stride)
        self.padding = padding
        
        self.offset_conv = nn.Conv2d(in_channels*2, 
                                     2 * kernel_size[0] * kernel_size[1],
                                     kernel_size=kernel_size, 
                                     stride=stride,
                                     padding=self.padding, 
                                     bias=True)

        nn.init.constant_(self.offset_conv.weight, 0.)
        nn.init.constant_(self.offset_conv.bias, 0.)
        
        self.modulator_conv = nn.Conv2d(in_channels, 
                                     1 * kernel_size[0] * kernel_size[1],
                                     kernel_size=kernel_size, 
                                     stride=stride,
                                     padding=self.padding, 
                                     bias=True)

        nn.init.constant_(self.modulator_conv.weight, 0.)
        nn.init.constant_(self.modulator_conv.bias, 0.)
        
        self.regular_conv = nn.Conv2d(in_channels=in_channels,
                                      out_channels=out_channels,
                                      kernel_size=kernel_size,
                                      stride=stride,
                                      padding=self.padding,
                                      bias=bias)

    def forward(self, x, ref):
        h, w = x.shape[2:]
        max_offset = max(h, w)/4.

        offset = self.offset_conv(torch.cat([ref, x], dim =1)).clamp(-max_offset, max_offset)
        modulator = 2. * torch.sigmoid(self.modulator_conv(x))
        
        x = torchvision.ops.deform_conv2d(input=x, 
                                          offset=offset, 
                                          weight=self.regular_conv.weight, 
                                          bias=self.regular_conv.bias, 
                                          padding=self.padding,
                                          mask=modulator,
                                          stride=self.stride,
                                          )
        return x

class DenseBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, bias=True, activation='prelu', norm='batch'):
        super(DenseBlock, self).__init__()
        self.fc = torch.nn.Linear(input_size, output_size, bias=bias)

        self.norm = norm
        if self.norm =='batch':
            self.bn = torch.nn.BatchNorm1d(output_size)
        elif self.norm == 'instance':
            self.bn = torch.nn.InstanceNorm1d(output_size)

        self.activation = activation
        if self.activation == 'relu':
            self.act = torch.nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = torch.nn.PReLU()
        elif self.activation == 'lrelu':
            self.act = torch.nn.LeakyReLU(0.2, True)
        elif self.activation == 'tanh':
            self.act = torch.nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = torch.nn.Sigmoid()
        elif self.activation == 'mprelu':
            self.act = MPReLU() 

    def forward(self, x):
        if self.norm is not None:
            out = self.bn(self.fc(x))
        else:
            out = self.fc(x)

        if self.activation is not None:
            return self.act(out)
        else:
            return out

class ConvBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size=3, stride=1, padding=1, bias=True, activation='gelu', norm='instance'):
        super(ConvBlock, self).__init__()
        self.conv = torch.nn.Conv2d(input_size, output_size, kernel_size, stride, padding, bias=bias)

        self.norm = norm
        if self.norm =='batch':
            self.bn = torch.nn.BatchNorm2d(output_size)
        elif self.norm == 'instance':
            self.bn = torch.nn.InstanceNorm2d(output_size)

        self.activation = activation
        if self.activation == 'relu':
            self.act = torch.nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = torch.nn.PReLU()
        elif self.activation == 'lrelu':
            self.act = torch.nn.LeakyReLU(0.2, True)
        elif self.activation == 'tanh':
            self.act = torch.nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = torch.nn.Sigmoid()
        elif self.activation == 'mprelu':
            self.act = MPReLU()
        elif self.activation == 'gelu':
            self.act = torch.nn.GELU()

    def forward(self, x):
        if self.norm is not None:
            out = self.bn(self.conv(x))
        else:
            out = self.conv(x)

        if self.activation is not None:
            return self.act(out)
        else:
            return out

class Upsample2xBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, bias=True, upsample='rnc', activation='relu', norm='batch'):
        super(Upsample2xBlock, self).__init__()
        scale_factor = 2
        
        self.upsample = torch.nn.Sequential(
            torch.nn.Upsample(scale_factor=scale_factor, mode='nearest'),
            ConvBlock(input_size, output_size,
                      kernel_size=3, stride=1, padding=1,
                      bias=bias, activation=activation, norm=norm)
        )

    def forward(self, x):
        out = self.upsample(x)
        return out

class Multi_scaleCNN(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Multi_scaleCNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, padding=0, padding_mode='zeros', bias=False),
            LayerNorm2d(out_ch),
            nn.GELU(),

        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, padding_mode='zeros', bias=False),
            LayerNorm2d(out_ch),
            nn.GELU(),

        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 5, padding=2, padding_mode='zeros', bias=False),
            LayerNorm2d(out_ch),
            nn.GELU(),

        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(out_ch * 3, in_ch, 1, padding=0, padding_mode='zeros', bias=False),
            LayerNorm2d(out_ch),
            nn.GELU(),

        )

    def forward(self, x):
        out1 = self.conv1(x)
        out2 = self.conv2(x)
        out3 = self.conv3(x)

        out = torch.cat([out1, out2, out3], dim=1)

        out = self.conv4(out) + x

        return out

class Multi_scaleCNN1(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Multi_scaleCNN1, self).__init__()
        self.conv = nn.Sequential(
            Multi_scaleCNN(in_ch, out_ch),
        )

    def forward(self, x):
        out = self.conv(x)
        return out

class Multi_scaleCNN2(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Multi_scaleCNN2, self).__init__()
        self.conv = nn.Sequential(
            Multi_scaleCNN(in_ch, out_ch),
            Multi_scaleCNN(in_ch, out_ch),

        )

    def forward(self, x):
        out = self.conv(x)
        return out

class Multi_scaleCNN3(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Multi_scaleCNN3, self).__init__()
        self.conv = nn.Sequential(
            Multi_scaleCNN(in_ch, out_ch),
            Multi_scaleCNN(in_ch, out_ch),
            Multi_scaleCNN(in_ch, out_ch),
        )

    def forward(self, x):
        out = self.conv(x)
        return out

class Multi_scaleCNN4(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Multi_scaleCNN4, self).__init__()
        self.conv = nn.Sequential(
            Multi_scaleCNN(in_ch, out_ch),
            Multi_scaleCNN(in_ch, out_ch),
            Multi_scaleCNN(in_ch, out_ch),
        )

    def forward(self, x):
        out = self.conv(x)
        return out

class Multi_scaleCNN32(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Multi_scaleCNN32, self).__init__()
        self.conv = nn.Sequential(
            Multi_scaleCNN4(in_ch, out_ch),
            Multi_scaleCNN4(in_ch, out_ch),
            Multi_scaleCNN4(in_ch, out_ch),
        )

    def forward(self, x):
        out = self.conv(x)
        return out

class Encoder1Conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Encoder1Conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, padding_mode='zeros', bias=False),
            LayerNorm2d(out_ch),
            nn.GELU(),
        )

    def forward(self, x):
        out = self.conv(x)
        return out

class conv_block(nn.Module):
    """
    Convolution Block 
    """
    def __init__(self, in_ch, out_ch, stride):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride = stride, padding=1, bias=False),
            LayerNorm2d(out_ch),
            nn.GELU()
            )

    def forward(self, x):
        x = self.conv(x)
        return x

class up_conv(nn.Module):
    """
    Up Convolution Block
    """
    def __init__(self, in_ch, out_ch):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            LayerNorm2d(out_ch),
            nn.GELU()            
        )

    def forward(self, x):
        x = self.up(x)
        return x

class Resblock(nn.Module):
    """
    Up Convolution Block
    """
    def __init__(self, in_ch, out_ch):
        super(Resblock, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            LayerNorm2d(out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            LayerNorm2d(out_ch),
            # nn.GELU()
        )

    def forward(self, x):
        out = self.net(x) + x
        return out

class DRDB(nn.Module):
    def __init__(self, in_ch=128, growth_rate=64):
        super(DRDB, self).__init__()
        in_ch_ = in_ch
        self.Dcov1 = nn.Conv2d(in_ch_, growth_rate, 3, padding=2, dilation=2)
        in_ch_ += growth_rate
        self.Dcov2 = nn.Conv2d(in_ch_, growth_rate, 3, padding=2, dilation=2)
        in_ch_ += growth_rate
        self.Dcov3 = nn.Conv2d(in_ch_, growth_rate, 3, padding=2, dilation=2)
        in_ch_ += growth_rate
        self.Dcov4 = nn.Conv2d(in_ch_, growth_rate, 3, padding=2, dilation=2)
        in_ch_ += growth_rate
        self.Dcov5 = nn.Conv2d(in_ch_, growth_rate, 3, padding=2, dilation=2)
        in_ch_ += growth_rate
        self.conv = nn.Conv2d(in_ch_, in_ch, 1, padding=0)

    def forward(self, x):
        x1 = self.Dcov1(x)
        x1 = F.gelu(x1)
        x1 = torch.cat([x, x1], dim=1)

        x2 = self.Dcov2(x1)
        x2 = F.gelu(x2)
        x2 = torch.cat([x1, x2], dim=1)

        x3 = self.Dcov3(x2) 
        x3 = F.gelu(x3)
        x3 = torch.cat([x2, x3], dim=1)

        x4 = self.Dcov4(x3)
        x4 = F.gelu(x4)
        x4 = torch.cat([x3, x4], dim=1)

        x5 = self.Dcov5(x4)
        x5 = F.gelu(x5)
        x5 = torch.cat([x4, x5], dim=1)

        x6 = self.conv(x5)
        out = x + F.gelu(x6)
        return out

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        out = self.sigmoid(avg_out + max_out)
        return out * x

class SpatialAttentionModule(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.att1 = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, padding=1, bias=True)
        self.att2 = nn.Conv2d(dim * 2, dim, kernel_size=3, padding=1, bias=True)
        self.relu = nn.GELU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x1, x2):
        f_cat = torch.cat([x1, x2], dim=1)
        att_map = self.sigmoid(self.att2(self.relu(self.att1(x1))))
        return att_map

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=None, bias=True):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            Encoder1Conv(in_channels, out_channels),
            Multi_scaleCNN1(out_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels

        self.up = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)

        self.conv = nn.Sequential(
            Encoder1Conv(in_channels, out_channels),
            Multi_scaleCNN1(out_channels, out_channels)
        )

    def forward(self, x1):
        x1 = self.up(x1)

        return self.conv(x1)

class DownSample(nn.Module):
    def __init__(self, channels):
        super(DownSample, self).__init__()
        self.body = nn.Sequential(nn.Conv2d(channels, channels // 2, kernel_size=3, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)

class UpSample(nn.Module):
    def __init__(self, channels):
        super(UpSample, self).__init__()
        self.body = nn.Sequential(nn.Conv2d(channels, channels * 2, kernel_size=3, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(hidden_dim, dim, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class TransformerBlock(nn.Module):
    def __init__(self, in_dim, ker):
        super(TransformerBlock, self).__init__()

        # self.q_mr = nn.Conv2d(in_dim, in_dim, kernel_size=1, bias=False)
        # self.k_mr = nn.Conv2d(in_dim, in_dim, kernel_size=1, bias=False)
        # self.v_mr = nn.Conv2d(in_dim, in_dim, kernel_size=1, bias=False)
        # self.k_lr = nn.Conv2d(in_dim, in_dim, kernel_size=1, bias=False)
        # self.v_lr = nn.Conv2d(in_dim, in_dim, kernel_size=1, bias=False)
        
        self.qkv_mr = nn.Conv2d(in_dim, in_dim * 3, kernel_size=1, bias=False)
        self.qkv_conv_mr = nn.Conv2d(in_dim * 3, in_dim * 3, kernel_size=3, padding=1, groups=in_dim * 3, bias=False)
        self.qkv_lr = nn.Conv2d(in_dim, in_dim * 2, kernel_size=1, bias=False)
        self.qkv_conv_lr = nn.Conv2d(in_dim * 2, in_dim * 2, kernel_size=3, padding=1, groups=in_dim * 2, bias=False)
        self.qkv_lr_fake = nn.Conv2d(in_dim, in_dim * 3, kernel_size=1, bias=False)
        self.qkv_conv_lr_fake = nn.Conv2d(in_dim * 3, in_dim * 3, kernel_size=3, padding=1, groups=in_dim * 3, bias=False)

        # self.ma = nn.Conv2d(in_dim, 1, kernel_size=1, bias=False)
        # self.sig = nn.Sigmoid()
             

        
        self.q_merge = nn.Conv2d(in_dim * 2, in_dim * 1, kernel_size=1, bias=False)
       
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=in_dim * 2, out_channels=in_dim * 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )
        # self.fc_q_mr = nn.Linear(ker * ker * in_dim, 2048, bias=False)
        # self.bn_q_mr = nn.LayerNorm(2048)

        # self.fc_k_mr = nn.Linear(ker * ker * in_dim, 2048, bias=False)
        # self.bn_k_mr = nn.LayerNorm(2048)

        # self.fc_v_mr = nn.Linear(ker * ker * in_dim, ker * ker * in_dim, bias=False)
        # self.bn_v_mr = nn.LayerNorm(ker * ker * in_dim)

        # self.fc_k_lr = nn.Linear(ker * ker * in_dim, 2048, bias=False)
        # self.bn_k_lr = nn.LayerNorm(2048)

        # self.fc_v_lr = nn.Linear(ker * ker * in_dim, ker * ker * in_dim, bias=False)
        # self.bn_v_lr = nn.LayerNorm(ker * ker * in_dim)
        

        self.feed_pre = nn.Conv2d(in_dim * 3, in_dim, kernel_size=1, bias=False)
        self.feed = FeedForward(in_dim, in_dim * 4)
        self.ker = ker

    def forward(self, mr, lr, lr_fake):
        b, c, h, w = mr.shape

        feature_q_mr, feature_k_mr, feature_v_mr = self.qkv_conv_mr(self.qkv_mr(mr)).chunk(3, dim=1)
        feature_k_lr, feature_v_lr = self.qkv_conv_lr(self.qkv_lr(lr)).chunk(2, dim=1)
        feature_q_fk, feature_k_fk, feature_v_fk = self.qkv_conv_lr_fake(self.qkv_lr_fake(lr_fake)).chunk(3, dim=1)

        # q_mr_un = F.unfold(feature_q_mr, kernel_size=(self.ker, self.ker), stride=self.ker).transpose(1, 2)
        k_mr_un = F.unfold(feature_k_mr, kernel_size=(self.ker, self.ker), stride=self.ker).transpose(1, 2)
        v_mr_un = F.unfold(feature_v_mr, kernel_size=(self.ker, self.ker), stride=self.ker).transpose(1, 2)

        # q_lr_un = F.unfold(feature_q_fk, kernel_size=(self.ker, self.ker), stride=self.ker).transpose(1, 2)
        k_lr_un = F.unfold(feature_k_lr, kernel_size=(self.ker, self.ker), stride=self.ker).transpose(1, 2)
        v_lr_un = F.unfold(feature_v_lr, kernel_size=(self.ker, self.ker), stride=self.ker).transpose(1, 2)

        k_lr_fk_un = F.unfold(feature_k_fk, kernel_size=(self.ker, self.ker), stride=self.ker).transpose(1, 2)
        v_lr_fk_un = F.unfold(feature_v_fk, kernel_size=(self.ker, self.ker), stride=self.ker).transpose(1, 2)



        # mask_mr = self.sig(self.ma(mr))
        # mask_mr = F.unfold(mask_mr, kernel_size=(self.ker, self.ker), stride=self.ker).transpose(1, 2)

        # q = q_mr_un * mask_mr + q_lr_un * (1 - mask_mr)
        q = F.unfold(self.q_merge((torch.cat([feature_q_mr , feature_q_fk], dim=1))), kernel_size=(self.ker, self.ker), stride=self.ker).transpose(1, 2)

        k_mr_un, v_mr_un = F.normalize(k_mr_un, dim=-1), F.normalize(v_mr_un, dim=-1)
        k_lr_un, v_lr_un = F.normalize(k_lr_un, dim=-1), F.normalize(v_lr_un, dim=-1)
        k_lr_fk_un, v_lr_fk_un = F.normalize(k_lr_fk_un, dim=-1), F.normalize(v_lr_fk_un, dim=-1)
        q = F.normalize(q, dim=-1)
        # q_mr_un = self.bn_q_mr(self.fc_q_mr(q_mr_un))
        # k_mr_un = self.bn_k_mr(self.fc_k_mr(k_mr_un)).permute(0, 2, 1)
        # v_mr_un = self.bn_v_mr(self.fc_v_mr(v_mr_un))
        
        # k_lr_un = self.bn_k_lr(self.fc_k_lr(k_lr_un)).permute(0, 2, 1)
        # v_lr_un = self.bn_v_lr(self.fc_v_lr(v_lr_un))

        k_mr_un = k_mr_un.permute(0, 2, 1)
        k_lr_un = k_lr_un.permute(0, 2, 1)
        k_lr_fk_un = k_lr_fk_un.permute(0, 2, 1)

        qk_mr = torch.bmm(q, k_mr_un).div(q.size(2) ** 0.5)
        qk_lr = torch.bmm(q, k_lr_un).div(q.size(2) ** 0.5)
        qk_lr_fk = torch.bmm(q, k_lr_fk_un).div(q.size(2) ** 0.5)

        att_mr = F.softmax(qk_mr, dim=2)
        att_lr = F.softmax(qk_lr, dim=2)
        att_lr_fk = F.softmax(qk_lr_fk, dim=2)

        output_self = torch.bmm(att_mr, v_mr_un)
        output_cross = torch.bmm(att_lr, v_lr_un)
        output_cross2 = torch.bmm(att_lr_fk, v_lr_fk_un)

        output_self = output_self.transpose(1, 2)
        output_cross = output_cross.transpose(1, 2)
        output_cross2 = output_cross2.transpose(1, 2)

        output_self = F.fold(output_self, output_size=mr.size()[-2:], kernel_size=(self.ker, self.ker), stride=self.ker)
        output_cross = F.fold(output_cross, output_size=mr.size()[-2:], kernel_size=(self.ker, self.ker), stride=self.ker)
        output_cross2 = F.fold(output_cross2, output_size=mr.size()[-2:], kernel_size=(self.ker, self.ker), stride=self.ker)

        pre_out = self.feed_pre(torch.cat([output_self, output_cross, output_cross2], dim = 1))
        output = self.feed(pre_out) + pre_out
        return output

class CAS(nn.Module):
    def __init__(self, in_dim, ker):
        super(CAS, self).__init__()
       
        self.q_mr = nn.Conv2d(in_dim, in_dim, kernel_size=1, bias=False)
        self.q_conv_mr = nn.Conv2d(in_dim, in_dim, kernel_size=3, padding=1, groups=in_dim, bias=False)
        self.kv_lr = nn.Conv2d(in_dim, in_dim * 3, kernel_size=1, bias=False)
        self.kv_conv_lr = nn.Conv2d(in_dim * 3, in_dim * 3, kernel_size=3, padding=1, groups=in_dim * 3, bias=False)
        
        self.norm_q = LayerNorm2d(in_dim)
        self.norm_q_2 = LayerNorm2d(in_dim)
        self.norm_k = LayerNorm2d(in_dim)
        self.norm_v = LayerNorm2d(in_dim)

        self.g1 = nn.GELU()            
        self.g2 = nn.GELU()
        self.g3 = nn.GELU()
        self.g4 = nn.GELU()
        
        self.ker = ker


        self.fc_q_mr = nn.Linear(ker * ker * in_dim, ker * ker * in_dim)
        self.fc_q_lr = nn.Linear(ker * ker * in_dim, ker * ker * in_dim)
        self.fc_k_lr = nn.Linear(ker * ker * in_dim, ker * ker * in_dim)
        self.fc_v_lr = nn.Linear(ker * ker * in_dim, ker * ker * in_dim)

        self.pam1 = conv_block(in_dim * 2, in_dim * 2, stride=1)
        self.pam2 = nn.Conv2d(in_dim * 2, 1, kernel_size=1, padding=0, bias=True)
        self.pam3 = nn.Sigmoid()

        self.q_ca1 = conv_block(in_dim * 2, in_dim * 2, stride=1)
        self.q_ca2 = conv_block(in_dim * 2, in_dim, stride=1)

        self.feed = FeedForward(in_dim, in_dim * 4)

    def forward(self, mr, lr):
        b, c, h, w = mr.shape

        feature_q_mr = self.q_conv_mr(self.q_mr(mr))
        feature_q_lr, feature_k_lr, feature_v_lr = self.kv_conv_lr(self.kv_lr(lr)).chunk(3, dim=1)

        feature_q_mr = self.g1(self.norm_q(feature_q_mr))
        feature_q_lr = self.g2(self.norm_q_2(feature_q_lr))
        feature_k_lr = self.g3(self.norm_k(feature_k_lr))
        feature_v_lr = self.g4(self.norm_v(feature_v_lr))



        #GAM
        # cat = torch.cat([feature_q_mr, feature_q_lr], dim=1)
        # att_spat = self.pam3(self.pam2(self.pam1(cat)))
        # feature_q_lr = feature_q_lr * att_spat

        
        feature_q_mr = torch.cat([feature_q_mr, feature_q_lr], dim=1)
        feature_q_mr = self.q_ca2(self.q_ca1(feature_q_mr))

        q_mr_un = F.unfold(feature_q_mr, kernel_size=(self.ker, self.ker), stride=self.ker).transpose(1, 2)
        # q_lr_un = F.unfold(feature_q_lr, kernel_size=(self.ker, self.ker), stride=self.ker).transpose(1, 2)
        k_lr_un = F.unfold(feature_k_lr, kernel_size=(self.ker, self.ker), stride=self.ker).transpose(1, 2)
        v_lr_un = F.unfold(feature_v_lr, kernel_size=(self.ker, self.ker), stride=self.ker).transpose(1, 2)

        q_mr_un = self.fc_q_mr(q_mr_un)
        # q_lr_un = self.fc_q_lr(q_lr_un)
        k_lr_un = self.fc_k_lr(k_lr_un)
        v_lr_un = self.fc_v_lr(v_lr_un)

        feature_q = q_mr_un


        #PAM
        k_lr_un = k_lr_un.permute(0, 2, 1)
        qk_lr = torch.bmm(feature_q, k_lr_un).div(q_mr_un.size(2) ** 0.5)
        att_lr = F.softmax(qk_lr, dim=2)
        output = torch.bmm(att_lr, v_lr_un)
        output = output.transpose(1, 2)
        output = F.fold(output, output_size=mr.size()[-2:], kernel_size=(self.ker, self.ker), stride=self.ker)
        output = self.feed(output) + output



        # #Gating
        # ga1 = output * self.GA1(output2)
        # ga2 = output2 * self.GA2(output)
        # out = self.feed(self.outconv1(torch.cat([ga1, ga2], dim = 1)))
        return output

class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

class NAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        
        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp

        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma

class NAFNet(nn.Module):

    def __init__(self, img_channel=3, width=16, middle_blk_num=12, enc_blk_nums=[2, 2, 4, 8], dec_blk_nums=[2, 2, 2, 2]):
        super().__init__()

        self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)
        self.ending = nn.Conv2d(in_channels=width, out_channels=3, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = width
        for num in enc_blk_nums:
            self.encoders.append(
                nn.Sequential(
                    *[NAFBlock(chan) for _ in range(num)]
                )
            )
            self.downs.append(
                nn.Conv2d(chan, 2*chan, 2, 2)
            )
            chan = chan * 2

        self.middle_blks = \
            nn.Sequential(
                *[NAFBlock(chan) for _ in range(middle_blk_num)]
            )

        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            self.decoders.append(
                nn.Sequential(
                    *[NAFBlock(chan) for _ in range(num)]
                )
            )

        self.padder_size = 2 ** len(self.encoders)

    def forward(self, inp):
        B, C, H, W = inp.shape
        inp = self.check_image_size(inp)

        # x = self.intro(inp)
        x = inp

        encs = []

        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        x = self.middle_blks(x)

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)

        x = self.ending(x)
        # x = x + inp
        

        return x[:, :, :H, :W]

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x

class BaselineBlock(nn.Module):
    def __init__(self, c, DW_Expand=1, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        
        # Channel Attention
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
            nn.Sigmoid()
        )

        # GELU
        self.gelu = nn.GELU()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.zeta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)


        self.norm1_trans = nn.LayerNorm(c)
        self.attn_trans = MDTA(c, 4)
        self.norm2_trans = nn.LayerNorm(c)
        self.ffn_trans = GDFN(c, 2)



    def forward(self, inp):
        x = inp

        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.gelu(x)
        x = x * self.se(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.gelu(x)
        x = self.conv5(x)

        x = self.dropout2(x)


        b, c, h, w = inp.shape
        inp_trans = inp + self.attn_trans(self.norm1_trans(inp.reshape(b, c, -1).transpose(-2, -1).contiguous()).transpose(-2, -1)
                          .contiguous().reshape(b, c, h, w))
        inp_trans = inp_trans + self.ffn_trans(self.norm2_trans(inp_trans.reshape(b, c, -1).transpose(-2, -1).contiguous()).transpose(-2, -1)
                         .contiguous().reshape(b, c, h, w))

        return y + x * self.gamma + inp_trans * self.zeta

class Baseline(nn.Module):

    def __init__(self, img_channel=3, width=16, middle_blk_num=4, enc_blk_nums=[4,4,4,4], dec_blk_nums=[4,4,4,4], dw_expand=1, ffn_expand=2):
        super().__init__()

        self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)
        self.ending = nn.Conv2d(in_channels=width, out_channels=3, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = width
        for num in enc_blk_nums:
            self.encoders.append(
                nn.Sequential(
                    *[BaselineBlock(chan, dw_expand, ffn_expand) for _ in range(num)]
                )
            )
            self.downs.append(
                nn.Conv2d(chan, 2*chan, 2, 2)
            )
            chan = chan * 2
        
        self.middle_blks = \
            nn.Sequential(
                *[BaselineBlock(chan, dw_expand, ffn_expand) for _ in range(middle_blk_num)]
            )

        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            self.decoders.append(
                nn.Sequential(
                    *[BaselineBlock(chan, dw_expand, ffn_expand) for _ in range(num)]
                )
            )

        self.padder_size = 2 ** len(self.encoders)

    def forward(self, inp):
        B, C, H, W = inp.shape
        inp = self.check_image_size(inp)

        # x = self.intro(inp)
        x = inp

        encs = []

        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        x = self.middle_blks(x)

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)
        
        x_feat = x
        x = x + inp

        x = self.ending(x_feat)


        return x[:, :, :H, :W]

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x

class MDTA(nn.Module):
    def __init__(self, channels, num_heads):
        super(MDTA, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(1, num_heads, 1, 1))

        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=False)
        self.qkv_conv = nn.Conv2d(channels * 3, channels * 3, kernel_size=3, padding=1, groups=channels * 3, bias=False)
        self.project_out = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

    def forward(self, x):
        b, c, h, w = x.shape
        q, k, v = self.qkv_conv(self.qkv(x)).chunk(3, dim=1)

        q = q.reshape(b, self.num_heads, -1, h * w)
        k = k.reshape(b, self.num_heads, -1, h * w)
        v = v.reshape(b, self.num_heads, -1, h * w)
        q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)

        attn = torch.softmax(torch.matmul(q, k.transpose(-2, -1).contiguous()) * self.temperature, dim=-1)
        out = self.project_out(torch.matmul(attn, v).reshape(b, -1, h, w))
        return out

class GDFN(nn.Module):
    def __init__(self, channels, expansion_factor):
        super(GDFN, self).__init__()

        hidden_channels = int(channels * expansion_factor)
        self.project_in = nn.Conv2d(channels, hidden_channels * 2, kernel_size=1, bias=False)
        self.conv = nn.Conv2d(hidden_channels * 2, hidden_channels * 2, kernel_size=3, padding=1,
                              groups=hidden_channels * 2, bias=False)
        self.project_out = nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=False)

    def forward(self, x):
        x1, x2 = self.conv(self.project_in(x)).chunk(2, dim=1)
        x = self.project_out(F.gelu(x1) * x2)
        return x



class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

def window_partition(x, window_size):

    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size, H, W):

    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)

        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):

        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x



class Transformer(nn.Module):

    def __init__(self, dim, input_resolution, num_heads, window_size=8, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            attn_mask = self.calculate_mask(self.input_resolution)
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)
        
    def calculate_mask(self, x_size):
        # calculate attention mask for SW-MSA
        H, W = x_size
        img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        return attn_mask

    def forward(self, x, x_size):
        H, W = x_size
        B, L, C = x.shape
        # assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)
        # local context features
        lcf = x.permute(0, 3, 1, 2)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA (to be compatible for testing on images whose shapes are the multiple of window size
        if self.input_resolution == x_size:
            attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C
        else:
            attn_windows = self.attn(x_windows, mask=self.calculate_mask(x_size).to(x.device))

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x
