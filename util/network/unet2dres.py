# -*- coding: utf-8 -*-
from __future__ import print_function, division

import time
import torch
import torch.nn as nn
import numpy as np
from pymic.layer.activation import get_acti_func
from pymic.layer.convolution import ConvolutionLayer, DepthSeperableConvolutionLayer
from pymic.layer.deconvolution import DeconvolutionLayer, DepthSeperableDeconvolutionLayer

def channel_shuffle(x, groups):
    B, C, H, W = x.data.size()
    channels_per_group  = C // groups

    # reshape
    x = x.view(B, groups, channels_per_group, H, W)
    x = torch.transpose(x, 1, 2).contiguous() 

    # flatten
    x = x.view(B, -1, H, W)
    return x 

class UNetBlock(nn.Module):
    def __init__(self,in_channels, out_channels, norm_type, groups, acti_func, acti_func_param):
        super(UNetBlock, self).__init__()
        
        self.in_chns   = in_channels
        self.out_chns  = out_channels
        self.acti_func = acti_func

        group1 = 1 if (in_channels < 8) else groups
        self.conv1 = ConvolutionLayer(in_channels,  out_channels, 1, 
                dim = 2, padding = 0, conv_group = group1, norm_type = norm_type, norm_group = group1,
                acti_func=get_acti_func(acti_func, acti_func_param))
        self.conv2 = ConvolutionLayer(out_channels, out_channels, 3, 
                dim = 2, padding = 1, conv_group = groups, norm_type = norm_type, norm_group = groups,
                acti_func=get_acti_func(acti_func, acti_func_param))

    def forward(self, x):
        f1 = self.conv1(x)
        f2 = self.conv2(f1)
        return f2

class UNetBlock_DW(nn.Module):
    """UNet block with depthwise seperable convolution
    """
    def __init__(self,in_channels, out_channels, norm_type, groups, acti_func, acti_func_param):
        super(UNetBlock_DW, self).__init__()
        self.in_chns   = in_channels
        self.out_chns  = out_channels
        self.acti_func = acti_func
        self.groups    = groups

        self.conv1 = DepthSeperableConvolutionLayer(in_channels, out_channels, 3, 
                dim = 2, padding = 1, conv_group = groups, norm_type = norm_type, norm_group = groups,
                acti_func=get_acti_func(acti_func, acti_func_param))
        self.conv2 = DepthSeperableConvolutionLayer(out_channels, out_channels, 3, 
                dim = 2, padding = 1, conv_group = groups, norm_type = norm_type, norm_group = groups,
                acti_func=get_acti_func(acti_func, acti_func_param))
       
    def forward(self, x):
        f1 = self.conv1(x)
        f2 = self.conv2(f1)
        return f2

class UNetBlock_DW_CF(UNetBlock_DW):
    """UNet block with depthwise seperable convolution
    """
    def __init__(self,in_channels, out_channels, norm_type, groups, acti_func, acti_func_param):
        super(UNetBlock_DW_CF, self).__init__(in_channels, out_channels, norm_type, groups, acti_func, acti_func_param)
       
    def forward(self, x):
        f1 = self.conv1(x)
        if(self.groups > 1):
            f1 = channel_shuffle(f1, groups = self.groups)
        f2 = self.conv2(f1)
        if(self.groups > 1):
            f2 = channel_shuffle(f2, groups = int(self.out_chns / self.groups))
        return f2

class UNetBlock_DW_CF_Res(UNetBlock_DW):
    """UNet block with depthwise seperable convolution
    """
    def __init__(self,in_channels, out_channels, norm_type, groups, acti_func, acti_func_param):
        super(UNetBlock_DW_CF_Res, self).__init__(in_channels, out_channels, norm_type, groups, acti_func, acti_func_param)
       
    def forward(self, x):
        f1 = self.conv1(x)
        if(self.groups > 1):
            f1 = channel_shuffle(f1, groups = self.groups)
        f2 = self.conv2(f1)
        if(self.groups > 1):
            f2 = channel_shuffle(f2, groups = int(self.out_chns / self.groups))
        return f1 + f2

class VanillaBlock(nn.Module):
    def __init__(self,in_channels, out_channels, norm_type, groups, acti_func, acti_func_param):
        super(VanillaBlock, self).__init__()
        
        self.in_chns   = in_channels
        self.out_chns  = out_channels
        self.acti_func = acti_func

        group1 = 1 if (in_channels < 8) else groups
        self.conv1 = ConvolutionLayer(in_channels,  out_channels, 1, 
                dim = 2, padding = 0, conv_group = group1, norm_type = norm_type, norm_group = group1,
                acti_func=get_acti_func(acti_func, acti_func_param))
        self.conv2 = ConvolutionLayer(out_channels, out_channels, 3, 
                dim = 2, padding = 1, conv_group = groups, norm_type = norm_type, norm_group = groups,
                acti_func=get_acti_func(acti_func, acti_func_param))
        self.conv3 = ConvolutionLayer(out_channels, out_channels, 3, 
                dim = 2, padding = 1, conv_group = groups, norm_type = norm_type, norm_group = groups,
                acti_func=get_acti_func(acti_func, acti_func_param))

    def forward(self, x):
        f1 = self.conv1(x)
        f2 = self.conv2(f1)
        f3 = self.conv3(f2)
        return f3

class ResBlock(VanillaBlock):
    def __init__(self,in_channels, out_channels, norm_type, groups, acti_func, acti_func_param):
        super(ResBlock, self).__init__(in_channels, out_channels, norm_type, groups, acti_func, acti_func_param)

    def forward(self, x):
        f1 = self.conv1(x)
        f2 = self.conv2(f1)
        f3 = self.conv3(f2)
        return f1 + f3

class ResBlock_DW(nn.Module):
    """UNet block with depthwise seperable convolution
    """
    def __init__(self,in_channels, out_channels, norm_type, groups, acti_func, acti_func_param):
        super(ResBlock_DW, self).__init__()
        self.in_chns   = in_channels
        self.out_chns  = out_channels
        self.acti_func = acti_func
        self.groups    = groups

        self.conv1 = ConvolutionLayer(in_channels,  out_channels, 1, 
                dim = 2, padding = 0, conv_group = 1, norm_type = norm_type, norm_group = 1,
                acti_func=get_acti_func(acti_func, acti_func_param))
        self.conv2 = DepthSeperableConvolutionLayer(out_channels, out_channels, 3, 
                dim = 2, padding = 1, conv_group = groups, norm_type = norm_type, norm_group = groups,
                acti_func=get_acti_func(acti_func, acti_func_param))
        self.conv3 = DepthSeperableConvolutionLayer(out_channels, out_channels, 3, 
                dim = 2, padding = 1, conv_group = groups, norm_type = norm_type, norm_group = groups,
                acti_func=get_acti_func(acti_func, acti_func_param))
       
    def forward(self, x):
        f1 = self.conv1(x)
        f2 = self.conv2(f1)
        f3 = self.conv3(f2)
        return f1 + f3

class ResBlock_DWGC_CF(nn.Module):
    """UNet block with depthwise seperable convolution and group convolution + channel shuffle
    """
    def __init__(self,in_channels, out_channels, norm_type, groups, acti_func, acti_func_param):
        super(ResBlock_DWGC_CF, self).__init__()
        self.in_chns   = in_channels
        self.out_chns  = out_channels
        self.acti_func = acti_func
        self.groups    = groups
        groups2        = int(out_channels / groups)
        self.conv1 = ConvolutionLayer(in_channels,  out_channels, 1, 
                dim = 2, padding = 0, conv_group = 1, norm_type = norm_type, norm_group = 1,
                acti_func=get_acti_func(acti_func, acti_func_param))
        self.conv2 = DepthSeperableConvolutionLayer(out_channels, out_channels, 3, 
                dim = 2, padding = 1, conv_group = groups, norm_type = norm_type, norm_group = groups,
                acti_func=get_acti_func(acti_func, acti_func_param))
        self.conv3 = DepthSeperableConvolutionLayer(out_channels, out_channels, 3, 
                dim = 2, padding = 1, conv_group = groups2, norm_type = norm_type, norm_group = groups2,
                acti_func=get_acti_func(acti_func, acti_func_param))

    def forward(self, x):
        f1 = self.conv1(x)
        f2 = self.conv2(f1)
        if(self.groups > 1):
            f2 = channel_shuffle(f2, groups = self.groups)
        f3 = self.conv3(f2)
        if(self.groups > 1):
            f3 = channel_shuffle(f3, groups = int(self.out_chns / self.groups))
        return f1 + f3

class PEBlock(nn.Module):
    def __init__(self, channels, acti_func, acti_func_param):
        super(PEBlock, self).__init__()

        self.channels  = channels
        self.acti_func = acti_func

        self.conv1 = ConvolutionLayer(channels,  int(channels / 2), 1, 
                dim = 2, padding = 0, conv_group = 1, norm_type = None, norm_group = 1,
                acti_func=get_acti_func(acti_func, acti_func_param))
        self.conv2 = ConvolutionLayer(int(channels / 2),  channels, 1, 
                dim = 2, padding = 0, conv_group = 1, norm_type = None, norm_group = 1,
                acti_func=nn.Sigmoid())

    def forward(self, x):
        # projection along each dimension
        x_shape = list(x.shape) 
        [N, C, H, W] = x_shape
        p_w = torch.sum(x, dim = -1, keepdim = True) / W  # the shape becomes [N, C, H, 1]
        p_h = torch.sum(x, dim = -2, keepdim = True) / H  # the shape becomes [N, C, 1, W]
        p_w_repeat = p_w.repeat(1, 1, 1, W)               # the shape is [N, C, H, W]
        p_h_repeat = p_h.repeat(1, 1, H, 1)               # the shape is [N, C, H, W]
        f = p_w_repeat + p_h_repeat
        f = self.conv1(f)
        f = self.conv2(f)                                 # get attention coefficient 
        out = f*x + x                                     # use a residual connection
        return out

class ResBlock_DWGC_CF_PE(ResBlock_DW):
    """UNet block with depthwise seperable convolution and group convolution + channel shuffle
    """
    def __init__(self,in_channels, out_channels, norm_type, groups, acti_func, acti_func_param):
        super(ResBlock_DWGC_CF_PE, self).__init__(in_channels, out_channels, 
            norm_type, groups, acti_func, acti_func_param)
        self.pe_block = PEBlock(out_channels, acti_func, acti_func_param)
       
    def forward(self, x):
        f1 = self.conv1(x)
        if(self.groups > 1):
            f1 = channel_shuffle(f1, groups = self.groups)
        f2 = self.conv2(f1)
        f3 = self.conv3(f2)
        if(self.groups > 1):
            f3 = channel_shuffle(f3, groups = self.groups)
        out = f1 + f3
        out = self.pe_block(out)
        return out

class ResBlock_DWGC_CF_BE(nn.Module):
    """UNet block with depthwise seperable convolution and group convolution + channel shuffle
    """
    def __init__(self,in_channels, out_channels, norm_type, groups, acti_func, acti_func_param):
        super(ResBlock_DWGC_CF_BE, self).__init__()
        
        self.in_chns   = in_channels
        self.out_chns  = out_channels
        self.acti_func = acti_func
        self.groups    = groups

        self.conv1 = ConvolutionLayer(in_channels,  out_channels, 1, 
                dim = 2, padding = 0, conv_group = groups, norm_type = norm_type, norm_group = groups,
                acti_func=get_acti_func(acti_func, acti_func_param))
        self.conv2 = DepthSeperableConvolutionLayer(out_channels, out_channels*2, 3, 
                dim = 2, padding = 1, conv_group = groups, norm_type = norm_type, norm_group = groups,
                acti_func=get_acti_func(acti_func, acti_func_param))
        self.conv3 = DepthSeperableConvolutionLayer(out_channels*2, out_channels, 3, 
                dim = 2, padding = 1, conv_group = groups, norm_type = norm_type, norm_group = groups,
                acti_func=get_acti_func(acti_func, acti_func_param))
       
    def forward(self, x):
        f1 = self.conv1(x)
        if(self.groups > 1):
            f1 = channel_shuffle(f1, groups = self.groups)
        f2 = self.conv2(f1)
        f3 = self.conv3(f2)
        if(self.groups > 1):
            f3 = channel_shuffle(f3, groups = self.groups)
        return f1 + f3



class ResBlock_DWGC_BE_CPF(nn.Module):
    """UNet block with depthwise seperable convolution and group convolution + bottleneck with expansion layer
    + channel shuffle and channel split
    """
    def __init__(self,in_channels, out_channels, norm_type, groups, acti_func, acti_func_param):
        super(ResBlock_DWGC_BE_CPF, self).__init__()
        
        self.in_chns   = in_channels
        self.out_chns  = out_channels
        self.acti_func = acti_func

        chns_half = int(out_channels / 2)
        group1 = 1 if (in_channels < 8) else groups
        self.conv1 = ConvolutionLayer(in_channels,  out_channels, 1, 
                dim = 2, padding = 0,conv_group = group1, norm_type = norm_type, norm_group = group1,
                acti_func=get_acti_func(acti_func, acti_func_param))
        self.conv2 = DepthSeperableConvolutionLayer(chns_half, self.out_chns, 3, 
                dim = 2, padding = 1, conv_group = groups, norm_type = norm_type, norm_group = groups,
                acti_func=get_acti_func(acti_func, acti_func_param))
        self.conv3 = DepthSeperableConvolutionLayer(self.out_chns, chns_half, 3, 
                dim = 2, padding = 1, conv_group = groups, norm_type = norm_type, norm_group = groups,
                acti_func=get_acti_func(acti_func, acti_func_param))
       
    def forward(self, x):
        chns_half = int(self.out_chns / 2)

        f1 = self.conv1(x)
        f1_shuffle = channel_shuffle(f1, groups = 2)
        f1_a = f1_shuffle[:,0:chns_half, :, :]
        f1_b = f1_shuffle[:,chns_half:,  :, :]

        f2 = self.conv2(f1_b)
        f3 = self.conv3(f2)

        f3cat = torch.cat([f1_a, f3], dim = 1)
        out   = channel_shuffle(f3cat, groups = 2)
        return out


def get_unet_block(block_type):
    if(block_type == "UNetBlock"):
        return UNetBlock
    elif(block_type == "UNetBlock_DW"):
        return UNetBlock_DW
    elif(block_type == "UNetBlock_DW_CF"):
        return UNetBlock_DW_CF
    elif(block_type == "UNetBlock_DW_CF_Res"):
        return UNetBlock_DW_CF_Res
    elif(block_type == "VanillaBlock"):
        return VanillaBlock
    elif(block_type == "ResBlock"):
        return ResBlock
    elif(block_type == "ResBlock_DW"):
        return ResBlock_DW
    elif(block_type == "ResBlock_DWGC_CF"):
        return ResBlock_DWGC_CF
    elif(block_type == "ResBlock_DWGC_CF_BE"):
        return ResBlock_DWGC_CF_BE
    elif(block_type == "ResBlock_DWGC_CF_PE"):
        return ResBlock_DWGC_CF_PE
    else:
        raise ValueError("undefined type name {0:}".format(block_type))

def get_deconv_layer(depth_sep_deconv):
    if(depth_sep_deconv):
        return DepthSeperableDeconvolutionLayer
    else:
        return DeconvolutionLayer

class UNet2DRes(nn.Module):
    def __init__(self, params):
        super(UNet2DRes, self).__init__()
        self.params = params
        self.in_chns   = self.params['in_chns']
        self.ft_chns   = self.params['feature_chns']
        self.ft_groups = self.params['feature_grps']
        self.norm_type = self.params['norm_type']
        self.block_type= self.params['block_type']
        self.n_class   = self.params['class_num']
        self.acti_func = self.params['acti_func']
        self.dropout   = self.params['dropout']
        self.depth_sep_deconv= self.params['depth_sep_deconv']
        self.deep_spv  = self.params['deep_supervision']
        self.pe_block  = self.params.get('pe_block', False)
        self.resolution_level = len(self.ft_chns)
        assert(self.resolution_level == 5 or self.resolution_level == 4)

        Block = get_unet_block(self.block_type)
        self.block1 = Block(self.in_chns, self.ft_chns[0], self.norm_type, self.ft_groups[0],
             self.acti_func, self.params)

        self.block2 = Block(self.ft_chns[0], self.ft_chns[1], self.norm_type, self.ft_groups[1],
             self.acti_func, self.params)

        self.block3 = Block(self.ft_chns[1], self.ft_chns[2], self.norm_type, self.ft_groups[2],
             self.acti_func, self.params)

        self.block4 = Block(self.ft_chns[2], self.ft_chns[3], self.norm_type, self.ft_groups[3],
             self.acti_func, self.params)

        if(self.resolution_level == 5):
            self.block5 = Block(self.ft_chns[3], self.ft_chns[4], self.norm_type, self.ft_groups[4],
                self.acti_func, self.params)

            self.block6 = Block(self.ft_chns[3] * 2, self.ft_chns[3], self.norm_type, self.ft_groups[3],
                self.acti_func, self.params)

        self.block7 = Block(self.ft_chns[2] * 2, self.ft_chns[2], self.norm_type, self.ft_groups[2],
             self.acti_func, self.params)

        self.block8 = Block(self.ft_chns[1] * 2, self.ft_chns[1], self.norm_type, self.ft_groups[1],
             self.acti_func, self.params)

        self.block9 = Block(self.ft_chns[0] * 2, self.ft_chns[0], self.norm_type, self.ft_groups[0],
             self.acti_func, self.params)

        if(self.pe_block):
            self.pe1 = PEBlock(self.ft_chns[0], self.acti_func, self.params)
            self.pe2 = PEBlock(self.ft_chns[1], self.acti_func, self.params)
            self.pe3 = PEBlock(self.ft_chns[2], self.acti_func, self.params)
            self.pe4 = PEBlock(self.ft_chns[3], self.acti_func, self.params)
            self.pe7 = PEBlock(self.ft_chns[2], self.acti_func, self.params)
            self.pe8 = PEBlock(self.ft_chns[1], self.acti_func, self.params)
            self.pe9 = PEBlock(self.ft_chns[0], self.acti_func, self.params)
            if(self.resolution_level == 5):
                self.pe5 = PEBlock(self.ft_chns[4], self.acti_func, self.params)
                self.pe6 = PEBlock(self.ft_chns[3], self.acti_func, self.params)

        self.down1 = nn.MaxPool2d(kernel_size = 2)
        self.down2 = nn.MaxPool2d(kernel_size = 2)
        self.down3 = nn.MaxPool2d(kernel_size = 2)

        DeconvLayer = get_deconv_layer(self.depth_sep_deconv)
        if(self.resolution_level == 5):
            self.down4 = nn.MaxPool2d(kernel_size = 2)
            self.up1 = DeconvLayer(self.ft_chns[4], self.ft_chns[3], kernel_size = 2,
                dim = 2, stride = 2, groups = 1, acti_func = get_acti_func(self.acti_func, self.params))
        self.up2 = DeconvLayer(self.ft_chns[3], self.ft_chns[2], kernel_size = 2,
                dim = 2, stride = 2, groups = 1, acti_func = get_acti_func(self.acti_func, self.params))
        self.up3 = DeconvLayer(self.ft_chns[2], self.ft_chns[1], kernel_size = 2,
                dim = 2, stride = 2, groups = 1, acti_func = get_acti_func(self.acti_func, self.params))
        self.up4 = DeconvLayer(self.ft_chns[1], self.ft_chns[0], kernel_size = 2,
                dim = 2, stride = 2, groups = 1, acti_func = get_acti_func(self.acti_func, self.params))

        if(self.dropout):
            self.drop1 = nn.Dropout(p=0.1)
            self.drop2 = nn.Dropout(p=0.2)
            self.drop3 = nn.Dropout(p=0.3)
            self.drop4 = nn.Dropout(p=0.4)
            if(self.resolution_level == 5):
                self.drop5 = nn.Dropout(p=0.5)
        
        if(self.deep_spv):
            self.conv7 = nn.Conv2d(self.ft_chns[2], self.n_class,
                kernel_size = 3, padding = 1)
            self.conv8 = nn.Conv2d(self.ft_chns[1], self.n_class,
                kernel_size = 3, padding = 1)

        self.conv9 = nn.Conv2d(self.ft_chns[0], self.n_class, 
            kernel_size = 3, padding = 1)
            

    def forward(self, x):
        x_shape = list(x.shape)
        if(len(x_shape)==5):
            [N, C, D, H, W] = x_shape
            new_shape = [N*D, C, H, W]
            x = torch.transpose(x, 1, 2)
            x = torch.reshape(x, new_shape)
        f1 = self.block1(x)
        if(self.pe_block):
            f1 = self.pe1(f1)
        if(self.dropout):
             f1 = self.drop1(f1)
        d1 = self.down1(f1)

        f2 = self.block2(d1)
        if(self.pe_block):
            f2 = self.pe2(f2)
        if(self.dropout):
             f2 = self.drop2(f2)
        d2 = self.down2(f2)

        f3 = self.block3(d2)
        if(self.pe_block):
            f3 = self.pe3(f3)
        if(self.dropout):
             f3 = self.drop3(f3)
        d3 = self.down3(f3)

        f4 = self.block4(d3)
        if(self.pe_block):
            f4 = self.pe4(f4)
        if(self.dropout):
             f4 = self.drop4(f4)

        if(self.resolution_level == 5):
            d4 = self.down4(f4)
            f5 = self.block5(d4)
            if(self.pe_block):
                f5 = self.pe5(f5)
            if(self.dropout):
                f5 = self.drop5(f5)

            f5up  = self.up1(f5)
            f4cat = torch.cat((f4, f5up), dim = 1)
            f6    = self.block6(f4cat)
            if(self.pe_block):
                f6 = self.pe6(f6)
            f6up  = self.up2(f6)
            f3cat = torch.cat((f3, f6up), dim = 1)
        else:
            f4up  = self.up2(f4)
            f3cat = torch.cat((f3, f4up), dim = 1)
        f7    = self.block7(f3cat)
        if(self.pe_block):
            f7 = self.pe7(f7)
        f7up  = self.up3(f7)
        if(self.deep_spv):
            f7pred = self.conv7(f7)
            f7predup_out = nn.functional.interpolate(f7pred,
                        size = list(x.shape)[2:], mode = 'bilinear')

        f2cat = torch.cat((f2, f7up), dim = 1)
        f8    = self.block8(f2cat)
        if(self.pe_block):
            f8 = self.pe8(f8)
        f8up  = self.up4(f8)
        if(self.deep_spv):
            f8pred = self.conv8(f8)
            f8predup_out = nn.functional.interpolate(f8pred,
                        size = list(x.shape)[2:], mode = 'bilinear')

        f1cat = torch.cat((f1, f8up), dim = 1)
        f9    = self.block9(f1cat)
        if(self.pe_block):
            f9 = self.pe9(f9)
        output = self.conv9(f9)

        if(len(x_shape)==5):
            new_shape = [N, D] + list(output.shape)[1:]
            output = torch.reshape(output, new_shape)
            output = torch.transpose(output, 1, 2)

            if(self.deep_spv):
                f7predup_out = torch.reshape(f7predup_out, new_shape)
                f7predup_out = torch.transpose(f7predup_out, 1, 2)
                f8predup_out = torch.reshape(f8predup_out, new_shape)
                f8predup_out = torch.transpose(f8predup_out, 1, 2)
        if(self.deep_spv):
            return output, f8predup_out, f7predup_out
        else:
            return output

if __name__ == "__main__":
    methods = ["ResBlock", 
               "ResBlock_DW", 
               "ResBlock_DW", # GC
               "ResBlock_DWGC_CF", # GC
               "ResBlock_DWGC_CF_BE",
               "ResBlock_DWGC_CF_PE"]
    method_id = 5
    if(method_id > 1):
        feature_grps = [1, 2,  2,  4,  4]
    else:
        feature_grps = [1, 1, 1, 1, 1]
    
    params = {'in_chns':1,
              'feature_chns':[32, 64, 128, 256, 512],
              'feature_grps':feature_grps,
              'class_num'   : 2,
              'block_type'  : methods[method_id],
              'norm_type'   : 'batch_norm',
              'acti_func': 'relu',
              'dropout'  : True,
              'depth_sep_deconv' : True, 
              'deep_supervision': True}
    Net = UNet2DRes(params)
    Net = Net.double()
    device = torch.device('cuda:1')
    Net.to(device)
    x  = np.random.rand(1, 1, 12, 144, 144) # N, C, H, W
    xt = torch.from_numpy(x)
    xt = torch.tensor(xt)
    xt = xt.to(device)
    t_list = []
    for i in range(10):
        t0 = time.time()
        y, y1 = Net(xt)
        t  = time.time()  - t0 
        t_list.append(t)
    t_array = np.asarray(t_list)
    print('time', t_array.mean())
    print(len(y.size()))
    y = y.detach().cpu().numpy()
    print(y.shape)

    # device = torch.device('cpu')

    # param = {'acti_func':'relu'}
    # Net = PEBlock(12, 'relu', param)
    # Net = Net.double()
    # Net.to(device)
    # x = np.random.rand(1,  12, 144, 144) # N, C, H, W
    # xt = torch.from_numpy(x)
    # xt = torch.tensor(xt)
    # xt = xt.to(device)
    # y = Net(xt)
    # y = y.detach().numpy()
    # print(y.shape)