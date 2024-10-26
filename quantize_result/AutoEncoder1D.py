# GENETARED BY NNDCT, DO NOT EDIT!

import torch
from torch import tensor
import pytorch_nndct as py_nndct

class AutoEncoder1D(py_nndct.nn.NndctQuantModel):
    def __init__(self):
        super(AutoEncoder1D, self).__init__()
        self.module_0 = py_nndct.nn.Input() #AutoEncoder1D::input_0(AutoEncoder1D::nndct_input_0)
        self.module_1 = py_nndct.nn.Module('nndct_shape') #AutoEncoder1D::AutoEncoder1D/2289(AutoEncoder1D::nndct_shape_1)
        self.module_2 = py_nndct.nn.Module('nndct_shape') #AutoEncoder1D::AutoEncoder1D/2298(AutoEncoder1D::nndct_shape_2)
        self.module_3 = py_nndct.nn.Module('nndct_reshape') #AutoEncoder1D::AutoEncoder1D/ret.3(AutoEncoder1D::nndct_reshape_3)
        self.module_4 = py_nndct.nn.Module('aten::_convolution_mode') #AutoEncoder1D::AutoEncoder1D/ConvBlock[spatial_reduce]/Conv1d[conv1d]/ret.5(AutoEncoder1D::aten__convolution_mode_4)
        self.module_5 = py_nndct.nn.Module('nndct_transpose') #AutoEncoder1D::AutoEncoder1D/ConvBlock[spatial_reduce]/ret.7(AutoEncoder1D::nndct_transpose_5)
        self.module_6 = py_nndct.nn.LayerNorm(normalized_shape=[32], eps=1e-05, elementwise_affine=True) #AutoEncoder1D::AutoEncoder1D/ConvBlock[spatial_reduce]/LayerNorm[norm]/ret.9(AutoEncoder1D::nndct_layer_norm_6)
        self.module_7 = py_nndct.nn.Module('nndct_transpose') #AutoEncoder1D::AutoEncoder1D/ConvBlock[spatial_reduce]/ret.11(AutoEncoder1D::nndct_transpose_7)
        self.module_8 = py_nndct.nn.GELU(approximate='none') #AutoEncoder1D::AutoEncoder1D/ConvBlock[spatial_reduce]/GELU[activation]/ret.13(AutoEncoder1D::nndct_GELU_8)
        self.module_9 = py_nndct.nn.MaxPool1d(kernel_size=[1], stride=[1], padding=[0], dilation=[1], ceil_mode=False) #AutoEncoder1D::AutoEncoder1D/ConvBlock[spatial_reduce]/MaxPool1d[downsample]/ret.15(AutoEncoder1D::nndct_maxpool1d_9)
        self.module_10 = py_nndct.nn.Module('aten::_convolution_mode') #AutoEncoder1D::AutoEncoder1D/ConvBlock[downsample_blocks]/ModuleList[0]/Conv1d[conv1d]/ret.17(AutoEncoder1D::aten__convolution_mode_10)
        self.module_11 = py_nndct.nn.Module('nndct_transpose') #AutoEncoder1D::AutoEncoder1D/ConvBlock[downsample_blocks]/ModuleList[0]/ret.19(AutoEncoder1D::nndct_transpose_11)
        self.module_12 = py_nndct.nn.LayerNorm(normalized_shape=[32], eps=1e-05, elementwise_affine=True) #AutoEncoder1D::AutoEncoder1D/ConvBlock[downsample_blocks]/ModuleList[0]/LayerNorm[norm]/ret.21(AutoEncoder1D::nndct_layer_norm_12)
        self.module_13 = py_nndct.nn.Module('nndct_transpose') #AutoEncoder1D::AutoEncoder1D/ConvBlock[downsample_blocks]/ModuleList[0]/ret.23(AutoEncoder1D::nndct_transpose_13)
        self.module_14 = py_nndct.nn.GELU(approximate='none') #AutoEncoder1D::AutoEncoder1D/ConvBlock[downsample_blocks]/ModuleList[0]/GELU[activation]/ret.25(AutoEncoder1D::nndct_GELU_14)
        self.module_15 = py_nndct.nn.MaxPool1d(kernel_size=[2], stride=[2], padding=[0], dilation=[1], ceil_mode=False) #AutoEncoder1D::AutoEncoder1D/ConvBlock[downsample_blocks]/ModuleList[0]/MaxPool1d[downsample]/ret.27(AutoEncoder1D::nndct_maxpool1d_15)
        self.module_16 = py_nndct.nn.Module('aten::_convolution_mode') #AutoEncoder1D::AutoEncoder1D/ConvBlock[downsample_blocks]/ModuleList[1]/Conv1d[conv1d]/ret.29(AutoEncoder1D::aten__convolution_mode_16)
        self.module_17 = py_nndct.nn.Module('nndct_transpose') #AutoEncoder1D::AutoEncoder1D/ConvBlock[downsample_blocks]/ModuleList[1]/ret.31(AutoEncoder1D::nndct_transpose_17)
        self.module_18 = py_nndct.nn.LayerNorm(normalized_shape=[64], eps=1e-05, elementwise_affine=True) #AutoEncoder1D::AutoEncoder1D/ConvBlock[downsample_blocks]/ModuleList[1]/LayerNorm[norm]/ret.33(AutoEncoder1D::nndct_layer_norm_18)
        self.module_19 = py_nndct.nn.Module('nndct_transpose') #AutoEncoder1D::AutoEncoder1D/ConvBlock[downsample_blocks]/ModuleList[1]/ret.35(AutoEncoder1D::nndct_transpose_19)
        self.module_20 = py_nndct.nn.GELU(approximate='none') #AutoEncoder1D::AutoEncoder1D/ConvBlock[downsample_blocks]/ModuleList[1]/GELU[activation]/ret.37(AutoEncoder1D::nndct_GELU_20)
        self.module_21 = py_nndct.nn.MaxPool1d(kernel_size=[2], stride=[2], padding=[0], dilation=[1], ceil_mode=False) #AutoEncoder1D::AutoEncoder1D/ConvBlock[downsample_blocks]/ModuleList[1]/MaxPool1d[downsample]/ret.39(AutoEncoder1D::nndct_maxpool1d_21)
        self.module_22 = py_nndct.nn.Module('aten::_convolution_mode') #AutoEncoder1D::AutoEncoder1D/ConvBlock[downsample_blocks]/ModuleList[2]/Conv1d[conv1d]/ret.41(AutoEncoder1D::aten__convolution_mode_22)
        self.module_23 = py_nndct.nn.Module('nndct_transpose') #AutoEncoder1D::AutoEncoder1D/ConvBlock[downsample_blocks]/ModuleList[2]/ret.43(AutoEncoder1D::nndct_transpose_23)
        self.module_24 = py_nndct.nn.LayerNorm(normalized_shape=[64], eps=1e-05, elementwise_affine=True) #AutoEncoder1D::AutoEncoder1D/ConvBlock[downsample_blocks]/ModuleList[2]/LayerNorm[norm]/ret.45(AutoEncoder1D::nndct_layer_norm_24)
        self.module_25 = py_nndct.nn.Module('nndct_transpose') #AutoEncoder1D::AutoEncoder1D/ConvBlock[downsample_blocks]/ModuleList[2]/ret.47(AutoEncoder1D::nndct_transpose_25)
        self.module_26 = py_nndct.nn.GELU(approximate='none') #AutoEncoder1D::AutoEncoder1D/ConvBlock[downsample_blocks]/ModuleList[2]/GELU[activation]/ret.49(AutoEncoder1D::nndct_GELU_26)
        self.module_27 = py_nndct.nn.MaxPool1d(kernel_size=[2], stride=[2], padding=[0], dilation=[1], ceil_mode=False) #AutoEncoder1D::AutoEncoder1D/ConvBlock[downsample_blocks]/ModuleList[2]/MaxPool1d[downsample]/ret.51(AutoEncoder1D::nndct_maxpool1d_27)
        self.module_28 = py_nndct.nn.Module('aten::_convolution_mode') #AutoEncoder1D::AutoEncoder1D/ConvBlock[downsample_blocks]/ModuleList[3]/Conv1d[conv1d]/ret.53(AutoEncoder1D::aten__convolution_mode_28)
        self.module_29 = py_nndct.nn.Module('nndct_transpose') #AutoEncoder1D::AutoEncoder1D/ConvBlock[downsample_blocks]/ModuleList[3]/ret.55(AutoEncoder1D::nndct_transpose_29)
        self.module_30 = py_nndct.nn.LayerNorm(normalized_shape=[128], eps=1e-05, elementwise_affine=True) #AutoEncoder1D::AutoEncoder1D/ConvBlock[downsample_blocks]/ModuleList[3]/LayerNorm[norm]/ret.57(AutoEncoder1D::nndct_layer_norm_30)
        self.module_31 = py_nndct.nn.Module('nndct_transpose') #AutoEncoder1D::AutoEncoder1D/ConvBlock[downsample_blocks]/ModuleList[3]/ret.59(AutoEncoder1D::nndct_transpose_31)
        self.module_32 = py_nndct.nn.GELU(approximate='none') #AutoEncoder1D::AutoEncoder1D/ConvBlock[downsample_blocks]/ModuleList[3]/GELU[activation]/ret.61(AutoEncoder1D::nndct_GELU_32)
        self.module_33 = py_nndct.nn.MaxPool1d(kernel_size=[2], stride=[2], padding=[0], dilation=[1], ceil_mode=False) #AutoEncoder1D::AutoEncoder1D/ConvBlock[downsample_blocks]/ModuleList[3]/MaxPool1d[downsample]/ret.63(AutoEncoder1D::nndct_maxpool1d_33)
        self.module_34 = py_nndct.nn.Module('aten::_convolution_mode') #AutoEncoder1D::AutoEncoder1D/ConvBlock[downsample_blocks]/ModuleList[4]/Conv1d[conv1d]/ret.65(AutoEncoder1D::aten__convolution_mode_34)
        self.module_35 = py_nndct.nn.Module('nndct_transpose') #AutoEncoder1D::AutoEncoder1D/ConvBlock[downsample_blocks]/ModuleList[4]/ret.67(AutoEncoder1D::nndct_transpose_35)
        self.module_36 = py_nndct.nn.LayerNorm(normalized_shape=[128], eps=1e-05, elementwise_affine=True) #AutoEncoder1D::AutoEncoder1D/ConvBlock[downsample_blocks]/ModuleList[4]/LayerNorm[norm]/ret.69(AutoEncoder1D::nndct_layer_norm_36)
        self.module_37 = py_nndct.nn.Module('nndct_transpose') #AutoEncoder1D::AutoEncoder1D/ConvBlock[downsample_blocks]/ModuleList[4]/ret.71(AutoEncoder1D::nndct_transpose_37)
        self.module_38 = py_nndct.nn.GELU(approximate='none') #AutoEncoder1D::AutoEncoder1D/ConvBlock[downsample_blocks]/ModuleList[4]/GELU[activation]/ret.73(AutoEncoder1D::nndct_GELU_38)
        self.module_39 = py_nndct.nn.MaxPool1d(kernel_size=[2], stride=[2], padding=[0], dilation=[1], ceil_mode=False) #AutoEncoder1D::AutoEncoder1D/ConvBlock[downsample_blocks]/ModuleList[4]/MaxPool1d[downsample]/ret.75(AutoEncoder1D::nndct_maxpool1d_39)
        self.module_40 = py_nndct.nn.Module('aten::_convolution_mode') #AutoEncoder1D::AutoEncoder1D/UpConvBlock[upsample_blocks]/ModuleList[0]/ConvBlock[conv_block]/Conv1d[conv1d]/ret.77(AutoEncoder1D::aten__convolution_mode_40)
        self.module_41 = py_nndct.nn.Module('nndct_transpose') #AutoEncoder1D::AutoEncoder1D/UpConvBlock[upsample_blocks]/ModuleList[0]/ConvBlock[conv_block]/ret.79(AutoEncoder1D::nndct_transpose_41)
        self.module_42 = py_nndct.nn.LayerNorm(normalized_shape=[128], eps=1e-05, elementwise_affine=True) #AutoEncoder1D::AutoEncoder1D/UpConvBlock[upsample_blocks]/ModuleList[0]/ConvBlock[conv_block]/LayerNorm[norm]/ret.81(AutoEncoder1D::nndct_layer_norm_42)
        self.module_43 = py_nndct.nn.Module('nndct_transpose') #AutoEncoder1D::AutoEncoder1D/UpConvBlock[upsample_blocks]/ModuleList[0]/ConvBlock[conv_block]/ret.83(AutoEncoder1D::nndct_transpose_43)
        self.module_44 = py_nndct.nn.GELU(approximate='none') #AutoEncoder1D::AutoEncoder1D/UpConvBlock[upsample_blocks]/ModuleList[0]/ConvBlock[conv_block]/GELU[activation]/ret.85(AutoEncoder1D::nndct_GELU_44)
        self.module_45 = py_nndct.nn.MaxPool1d(kernel_size=[1], stride=[1], padding=[0], dilation=[1], ceil_mode=False) #AutoEncoder1D::AutoEncoder1D/UpConvBlock[upsample_blocks]/ModuleList[0]/ConvBlock[conv_block]/MaxPool1d[downsample]/ret.87(AutoEncoder1D::nndct_maxpool1d_45)
        self.module_46 = py_nndct.nn.Module('aten::upsample_linear1d') #AutoEncoder1D::AutoEncoder1D/UpConvBlock[upsample_blocks]/ModuleList[0]/Upsample[upsample]/ret.89(AutoEncoder1D::aten_upsample_linear1d_46)
        self.module_47 = py_nndct.nn.Cat() #AutoEncoder1D::AutoEncoder1D/ret.91(AutoEncoder1D::nndct_concat_47)
        self.module_48 = py_nndct.nn.Module('aten::_convolution_mode') #AutoEncoder1D::AutoEncoder1D/UpConvBlock[upsample_blocks]/ModuleList[1]/ConvBlock[conv_block]/Conv1d[conv1d]/ret.93(AutoEncoder1D::aten__convolution_mode_48)
        self.module_49 = py_nndct.nn.Module('nndct_transpose') #AutoEncoder1D::AutoEncoder1D/UpConvBlock[upsample_blocks]/ModuleList[1]/ConvBlock[conv_block]/ret.95(AutoEncoder1D::nndct_transpose_49)
        self.module_50 = py_nndct.nn.LayerNorm(normalized_shape=[64], eps=1e-05, elementwise_affine=True) #AutoEncoder1D::AutoEncoder1D/UpConvBlock[upsample_blocks]/ModuleList[1]/ConvBlock[conv_block]/LayerNorm[norm]/ret.97(AutoEncoder1D::nndct_layer_norm_50)
        self.module_51 = py_nndct.nn.Module('nndct_transpose') #AutoEncoder1D::AutoEncoder1D/UpConvBlock[upsample_blocks]/ModuleList[1]/ConvBlock[conv_block]/ret.99(AutoEncoder1D::nndct_transpose_51)
        self.module_52 = py_nndct.nn.GELU(approximate='none') #AutoEncoder1D::AutoEncoder1D/UpConvBlock[upsample_blocks]/ModuleList[1]/ConvBlock[conv_block]/GELU[activation]/ret.101(AutoEncoder1D::nndct_GELU_52)
        self.module_53 = py_nndct.nn.MaxPool1d(kernel_size=[1], stride=[1], padding=[0], dilation=[1], ceil_mode=False) #AutoEncoder1D::AutoEncoder1D/UpConvBlock[upsample_blocks]/ModuleList[1]/ConvBlock[conv_block]/MaxPool1d[downsample]/ret.103(AutoEncoder1D::nndct_maxpool1d_53)
        self.module_54 = py_nndct.nn.Module('aten::upsample_linear1d') #AutoEncoder1D::AutoEncoder1D/UpConvBlock[upsample_blocks]/ModuleList[1]/Upsample[upsample]/ret.105(AutoEncoder1D::aten_upsample_linear1d_54)
        self.module_55 = py_nndct.nn.Cat() #AutoEncoder1D::AutoEncoder1D/ret.107(AutoEncoder1D::nndct_concat_55)
        self.module_56 = py_nndct.nn.Module('aten::_convolution_mode') #AutoEncoder1D::AutoEncoder1D/UpConvBlock[upsample_blocks]/ModuleList[2]/ConvBlock[conv_block]/Conv1d[conv1d]/ret.109(AutoEncoder1D::aten__convolution_mode_56)
        self.module_57 = py_nndct.nn.Module('nndct_transpose') #AutoEncoder1D::AutoEncoder1D/UpConvBlock[upsample_blocks]/ModuleList[2]/ConvBlock[conv_block]/ret.111(AutoEncoder1D::nndct_transpose_57)
        self.module_58 = py_nndct.nn.LayerNorm(normalized_shape=[64], eps=1e-05, elementwise_affine=True) #AutoEncoder1D::AutoEncoder1D/UpConvBlock[upsample_blocks]/ModuleList[2]/ConvBlock[conv_block]/LayerNorm[norm]/ret.113(AutoEncoder1D::nndct_layer_norm_58)
        self.module_59 = py_nndct.nn.Module('nndct_transpose') #AutoEncoder1D::AutoEncoder1D/UpConvBlock[upsample_blocks]/ModuleList[2]/ConvBlock[conv_block]/ret.115(AutoEncoder1D::nndct_transpose_59)
        self.module_60 = py_nndct.nn.GELU(approximate='none') #AutoEncoder1D::AutoEncoder1D/UpConvBlock[upsample_blocks]/ModuleList[2]/ConvBlock[conv_block]/GELU[activation]/ret.117(AutoEncoder1D::nndct_GELU_60)
        self.module_61 = py_nndct.nn.MaxPool1d(kernel_size=[1], stride=[1], padding=[0], dilation=[1], ceil_mode=False) #AutoEncoder1D::AutoEncoder1D/UpConvBlock[upsample_blocks]/ModuleList[2]/ConvBlock[conv_block]/MaxPool1d[downsample]/ret.119(AutoEncoder1D::nndct_maxpool1d_61)
        self.module_62 = py_nndct.nn.Module('aten::upsample_linear1d') #AutoEncoder1D::AutoEncoder1D/UpConvBlock[upsample_blocks]/ModuleList[2]/Upsample[upsample]/ret.121(AutoEncoder1D::aten_upsample_linear1d_62)
        self.module_63 = py_nndct.nn.Cat() #AutoEncoder1D::AutoEncoder1D/ret.123(AutoEncoder1D::nndct_concat_63)
        self.module_64 = py_nndct.nn.Module('aten::_convolution_mode') #AutoEncoder1D::AutoEncoder1D/UpConvBlock[upsample_blocks]/ModuleList[3]/ConvBlock[conv_block]/Conv1d[conv1d]/ret.125(AutoEncoder1D::aten__convolution_mode_64)
        self.module_65 = py_nndct.nn.Module('nndct_transpose') #AutoEncoder1D::AutoEncoder1D/UpConvBlock[upsample_blocks]/ModuleList[3]/ConvBlock[conv_block]/ret.127(AutoEncoder1D::nndct_transpose_65)
        self.module_66 = py_nndct.nn.LayerNorm(normalized_shape=[32], eps=1e-05, elementwise_affine=True) #AutoEncoder1D::AutoEncoder1D/UpConvBlock[upsample_blocks]/ModuleList[3]/ConvBlock[conv_block]/LayerNorm[norm]/ret.129(AutoEncoder1D::nndct_layer_norm_66)
        self.module_67 = py_nndct.nn.Module('nndct_transpose') #AutoEncoder1D::AutoEncoder1D/UpConvBlock[upsample_blocks]/ModuleList[3]/ConvBlock[conv_block]/ret.131(AutoEncoder1D::nndct_transpose_67)
        self.module_68 = py_nndct.nn.GELU(approximate='none') #AutoEncoder1D::AutoEncoder1D/UpConvBlock[upsample_blocks]/ModuleList[3]/ConvBlock[conv_block]/GELU[activation]/ret.133(AutoEncoder1D::nndct_GELU_68)
        self.module_69 = py_nndct.nn.MaxPool1d(kernel_size=[1], stride=[1], padding=[0], dilation=[1], ceil_mode=False) #AutoEncoder1D::AutoEncoder1D/UpConvBlock[upsample_blocks]/ModuleList[3]/ConvBlock[conv_block]/MaxPool1d[downsample]/ret.135(AutoEncoder1D::nndct_maxpool1d_69)
        self.module_70 = py_nndct.nn.Module('aten::upsample_linear1d') #AutoEncoder1D::AutoEncoder1D/UpConvBlock[upsample_blocks]/ModuleList[3]/Upsample[upsample]/ret.137(AutoEncoder1D::aten_upsample_linear1d_70)
        self.module_71 = py_nndct.nn.Cat() #AutoEncoder1D::AutoEncoder1D/ret.139(AutoEncoder1D::nndct_concat_71)
        self.module_72 = py_nndct.nn.Module('aten::_convolution_mode') #AutoEncoder1D::AutoEncoder1D/UpConvBlock[upsample_blocks]/ModuleList[4]/ConvBlock[conv_block]/Conv1d[conv1d]/ret.141(AutoEncoder1D::aten__convolution_mode_72)
        self.module_73 = py_nndct.nn.Module('nndct_transpose') #AutoEncoder1D::AutoEncoder1D/UpConvBlock[upsample_blocks]/ModuleList[4]/ConvBlock[conv_block]/ret.143(AutoEncoder1D::nndct_transpose_73)
        self.module_74 = py_nndct.nn.LayerNorm(normalized_shape=[32], eps=1e-05, elementwise_affine=True) #AutoEncoder1D::AutoEncoder1D/UpConvBlock[upsample_blocks]/ModuleList[4]/ConvBlock[conv_block]/LayerNorm[norm]/ret.145(AutoEncoder1D::nndct_layer_norm_74)
        self.module_75 = py_nndct.nn.Module('nndct_transpose') #AutoEncoder1D::AutoEncoder1D/UpConvBlock[upsample_blocks]/ModuleList[4]/ConvBlock[conv_block]/ret.147(AutoEncoder1D::nndct_transpose_75)
        self.module_76 = py_nndct.nn.GELU(approximate='none') #AutoEncoder1D::AutoEncoder1D/UpConvBlock[upsample_blocks]/ModuleList[4]/ConvBlock[conv_block]/GELU[activation]/ret.149(AutoEncoder1D::nndct_GELU_76)
        self.module_77 = py_nndct.nn.MaxPool1d(kernel_size=[1], stride=[1], padding=[0], dilation=[1], ceil_mode=False) #AutoEncoder1D::AutoEncoder1D/UpConvBlock[upsample_blocks]/ModuleList[4]/ConvBlock[conv_block]/MaxPool1d[downsample]/ret.151(AutoEncoder1D::nndct_maxpool1d_77)
        self.module_78 = py_nndct.nn.Module('aten::upsample_linear1d') #AutoEncoder1D::AutoEncoder1D/UpConvBlock[upsample_blocks]/ModuleList[4]/Upsample[upsample]/ret.153(AutoEncoder1D::aten_upsample_linear1d_78)
        self.module_79 = py_nndct.nn.Cat() #AutoEncoder1D::AutoEncoder1D/ret.155(AutoEncoder1D::nndct_concat_79)
        self.module_80 = py_nndct.nn.Module('aten::_convolution_mode') #AutoEncoder1D::AutoEncoder1D/Conv1d[conv1x1_one]/ret(AutoEncoder1D::aten__convolution_mode_80)
        self.spatial_reduce_conv1d_weight = torch.nn.parameter.Parameter(torch.Tensor(32, 2480, 3))
        self.downsample_blocks_0_conv1d_weight = torch.nn.parameter.Parameter(torch.Tensor(32, 32, 7))
        self.downsample_blocks_1_conv1d_weight = torch.nn.parameter.Parameter(torch.Tensor(64, 32, 7))
        self.downsample_blocks_2_conv1d_weight = torch.nn.parameter.Parameter(torch.Tensor(64, 64, 5))
        self.downsample_blocks_3_conv1d_weight = torch.nn.parameter.Parameter(torch.Tensor(128, 64, 5))
        self.downsample_blocks_4_conv1d_weight = torch.nn.parameter.Parameter(torch.Tensor(128, 128, 5))
        self.upsample_blocks_0_conv_block_conv1d_weight = torch.nn.parameter.Parameter(torch.Tensor(128, 128, 5))
        self.upsample_blocks_1_conv_block_conv1d_weight = torch.nn.parameter.Parameter(torch.Tensor(64, 256, 5))
        self.upsample_blocks_2_conv_block_conv1d_weight = torch.nn.parameter.Parameter(torch.Tensor(64, 128, 5))
        self.upsample_blocks_3_conv_block_conv1d_weight = torch.nn.parameter.Parameter(torch.Tensor(32, 128, 7))
        self.upsample_blocks_4_conv_block_conv1d_weight = torch.nn.parameter.Parameter(torch.Tensor(32, 64, 7))
        self.conv1x1_one_weight = torch.nn.parameter.Parameter(torch.Tensor(5, 64, 1))
        self.conv1x1_one_bias = torch.nn.parameter.Parameter(torch.Tensor(5,))

    @py_nndct.nn.forward_processor
    def forward(self, *args):
        output_module_0 = self.module_0(input=args[0])
        output_module_1 = self.module_1(input=output_module_0, dim=0)
        output_module_2 = self.module_2(input=output_module_0, dim=3)
        output_module_3 = self.module_3(input=output_module_0, shape=[output_module_1,-1,output_module_2])
        output_module_3 = self.module_4({'input': output_module_3,'weight': self.spatial_reduce_conv1d_weight,'bias': None,'stride': [1],'padding': 'same','dilation': [1],'groups': 1})
        output_module_3 = self.module_5(input=output_module_3, dim0=-2, dim1=-1)
        output_module_3 = self.module_6(output_module_3)
        output_module_3 = self.module_7(input=output_module_3, dim0=-2, dim1=-1)
        output_module_3 = self.module_8(output_module_3)
        output_module_3 = self.module_9(output_module_3)
        output_module_10 = self.module_10({'input': output_module_3,'weight': self.downsample_blocks_0_conv1d_weight,'bias': None,'stride': [1],'padding': 'same','dilation': [1],'groups': 1})
        output_module_10 = self.module_11(input=output_module_10, dim0=-2, dim1=-1)
        output_module_10 = self.module_12(output_module_10)
        output_module_10 = self.module_13(input=output_module_10, dim0=-2, dim1=-1)
        output_module_10 = self.module_14(output_module_10)
        output_module_10 = self.module_15(output_module_10)
        output_module_16 = self.module_16({'input': output_module_10,'weight': self.downsample_blocks_1_conv1d_weight,'bias': None,'stride': [1],'padding': 'same','dilation': [1],'groups': 1})
        output_module_16 = self.module_17(input=output_module_16, dim0=-2, dim1=-1)
        output_module_16 = self.module_18(output_module_16)
        output_module_16 = self.module_19(input=output_module_16, dim0=-2, dim1=-1)
        output_module_16 = self.module_20(output_module_16)
        output_module_16 = self.module_21(output_module_16)
        output_module_22 = self.module_22({'input': output_module_16,'weight': self.downsample_blocks_2_conv1d_weight,'bias': None,'stride': [1],'padding': 'same','dilation': [1],'groups': 1})
        output_module_22 = self.module_23(input=output_module_22, dim0=-2, dim1=-1)
        output_module_22 = self.module_24(output_module_22)
        output_module_22 = self.module_25(input=output_module_22, dim0=-2, dim1=-1)
        output_module_22 = self.module_26(output_module_22)
        output_module_22 = self.module_27(output_module_22)
        output_module_28 = self.module_28({'input': output_module_22,'weight': self.downsample_blocks_3_conv1d_weight,'bias': None,'stride': [1],'padding': 'same','dilation': [1],'groups': 1})
        output_module_28 = self.module_29(input=output_module_28, dim0=-2, dim1=-1)
        output_module_28 = self.module_30(output_module_28)
        output_module_28 = self.module_31(input=output_module_28, dim0=-2, dim1=-1)
        output_module_28 = self.module_32(output_module_28)
        output_module_28 = self.module_33(output_module_28)
        output_module_34 = self.module_34({'input': output_module_28,'weight': self.downsample_blocks_4_conv1d_weight,'bias': None,'stride': [1],'padding': 'same','dilation': [1],'groups': 1})
        output_module_34 = self.module_35(input=output_module_34, dim0=-2, dim1=-1)
        output_module_34 = self.module_36(output_module_34)
        output_module_34 = self.module_37(input=output_module_34, dim0=-2, dim1=-1)
        output_module_34 = self.module_38(output_module_34)
        output_module_34 = self.module_39(output_module_34)
        output_module_34 = self.module_40({'input': output_module_34,'weight': self.upsample_blocks_0_conv_block_conv1d_weight,'bias': None,'stride': [1],'padding': 'same','dilation': [1],'groups': 1})
        output_module_34 = self.module_41(input=output_module_34, dim0=-2, dim1=-1)
        output_module_34 = self.module_42(output_module_34)
        output_module_34 = self.module_43(input=output_module_34, dim0=-2, dim1=-1)
        output_module_34 = self.module_44(output_module_34)
        output_module_34 = self.module_45(output_module_34)
        output_module_34 = self.module_46({'input': output_module_34,'output_size': None,'align_corners': False,'scale_factors': [2.0]})
        output_module_34 = self.module_47(dim=1, tensors=[output_module_34,output_module_28])
        output_module_34 = self.module_48({'input': output_module_34,'weight': self.upsample_blocks_1_conv_block_conv1d_weight,'bias': None,'stride': [1],'padding': 'same','dilation': [1],'groups': 1})
        output_module_34 = self.module_49(input=output_module_34, dim0=-2, dim1=-1)
        output_module_34 = self.module_50(output_module_34)
        output_module_34 = self.module_51(input=output_module_34, dim0=-2, dim1=-1)
        output_module_34 = self.module_52(output_module_34)
        output_module_34 = self.module_53(output_module_34)
        output_module_34 = self.module_54({'input': output_module_34,'output_size': None,'align_corners': False,'scale_factors': [2.0]})
        output_module_34 = self.module_55(dim=1, tensors=[output_module_34,output_module_22])
        output_module_34 = self.module_56({'input': output_module_34,'weight': self.upsample_blocks_2_conv_block_conv1d_weight,'bias': None,'stride': [1],'padding': 'same','dilation': [1],'groups': 1})
        output_module_34 = self.module_57(input=output_module_34, dim0=-2, dim1=-1)
        output_module_34 = self.module_58(output_module_34)
        output_module_34 = self.module_59(input=output_module_34, dim0=-2, dim1=-1)
        output_module_34 = self.module_60(output_module_34)
        output_module_34 = self.module_61(output_module_34)
        output_module_34 = self.module_62({'input': output_module_34,'output_size': None,'align_corners': False,'scale_factors': [2.0]})
        output_module_34 = self.module_63(dim=1, tensors=[output_module_34,output_module_16])
        output_module_34 = self.module_64({'input': output_module_34,'weight': self.upsample_blocks_3_conv_block_conv1d_weight,'bias': None,'stride': [1],'padding': 'same','dilation': [1],'groups': 1})
        output_module_34 = self.module_65(input=output_module_34, dim0=-2, dim1=-1)
        output_module_34 = self.module_66(output_module_34)
        output_module_34 = self.module_67(input=output_module_34, dim0=-2, dim1=-1)
        output_module_34 = self.module_68(output_module_34)
        output_module_34 = self.module_69(output_module_34)
        output_module_34 = self.module_70({'input': output_module_34,'output_size': None,'align_corners': False,'scale_factors': [2.0]})
        output_module_34 = self.module_71(dim=1, tensors=[output_module_34,output_module_10])
        output_module_34 = self.module_72({'input': output_module_34,'weight': self.upsample_blocks_4_conv_block_conv1d_weight,'bias': None,'stride': [1],'padding': 'same','dilation': [1],'groups': 1})
        output_module_34 = self.module_73(input=output_module_34, dim0=-2, dim1=-1)
        output_module_34 = self.module_74(output_module_34)
        output_module_34 = self.module_75(input=output_module_34, dim0=-2, dim1=-1)
        output_module_34 = self.module_76(output_module_34)
        output_module_34 = self.module_77(output_module_34)
        output_module_34 = self.module_78({'input': output_module_34,'output_size': None,'align_corners': False,'scale_factors': [2.0]})
        output_module_34 = self.module_79(dim=1, tensors=[output_module_34,output_module_3])
        output_module_34 = self.module_80({'input': output_module_34,'weight': self.conv1x1_one_weight,'bias': self.conv1x1_one_bias,'stride': [1],'padding': 'same','dilation': [1],'groups': 1})
        return output_module_34