import math
import torch
import torch.nn as nn
from mypath import Path
from torch.nn.modules.utils import _triple


class SpatioTemporalConv(nn.Module):
    r"""Applies a factored 3D convolution over an input signal composed of several input
    planes with distinct spatial and time axes, by performing a 2D convolution over the
    spatial axes to an intermediate subspace, followed by a 1D convolution over the time
    axis to produce the final output.
    Args:
        in_channels (int): Number of channels in the input tensor
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to the sides of the input during their respective convolutions. Default: 0
        bias (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``True``
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False):
        super(SpatioTemporalConv, self).__init__()

        # if ints are entered, convert them to iterables, 1 -> [1, 1, 1]
        kernel_size = _triple(kernel_size)
        stride = _triple(stride)
        padding = _triple(padding)


        self.temporal_spatial_conv = nn.Conv3d(in_channels, out_channels, kernel_size,
                                    stride=stride, padding=padding, bias=bias)
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU()


    def forward(self, x):
        x = self.bn(self.temporal_spatial_conv(x))
        x = self.relu(x)
        return x


class SpatioTemporalResBlock(nn.Module):
    r"""Single block for the ResNet network. Uses SpatioTemporalConv in
        the standard ResNet block layout (conv->batchnorm->ReLU->conv->batchnorm->sum->ReLU)

        Args:
            in_channels (int): Number of channels in the input tensor.
            out_channels (int): Number of channels in the output produced by the block.
            kernel_size (int or tuple): Size of the convolving kernels.
            downsample (bool, optional): If ``True``, the output size is to be smaller than the input. Default: ``False``
        """

    def __init__(self, in_channels, out_channels, kernel_size, downsample=False):
        super(SpatioTemporalResBlock, self).__init__()

        # If downsample == True, the first conv of the layer has stride = 2
        # to halve the residual output size, and the input x is passed
        # through a seperate 1x1x1 conv with stride = 2 to also halve it.

        # no pooling layers are used inside ResNet
        self.downsample = downsample

        # to allow for SAME padding
        padding = kernel_size // 2

        if self.downsample:
            # downsample with stride =2 the input x
            self.downsampleconv = SpatioTemporalConv(in_channels, out_channels, 1, stride=2)
            self.downsamplebn = nn.BatchNorm3d(out_channels)

            # downsample with stride = 2when producing the residual
            self.conv1 = SpatioTemporalConv(in_channels, out_channels, kernel_size, padding=padding, stride=2)
        else:
            self.conv1 = SpatioTemporalConv(in_channels, out_channels, kernel_size, padding=padding)

        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu1 = nn.ReLU()

        # standard conv->batchnorm->ReLU
        self.conv2 = SpatioTemporalConv(out_channels, out_channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.outrelu = nn.ReLU()

    def forward(self, x):
        res = self.relu1(self.bn1(self.conv1(x)))
        res = self.bn2(self.conv2(res))

        if self.downsample:
            x = self.downsamplebn(self.downsampleconv(x))

        return self.outrelu(x + res)


class SpatioTemporalResLayer(nn.Module):
    r"""Forms a single layer of the ResNet network, with a number of repeating
    blocks of same output size stacked on top of each other

        Args:
            in_channels (int): Number of channels in the input tensor.
            out_channels (int): Number of channels in the output produced by the layer.
            kernel_size (int or tuple): Size of the convolving kernels.
            layer_size (int): Number of blocks to be stacked to form the layer
            block_type (Module, optional): Type of block that is to be used to form the layer. Default: SpatioTemporalResBlock.
            downsample (bool, optional): If ``True``, the first block in layer will implement downsampling. Default: ``False``
        """

    def __init__(self, in_channels, out_channels, kernel_size, layer_size, block_type=SpatioTemporalResBlock,
                 downsample=False):

        super(SpatioTemporalResLayer, self).__init__()

        # implement the first block
        self.block1 = block_type(in_channels, out_channels, kernel_size, downsample)

        # prepare module list to hold all (layer_size - 1) blocks
        self.blocks = nn.ModuleList([])
        for i in range(layer_size - 1):
            # all these blocks are identical, and have downsample = False by default
            self.blocks += [block_type(out_channels, out_channels, kernel_size)]

    def forward(self, x):
        x = self.block1(x)
        for block in self.blocks:
            x = block(x)

        return x


class R3DNet(nn.Module):
    def __init__(self, layer_sizes, block_type=SpatioTemporalResBlock):
        super(R3DNet, self).__init__()
        self.conv1 = SpatioTemporalConv(3, 64, [3, 7, 7], stride=[1, 2, 2], padding=[1, 3, 3])
        # 添加用于早期融合的卷积层，此处示例假设早期特征通道数为64
        
        self.early_fusion_conv = nn.Conv3d(128, 64, kernel_size=1, stride=1, padding=0)
        self.conv2 = SpatioTemporalResLayer(64, 64, 3, layer_sizes[0], block_type=block_type)
        self.conv3 = SpatioTemporalResLayer(64, 128, 3, layer_sizes[1], block_type=block_type, downsample=True)
        
        
        self.conv4 = SpatioTemporalResLayer(128, 256, 3, layer_sizes[2], block_type=block_type, downsample=True)
        # 添加用于晚期融合的卷积层，此处示例假设晚期特征通道数为512
        
        self.conv5 = SpatioTemporalResLayer(256, 512, 3, layer_sizes[3], block_type=block_type, downsample=True)
        self.late_fusion_conv = nn.Conv3d(512+512, 512, kernel_size=1, stride=1, padding=0)
        self.pool = nn.AdaptiveAvgPool3d(1)

    def forward(self, x, early_fusion_feature=None, late_fusion_feature=None):
        x = self.conv1(x)
        
        if early_fusion_feature is not None:
            x = torch.cat([x, early_fusion_feature], dim=1)
            x = self.early_fusion_conv(x)
        early_feature_output = x  # 保存早期特征输出
        x = self.conv2(x)
        x = self.conv3(x)
        


        x = self.conv4(x)
        
        x = self.conv5(x)
                
        if late_fusion_feature is not None:
            x = torch.cat([x, late_fusion_feature], dim=1)
            x = self.late_fusion_conv(x)
        late_feature_output = x  # 保存晚期特征输出
        
        x = self.pool(x)
        return x.view(-1, 512), early_feature_output, late_feature_output


class R3DClassifier(nn.Module):
    def __init__(self, num_classes, layer_sizes, block_type=SpatioTemporalResBlock, pretrained=False):
        super(R3DClassifier, self).__init__()
        self.res3d = R3DNet(layer_sizes, block_type)
        self.linear = nn.Linear(512, num_classes)

        self.__init_weight()

        if pretrained:
            self.__load_pretrained_weights()

    def forward(self, x, early_fusion_feature=None, late_fusion_feature=None):
        x, early_feature_output, late_feature_output = self.res3d(x, early_fusion_feature, late_fusion_feature)
        logits = self.linear(x)
        return logits, early_feature_output, late_feature_output

    def __load_pretrained_weights(self):
        p_dict = torch.load(Path.r3dmodel_dir())

        print("Keys in the loaded pretrained weights:", p_dict.keys())

        # Get the current model's state_dict
        s_dict = self.state_dict()

        # Filter out unnecessary keys
        p_dict = {k: v for k, v in p_dict.items() if k in s_dict}
        print("Matching keys between model and pretrained weights:", p_dict.keys())

        # Update the model's state_dict with the pretrained weights
        s_dict.update(p_dict)

        # Load the updated state_dict into the model
        self.load_state_dict(s_dict)
        
    def __init_weight(self):
      for m in self.modules():
          if isinstance(m, nn.Conv3d):
              m.weight.data = nn.init.kaiming_normal_(m.weight.data.cuda())
          elif isinstance(m, nn.BatchNorm3d):
              m.weight.data = torch.ones_like(m.weight.data).cuda()
              m.bias.data = torch.zeros_like(m.bias.data).cuda()




    #def __init_weight(self):
        #for m in self.modules():
            #if isinstance(m, nn.Conv3d):
                #nn.init.kaiming_normal_(m.weight)
            #elif isinstance(m, nn.BatchNorm3d):
                #m.weight.data.fill_(1)
               # m.bias.data.zero_()

class R3DFeatureExtractor(nn.Module):
    def __init__(self, num_classes, layer_sizes, pretrained=False):
        super(R3DFeatureExtractor, self).__init__()
        # 初始化R3DNet模型
        self.r3dnet = R3DClassifier(num_classes, layer_sizes, pretrained=pretrained)
        self.r3dnet.to("cuda")

    def forward(self, x):
        # 使用R3DNet模型进行前向传播，获取分类logits和特征
        logits, early_fusion_feature, late_fusion_feature = self.r3dnet(x)
        # 根据需求返回早期和晚期特征
        return early_fusion_feature, late_fusion_feature

    def extract_features(self, x):
        # 提供一个专门的方法来提取和返回特征
        return self.forward(x)


def get_1x_lr_params(model):
    """
    This generator returns all the parameters for the conv layer of the net.
    """
    b = [model.res3d]
    for i in range(len(b)):
        for k in b[i].parameters():
            if k.requires_grad:
                yield k


def get_10x_lr_params(model):
    """
    This generator returns all the parameters for the fc layer of the net.
    """
    b = [model.linear]
    for j in range(len(b)):
        for k in b[j].parameters():
            if k.requires_grad:
                yield k

if __name__ == "__main__":
    import torch

    # 主输入：模拟一个小批量的3D视频数据，假设批量大小为1，通道数为3（RGB），时间深度为16，空间尺寸为112x112
    main_input_RGB = torch.randn(1, 3, 16, 112, 112)
    main_input_IR = torch.randn(1, 3, 16, 112, 112)

    # # 早期融合特征：模拟一些早期特征，例如来自较浅层的特征表示
    # # 假设有64个通道，时间深度为16，空间尺寸减半为56x56
    # early_fusion_feature = torch.randn(1, 64, 16, 56, 56)
    #
    # # 晚期融合特征：模拟一些晚期特征，例如来自较深层或外部模型的特征表示
    # # 假设有512个通道，时间深度为4（经过多次下采样），空间尺寸进一步减少为14x14
    # late_fusion_feature = torch.randn(1, 512, 4, 14, 14)


    feature_extractor = R3DFeatureExtractor(num_classes=101, layer_sizes=(2, 2, 2, 2), pretrained=True)

    # 提取特征
    early_feature, late_feature = feature_extractor.extract_features(main_input_IR)
    # print("Early Fusion Feature Shape:", early_feature.shape)
    # print("Late Fusion Feature Shape:", late_feature.shape)


    # 初始化R3D分类器模型，指定类别数和层级大小
    num_classes = 101  # 假设为一个具有101个类别的分类任务
    layer_sizes = (2, 2, 2, 2)  # 示例中的层级大小

    model = R3DClassifier(num_classes=num_classes, layer_sizes=layer_sizes,pretrained=True)
    print(model)
    # 使用模型进行前向传播，传入主输入、早期和晚期融合特征
    logits, early_output, late_output = model(main_input_RGB, early_feature, late_feature)

    # # 打印输出以验证
    # print(f"Logits shape: {logits.shape}")  # 预期形状：(1, 101)，对应于批量大小和类别数
    # print(f"Early fusion feature output shape: {early_output.shape}")  # 早期特征输出的形状
    # print(f"Late fusion feature output shape: {late_output.shape}")  # 晚期特征输出diyi的形状




