from einops import rearrange

import torch
import torch.nn as nn


class Conv2x2(nn.Module):
    """
    2x2 Convolution
    """
    def __init__(self, in_channels, out_channels, stride = 1):
        super(Conv2x2, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size = 2,
            stride = stride, padding = 0, bias = True
        )
        self.init_weight()

    def init_weight(self):
        nn.init.kaiming_normal_(self.conv.weight)
        self.conv.bias.data.zero_()

    def forward(self, x):
        x = self.conv(x)
        return x


class DWCONV(nn.Module):
    """
    Depthwise Convolution
    """
    def __init__(self, in_channels, out_channels, stride = 1):
        super(DWCONV, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, out_channels, kernel_size = 3,
            stride = stride, padding = 1, groups = in_channels, bias = True
        )
        self.init_weight()

    def init_weight(self):
        nn.init.kaiming_normal_(self.depthwise.weight)
        self.depthwise.bias.data.zero_()

    def forward(self, x):
        out = self.depthwise(x)
        return out


class LPU(nn.Module):
    """
    Local Perception Unit to extract local infomation.
    LPU(X) = DWConv(X) + X
    """
    def __init__(self, in_channels, out_channels):
        super(LPU, self).__init__()
        self.DWConv = DWCONV(in_channels, out_channels)

    def forward(self, x):
        x = self.DWConv(x) + x
        return x


# class LMHSA(nn.Module):
#     """
#     Lightweight Multi-head-self-attention module.

#     Inputs:
#         Q: [N, C, H, W]
#         K: [N, C, H / k, W / k]
#         V: [N, C, H / k, W / k]
#     Outputs:
#         X: [N, C, H, W]
#     """
#     def __init__(self, channels, d_k, d_v, stride, heads, dropout):
#         super(LMHSA, self).__init__()
#         self.dwconv_k = DWCONV(channels, channels, stride = stride)
#         self.dwconv_v = DWCONV(channels, channels, stride = stride)
#         self.fc_q = nn.Linear(channels, heads * d_k)
#         self.fc_k = nn.Linear(channels, heads * d_k)
#         self.fc_v = nn.Linear(channels, heads * d_v)
#         self.fc_o = nn.Linear(heads * d_v, channels)

#         self.channels = channels
#         self.d_k = d_k
#         self.d_v = d_v
#         self.stride = stride
#         self.heads = heads
#         self.dropout = dropout
#         self.scaled_factor = self.d_k ** -0.5
#         self.num_patches = (self.d_k // self.stride) ** 2
#         self.B = nn.parameter.Parameter(torch.Tensor(self.num_patches, self.d_k // self.stride, self.d_k // self.stride))

#     def forward(self, x):
#         # 10 x 64 x 56 x 56
#         batch_size, num_channel, h, w = x.shape
#         prev_q = x.clone().permute(0, 3, 2, 1).contiguous().view(batch_size, -1, num_channel)
#         prev_k = self.dwconv_k(x).permute(0, 3, 2, 1).contiguous().view(batch_size, -1, num_channel)
#         prev_v = self.dwconv_v(x).permute(0, 3, 2, 1).contiguous().view(batch_size, -1, num_channel)

#         q = self.fc_q(prev_q).view(batch_size, -1, self.d_k)
#         k = self.fc_k(prev_k).view(batch_size, -1, self.d_k)
#         v = self.fc_v(prev_v).view(batch_size, -1, self.d_k)

#         attn = torch.einsum('... i d, ... j d -> ... i j', q, k) * self.scaled_factor
#         attn = torch.softmax(attn, dim = -1)

#         result = torch.matmul(attn, v).view(batch_size, -1, self.heads * self.d_k)
#         out = self.fc_o(result).view(batch_size, num_channel, h, w)
#         return out


class LMHSA(nn.Module):
    """
    Lightweight Multi-head-self-attention module.

    Inputs:
        Q: [N, C, H, W]
        K: [N, C, H / k, W / k]
        V: [N, C, H / k, W / k]
    Outputs:
        X: [N, C, H, W]
    """
    def __init__(self, input_size, channels, d_k, d_v, stride, heads, dropout):
        super(LMHSA, self).__init__()
        self.dwconv_k = DWCONV(channels, channels, stride = stride)
        self.dwconv_v = DWCONV(channels, channels, stride = stride)
        self.fc_q = nn.Linear(channels, heads * d_k)
        self.fc_k = nn.Linear(channels, heads * d_k)
        self.fc_v = nn.Linear(channels, heads * d_v)
        self.fc_o = nn.Linear(heads * d_k, channels)

        self.channels = channels
        self.d_k = d_k
        self.d_v = d_v
        self.stride = stride
        self.heads = heads
        self.dropout = dropout
        self.scaled_factor = self.d_k ** -0.5
        self.num_patches = (self.d_k // self.stride) ** 2
        self.B = nn.parameter.Parameter(torch.Tensor(1, self.heads, input_size ** 2, (input_size // stride) ** 2), requires_grad = True)


    def forward(self, x):
        b, c, h, w = x.shape

        # Reshape
        x_reshape = x.view(b, c, h * w).permute(0, 2, 1)
        x_reshape = nn.LayerNorm(c)(x_reshape)

        # Get q, k, v
        q = self.fc_q(x_reshape)
        q = q.view(b, h * w, self.heads, self.d_k).permute(0, 2, 1, 3).contiguous()  # [b, heads, h * w, d_k]

        k = self.dwconv_k(x)
        k_b, k_c, k_h, k_w = k.shape
        k = k.view(k_b, k_c, k_h * k_w).permute(0, 2, 1).contiguous()
        k = self.fc_k(k)
        k = k.view(k_b, k_h * k_w, self.heads, self.d_k).permute(0, 2, 1, 3).contiguous()  # [b, heads, k_h * k_w, d_k]

        v = self.dwconv_v(x)
        v_b, v_c, v_h, v_w = v.shape
        v = v.view(v_b, v_c, v_h * v_w).permute(0, 2, 1).contiguous()
        v = self.fc_v(v)
        v = v.view(v_b, v_h * v_w, self.heads, self.d_v).permute(0, 2, 1, 3).contiguous() # [b, heads, v_h * v_w, d_v]

        # Attention
        attn = torch.einsum('... i d, ... j d -> ... i j', q, k) * self.scaled_factor
        attn = attn + self.B
        attn = torch.softmax(attn, dim = -1) # [b, heads, h * w, k_h * k_w]

        result = torch.matmul(attn, v).permute(0, 2, 1, 3)
        result = result.contiguous().view(b, h * w, self.heads * self.d_v)
        result = self.fc_o(result).view(b, self.channels, h, w)
        return result

class IRFFN(nn.Module):
    """
    Inverted Residual Feed-forward Network
    """
    def __init__(self, in_channels, R):
        super(IRFFN, self).__init__()
        exp_channels = int(in_channels * R)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, exp_channels, kernel_size = 1),
            nn.BatchNorm2d(exp_channels),
            nn.GELU()
        )

        self.dwconv = nn.Sequential(
            DWCONV(exp_channels, exp_channels),
            nn.BatchNorm2d(exp_channels),
            nn.GELU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(exp_channels, in_channels, 1),
            nn.BatchNorm2d(in_channels)
        )

    def forward(self, x):
        x = x + self.conv2(self.dwconv(self.conv1(x)))
        return x


class Patch_Aggregate(nn.Module):
    """
    Aggregate the patches into a single image.
    To produce the hierachical representation.

    Applied before each stage to reduce the size of intermediate features
    (2x downsampling of resolution), and project it to a larger dimension
    (2x enlargement of dimension).

    Input:
        - x: (N, C, H, W)
    Output:
        - x: (N, C * 2, H / 2, W / 2)
    """
    def __init__(self, in_channels, out_channels = None):
        super(Patch_Aggregate, self).__init__()
        if out_channels is None:
            out_channels = in_channels
        self.conv = Conv2x2(in_channels, out_channels, stride = 2)
        self.init_weight()

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.LayerNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.conv(x)
        _, c, h, w = x.size()
        x = nn.LayerNorm((c, h, w))(x)
        return x


class CMTStem(nn.Module):
    """
    Use CMTStem module to process input image and overcome the limitation of the
    non-overlapping patches.

    First past through the image with a 2x2 convolution to reduce the image size.
    Then past throught two 1x1 convolution for better local information.

    Input:
        - x: (N, 3, 256, 256)
    Output:
        - x_cmt: (N, 32, 128, 128)
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = 2, padding = 1, bias = False)
        self.gelu1 = nn.GELU()
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.gelu2 = nn.GELU()
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.gelu3 = nn.GELU()
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.init_weight()

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.conv1(x) # 32x128x128
        x = self.gelu1(x)
        x = self.bn1(x)
        x = self.conv2(x) # 32x128x128
        x = self.gelu2(x)
        x = self.bn2(x)
        x = self.conv3(x) # 32x128x128
        x = self.gelu3(x)
        x_cmt = self.bn3(x)
        return x_cmt


class CMTBlock(nn.Module):
    def __init__(self, img_size, stride, d_k, d_v, num_heads, R = 3.6, in_channels = 46):
        super(CMTBlock, self).__init__()

        # Local Perception Unit
        self.lpu = LPU(in_channels, in_channels)

        # Lightweight MHSA
        self.lmhsa = LMHSA(img_size, in_channels, d_k, d_v, stride, num_heads, 0.0)

        # Inverted Residual FFN
        self.irffn = IRFFN(in_channels, R)

    def forward(self, x):
        x = self.lpu(x)
        x = self.lmhsa(x)
        x = self.irffn(x)
        return x


# img = torch.randn(10, 3, 224, 224)
# stem = CMTStem(3, 32)
# patch1 = Patch_Aggregate(32, 46)
# lpu = LPU(46, 46)
# model = LMHSA(56, 46, 512, 512, 8, 8, 0)
# irffn = IRFFN(46, 4)

# x = stem(img)
# print(x.shape)
# x = patch1(x)
# print(x.shape)
# x = lpu(x)
# print(x.shape)
# x = model(x)
# print(x.shape)
# x = irffn(x)
# print(x.shape)