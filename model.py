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
    def __init__(self, channels, d_k, d_v, stride, heads, dropout):
        super(LMHSA, self).__init__()
        self.dwconv_k = DWCONV(channels, channels, stride = stride)
        self.dwconv_v = DWCONV(channels, channels, stride = stride)
        self.fc_q = nn.Linear(channels, heads * d_k)
        self.fc_k = nn.Linear(channels, heads * d_k)
        self.fc_v = nn.Linear(channels, heads * d_v)
        self.fc_o = nn.Linear(heads * d_v, channels)

        self.channels = channels
        self.d_k = d_k
        self.d_v = d_v
        self.stride = stride
        self.heads = heads
        self.dropout = dropout
        self.scaled_factor = self.d_k ** -0.5
        self.num_patches = (self.d_k // self.stride) ** 2

    def forward(self, x):
        # 10 x 64 x 56 x 56
        batch_size, num_channel, h, w = x.shape
        prev_q = x.clone().permute(0, 3, 2, 1).contiguous().view(batch_size, -1, num_channel)
        prev_k = self.dwconv_k(x).permute(0, 3, 2, 1).contiguous().view(batch_size, -1, num_channel)
        prev_v = self.dwconv_v(x).permute(0, 3, 2, 1).contiguous().view(batch_size, -1, num_channel)

        q = self.fc_q(prev_q).view(batch_size, -1, self.d_k)
        k = self.fc_k(prev_k).view(batch_size, -1, self.d_k)
        v = self.fc_v(prev_v).view(batch_size, -1, self.d_k)

        attn = torch.einsum('... i d, ... j d -> ... i j', q, k) * self.scaled_factor
        attn = torch.softmax(attn, dim = -1)

        result = torch.matmul(attn, v).view(batch_size, -1, self.heads * self.d_k)
        out = self.fc_o(result).view(batch_size, num_channel, h, w)
        return out


class IRFFN(nn.Module):
    """
    Inverted Residual Feed-forward Network
    """
    def __init__(self):
        super(IRFFN, self).__init__()

    def forward(self, x):
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
    def __init__(self, in_channels, ln_size = 128):
        super(Patch_Aggregate, self).__init__()
        self.conv = Conv2x2(in_channels, in_channels * 2, stride = 2)
        self.ln = nn.LayerNorm(ln_size)
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
        x = self.ln(x)
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
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size = 3, stride = 2, padding = 1, bias = False)
        self.gelu1 = nn.GELU()
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.gelu2 = nn.GELU()
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.gelu3 = nn.GELU()
        self.bn3 = nn.BatchNorm2d(32)
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
    def __init__(self):
        super(CMTBlock, self).__init__()


    def forward(self, x):
        return None


class CMT(nn.Module):
    def __init__(self):
        super(CMT, self).__init__()


    def forward(self, x):
        return None


# Stem + Stage 1
def test():
    B, C, H, W = 10, 3, 224, 224
    img = torch.randn(B, C, H, W)
    print(f"Input image size: \n\t{img.shape}")

    model = CMTStem()
    patch_agg = Patch_Aggregate(32, ln_size = H // 4)
    lpu = LPU(64, 64)

    out = model(img)
    print(f"After the CMTStem: \n\t{out.shape}")

    out = patch_agg(out)
    print(f"After the Patch_Aggregate: \n\t{out.shape}")

    out = lpu(out)
    print(f"After the LPU: \n\t{out.shape}")

    # lhmsa = LMHSA(64, 512, 512, 16, 1, 0.0)
    lhmsa = LMHSA(64, 512, 512, 8, 1, 0.0)
    print(lhmsa(out).shape)


# Stage 4:
def test2():
    B, C, H, W = 10, 184, 14, 14
    img = torch.randn(B, C, H, W)
    print(f"Input image size: \n\t{img.shape}")

    patch_agg = Patch_Aggregate(184, ln_size = 7)
    lpu = LPU(368, 368)

    out = patch_agg(img)
    print(f"After the Patch_Aggregate: \n\t{out.shape}")

    out = lpu(out)
    print(f"After the LPU: \n\t{out.shape}")

    lhmsa = LMHSA(368, 512, 512, 8, 1, 0.0)
    out = lhmsa(out)
    print(out.shape)

    out = nn.AdaptiveAvgPool2d((1, 1))(out)
    print(out.shape)

    out = torch.flatten(out, 1)
    print(out.shape)

    out = nn.Linear(368, 10)(out)
    print(out.shape)

# test() and test2() are ok.
test()
print("\n")
test2()