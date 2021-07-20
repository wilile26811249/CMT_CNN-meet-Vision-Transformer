import torch
import torch.nn as nn

from cmt_module import CMTStem, Patch_Aggregate, CMTBlock


class CMT(nn.Module):
    def __init__(self,
        in_channels = 3,
        stem_channel = 32,
        cmt_channel = [46, 92, 184, 368],
        patch_channel = [46, 92, 184, 368],
        block_layer = [2, 2, 10, 2],
        R = 3.6,
        img_size = 224,
        num_class = 10
    ):
        super(CMT, self).__init__()

        size = [56, 28, 14, 7]

        # Stem layer
        self.stem = CMTStem(in_channels, stem_channel)

        # Patch Aggregation Layer
        self.patch1 = Patch_Aggregate(stem_channel, patch_channel[0])
        self.patch2 = Patch_Aggregate(patch_channel[0], patch_channel[1])
        self.patch3 = Patch_Aggregate(patch_channel[1], patch_channel[2])
        self.patch4 = Patch_Aggregate(patch_channel[2], patch_channel[3])

        # CMT Block Layer
        cmt1 = []
        for _ in range(block_layer[0]):
            cmt_layer = CMTBlock(
                    img_size = size[0],
                    stride = 8,
                    d_k = cmt_channel[0],
                    d_v = cmt_channel[0],
                    num_heads = 1,
                    R = R,
                    in_channels = patch_channel[0]
            )
            cmt1.append(cmt_layer)
        self.cmt1 = nn.Sequential(*cmt1)

        cmt2 = []
        for _ in range(block_layer[1]):
            cmt_layer = CMTBlock(
                    img_size = size[1],
                stride = 4,
                d_k = cmt_channel[1] // 2,
                d_v = cmt_channel[1] // 2,
                num_heads = 2,
                R = R,
                in_channels = patch_channel[1]
            )
            cmt2.append(cmt_layer)
        self.cmt2 = nn.Sequential(*cmt2)

        cmt3 = []
        for _ in range(block_layer[2]):
            cmt_layer = CMTBlock(
                img_size = size[2],
                stride = 2,
                d_k = cmt_channel[2] // 4,
                d_v = cmt_channel[2] // 4,
                num_heads = 4,
                R = R,
                in_channels = patch_channel[2]
            )
            cmt3.append(cmt_layer)
        self.cmt3 = nn.Sequential(*cmt3)

        cmt4 = []
        for _ in range(block_layer[3]):
            cmt_layer = CMTBlock(
                img_size = size[3],
                stride = 1,
                d_k = cmt_channel[3] // 8,
                d_v = cmt_channel[3] // 8,
                num_heads = 8,
                R = R,
                in_channels = patch_channel[3]
            )
            cmt4.append(cmt_layer)
        self.cmt4 = nn.Sequential(*cmt4)

        # Global Average Pooling
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # FC
        self.fc = nn.Sequential(
            nn.Linear(cmt_channel[3], 1280),
            nn.ReLU(inplace = True),
        )

        # Final Classifier
        self.classifier = nn.Linear(1280, num_class)


    def forward(self, x):
        x = self.stem(x)

        x = self.patch1(x)
        x = self.cmt1(x)
        print(f"After cmt1 size: {x.shape}")

        x = self.patch2(x)
        x = self.cmt2(x)
        print(f"After cmt2 size: {x.shape}")

        x = self.patch3(x)
        x = self.cmt3(x)
        print(f"After cmt3 size: {x.shape}")

        x = self.patch4(x)
        x = self.cmt4(x)
        print(f"After cmt4 size: {x.shape}")

        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = self.classifier(x)
        print(f"After classifier size: {x.shape}")
        return x


img = torch.rand(2, 3, 224, 224)

CMT_Ti = CMT(
    in_channels = 3,
    stem_channel = 16,
    cmt_channel = [46, 92, 184, 368],
    patch_channel = [46, 92, 184, 368],
    block_layer = [2, 2, 10, 2],
    R = 3.6,
    img_size = 224,
    num_class = 10
)

CMT_XS = CMT(
    in_channels = 3,
    stem_channel = 16,
    cmt_channel = [52, 104, 208, 416],
    patch_channel = [52, 104, 208, 416],
    block_layer = [3, 3, 12, 3],
    R = 3.8,
    img_size = 224,
    num_class = 10
)

CMT_S = CMT(
    in_channels = 3,
    stem_channel = 32,
    cmt_channel = [64, 128, 256, 512],
    patch_channel = [64, 128, 256, 512],
    block_layer = [3, 3, 16, 3],
    R = 4,
    img_size = 224,
    num_class = 10
)

CMT_B = CMT(
    in_channels = 3,
    stem_channel = 38,
    cmt_channel = [76, 152, 304, 608],
    patch_channel = [76, 152, 304, 608],
    block_layer = [4, 4, 20, 4],
    R = 4,
    img_size = 224,
    num_class = 10
)


# output = CMT_Ti(img)
# output = CMT_XS(img)
# output = CMT_S(img)
# output = CMT_B(img)
print(f"{sum(p.numel() for p in CMT_Ti.parameters() if p.requires_grad) / 1e6 : .2f}M")
print(f"{sum(p.numel() for p in CMT_XS.parameters() if p.requires_grad) / 1e6 : .2f}M")
print(f"{sum(p.numel() for p in CMT_S.parameters() if p.requires_grad) / 1e6 : .2f}M")
print(f"{sum(p.numel() for p in CMT_B.parameters() if p.requires_grad) / 1e6 : .2f}M")
