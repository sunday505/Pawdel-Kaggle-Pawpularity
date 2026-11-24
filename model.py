import torch
from torch import nn

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

# class MBCONV(nn.Module):
#     def __init__(self, in_ch, out_ch, k, t, s, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.relu6 = nn.ReLU6()
#         self.flatten = nn.Flatten()
#         self.conv1 = nn.Conv2d(in_ch, t * in_ch, 1)
#         self.dwise = nn.Conv2d(t * in_ch, t * in_ch, kernel_size = k, stride = s, groups = t * in_ch)
#         self.conv2 = nn.Conv2d(t * in_ch, t * in_ch, 1)
#         self.linear = nn.Linear(t * in_ch, out_ch)
#         self.se = SEBlock(out_ch)

#     def forward(self, x):
#         x = self.relu6(self.conv1(x))
#         x = self.relu6(self.dwise(x))
#         x = self.flatten(self.conv2(x))
#         x = self.linear(x)
#         x = self.se(x)
#         return x

class MBCONV(nn.Module):
    def __init__(self, in_ch, out_ch, k, t, s):
        super().__init__()
        mid = in_ch * t

        self.block = nn.Sequential(
            # bias false
            nn.Conv2d(in_ch, mid, 1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU6(inplace=True),

            nn.Conv2d(mid, mid, k, stride=s, padding=k//2, groups=mid, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU6(inplace=True),

            SEBlock(mid),

            nn.Conv2d(mid, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            # SEBlock(out_ch)
        )

    def forward(self, x):
        out = self.block(x)
        if out.shape == x.shape:
            return x + out
        return out

class PAWDEL(nn.Module):
    def __init__(self, use_SI : bool = True):
        super().__init__()

        self.use_SI = use_SI

        # input resolution is 224 x 224

        # Side Information
        self.side = nn.Sequential(
            nn.Linear(12, 128),
            nn.ReLU(),
            nn.Linear(128, 3),
            nn.Sigmoid()
        )

        # EfficientNet

        scale = 1

        self.conv1_1 = nn.Conv2d(3, 32 * scale, kernel_size = 3, stride = 2)

        self.mbconv1_2 = MBCONV(32 * scale, 16 * scale, k = 3, t = 1, s=1)
        
        self.mbconv6_3 = self.mbconv_stage(16 * scale, 24 * scale, k = 3, t = 6, s = 2, n = 2)

        self.mbconv6_4 = self.mbconv_stage(24 * scale, 40 * scale, k = 5, t = 6, s = 2, n = 2)


        self.mbconv6_5 = self.mbconv_stage(40 * scale, 80 * scale, k = 3, t = 6, s = 2, n = 3)


        self.mbconv6_6 = self.mbconv_stage(80 * scale, 112 * scale, k = 5, t = 6, s = 1, n = 3)


        self.mbconv6_7 = self.mbconv_stage(112 * scale, 192 * scale, k = 5, t = 6, s = 2, n = 4)


        self.mbconv6_8 = self.mbconv_stage(192 * scale, 320 * scale, k = 3, t = 6, s = 1, n = 1)

        self.stage_9 = nn.Sequential(
            nn.Conv2d(320 * scale, 1280 * scale, 1),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(1280 * scale, 1),
        )
        
    def mbconv_stage(self, in_ch, out_ch, k, t, s, n):
        layers = []

        layers.append(MBCONV(in_ch, out_ch, k=k, t=t, s=s))

        for _ in range(1, n):
            layers.append(MBCONV(out_ch, out_ch, k=k, t=t, s=1))

        return nn.Sequential(*layers)

    def forward(self, x1, x2):
        if self.use_SI:
            B, C, H, W = x2.shape
            alpha = self.side(x1).view(B, C, 1, 1)
            x2 = x2 * alpha
        x2 = self.conv1_1(x2)
        x2 = self.mbconv1_2(x2)
        x2 = self.mbconv6_3(x2)
        x2 = self.mbconv6_4(x2)
        x2 = self.mbconv6_5(x2)
        x2 = self.mbconv6_6(x2)
        x2 = self.mbconv6_7(x2)
        x2 = self.mbconv6_8(x2)
        x2 = self.stage_9(x2)
        return x2
    
if __name__ == "__main__":
    pawdel = PAWDEL()
    print(pawdel)