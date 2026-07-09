import torch
import torch.nn as nn

# Helper: conv -> norm -> activation
def conv_block(in_c, out_c, kernel=4, stride=2, padding=1, norm=True, activation=nn.LeakyReLU(0.2, True)):
    layers = [nn.Conv2d(in_c, out_c, kernel, stride, padding, bias=not norm)]
    if norm:
        layers.append(nn.BatchNorm2d(out_c))
    if activation:
        layers.append(activation)
    return nn.Sequential(*layers)

# Helper: transpose conv block for decoder
def deconv_block(in_c, out_c, kernel=4, stride=2, padding=1, dropout=False):
    layers = [nn.ConvTranspose2d(in_c, out_c, kernel, stride, padding, bias=False),
              nn.BatchNorm2d(out_c),
              nn.ReLU(True)]
    if dropout:
        layers.append(nn.Dropout(0.5))
    return nn.Sequential(*layers)


class UNetGenerator(nn.Module):
    """
    UNet adjusted for 128x128 input. Channel counts are chosen so that every
    concatenation matches the next deconv's expected in_channels.
    """
    def __init__(self, in_channels=3, out_channels=3, ngf=64):
        super().__init__()
        # Encoder
        self.enc1 = conv_block(in_channels, ngf, norm=False, activation=nn.LeakyReLU(0.2, True))   # 128 -> 64
        self.enc2 = conv_block(ngf, ngf*2)  # 64 -> 32
        self.enc3 = conv_block(ngf*2, ngf*4) # 32 -> 16
        self.enc4 = conv_block(ngf*4, ngf*8) # 16 -> 8
        self.enc5 = conv_block(ngf*8, ngf*8) # 8 -> 4
        self.enc6 = conv_block(ngf*8, ngf*8) # 4 -> 2
        self.enc7 = conv_block(ngf*8, ngf*8, norm=False)  # 2 -> 1 (bottleneck)

        # Decoder
        # After each concat the in_channels for the next decoder are:
        # dec1_out = 512 -> cat with enc6 (512) => 1024  (dec2_in)
        # dec2_out = 512 -> cat with enc5 (512) => 1024  (dec3_in)
        # dec3_out = 512 -> cat with enc4 (512) => 1024  (dec4_in)
        # dec4_out = 512 -> cat with enc3 (256) => 768   (dec5_in)
        # dec5_out = 256 -> cat with enc2 (128) => 384   (dec6_in)
        # dec6_out = 128 -> cat with enc1 (64)  => 192   (dec7_in)
        self.dec1 = deconv_block(ngf*8, ngf*8, dropout=True)          # bottleneck -> 2
        self.dec2 = deconv_block(ngf*16, ngf*8, dropout=True)         # d1 (512) + enc6 (512) -> 512 out
        self.dec3 = deconv_block(ngf*16, ngf*8, dropout=True)
        self.dec4 = deconv_block(ngf*16, ngf*8)
        self.dec5 = deconv_block(ngf*12, ngf*4)   # 768 -> 256
        self.dec6 = deconv_block(ngf*6, ngf*2)    # 384 -> 128
        # final: after concatenating with enc1, we have 192 channels -> project to out_channels
        self.dec7 = nn.Sequential(
            nn.ConvTranspose2d(ngf*3, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        e1 = self.enc1(x)   # channels ngf
        e2 = self.enc2(e1)  # channels ngf*2
        e3 = self.enc3(e2)  # channels ngf*4
        e4 = self.enc4(e3)  # channels ngf*8
        e5 = self.enc5(e4)  # channels ngf*8
        e6 = self.enc6(e5)  # channels ngf*8
        e7 = self.enc7(e6)  # channels ngf*8 (bottleneck)

        d1 = self.dec1(e7)            # out ngf*8
        d1 = torch.cat([d1, e6], dim=1)  # -> ngf*16
        d2 = self.dec2(d1)            # out ngf*8
        d2 = torch.cat([d2, e5], dim=1)  # -> ngf*16
        d3 = self.dec3(d2)            # out ngf*8
        d3 = torch.cat([d3, e4], dim=1)  # -> ngf*16
        d4 = self.dec4(d3)            # out ngf*8
        d4 = torch.cat([d4, e3], dim=1)  # -> ngf*12 (512 + 256 = 768)
        d5 = self.dec5(d4)            # out ngf*4 (256)
        d5 = torch.cat([d5, e2], dim=1)  # -> ngf*6 (256 + 128 = 384)
        d6 = self.dec6(d5)            # out ngf*2 (128)
        d6 = torch.cat([d6, e1], dim=1)  # -> ngf*3 (128 + 64 = 192)
        out = self.dec7(d6)           # produce final RGB image
        return out


class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels=6, ndf=64):
        super().__init__()
        # input is (input_image + target_image) concatenated along channel dim
        self.layer1 = conv_block(in_channels, ndf, norm=False)
        self.layer2 = conv_block(ndf, ndf*2)
        self.layer3 = conv_block(ndf*2, ndf*4)
        # keep stride=1 in the last conv for reasonable patch size on 128x128
        self.layer4 = conv_block(ndf*4, ndf*8, stride=1, padding=1)
        self.final = nn.Conv2d(ndf*8, 1, kernel_size=4, stride=1, padding=1)  # Patch output

    def forward(self, inp, target):
        x = torch.cat([inp, target], dim=1)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.final(x)
        return x
