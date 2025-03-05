# This is where I build the brain of my AI.
# To get super sharp images, I am making a GAN (Generative Adversarial Network).
# This means I build TWO brains: an "Artist" to draw the image, and a "Critic" to judge it.

import torch
import torch.nn as nn
from torchvision.models import vgg19, VGG19_Weights

# ---------------------------------------------------------
# 1. THE ARTIST (Generator)
# This AI tries to upscale the blurry image and add sharp details.
# ---------------------------------------------------------
class ResidualDenseBlock(nn.Module):
    # A tiny math block that helps my AI remember textures without crashing my laptop.
    def __init__(self, channels=64, growth=32):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, growth, 3, 1, 1)
        self.conv2 = nn.Conv2d(channels + growth, growth, 3, 1, 1)
        self.conv3 = nn.Conv2d(channels + 2 * growth, channels, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.conv3(torch.cat((x, x1, x2), 1))
        return x3 * 0.2 + x

class LightweightSRNet(nn.Module):
    # My main Artist network. Designed to be lightweight for my 6GB RTX 4050.
    def __init__(self, in_channels=3, out_channels=3, features=64):
        super().__init__()
        self.start = nn.Conv2d(in_channels, features, 3, 1, 1)
        
        # I use 4 blocks so it learns well but stays fast.
        self.body = nn.Sequential(*[ResidualDenseBlock(features) for _ in range(4)])
        self.body_end = nn.Conv2d(features, features, 3, 1, 1)
        
        # This part makes the image 4x bigger (Upscaling).
        self.up1 = nn.Conv2d(features, features * 4, 3, 1, 1)
        self.up2 = nn.Conv2d(features, features * 4, 3, 1, 1)
        self.pixel_shuffle = nn.PixelShuffle(2)
        
        self.finish = nn.Conv2d(features, out_channels, 3, 1, 1)

    def forward(self, x):
        first_step = self.start(x)
        deep_features = self.body_end(self.body(first_step))
        added = first_step + deep_features
        
        bigger = self.pixel_shuffle(self.up1(added))
        even_bigger = self.pixel_shuffle(self.up2(bigger))
        
        return self.finish(even_bigger)

# ---------------------------------------------------------
# 2. THE CRITIC (Discriminator)
# This AI looks at an image and guesses if it is real or fake.
# ---------------------------------------------------------
class CriticNet(nn.Module):
    def __init__(self):
        super().__init__()
        # It takes an image and slowly shrinks it down to a single yes/no guess.
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1), # Cuts size in half
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # Cuts size in half again
            nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool2d(1), # Squeezes everything into 1 pixel
            nn.Conv2d(128, 1, kernel_size=1) 
        )
    def forward(self, x):
        return self.net(x).view(-1, 1) # Returns a score (1 means real, 0 means fake)

# ---------------------------------------------------------
# 3. TEXTURE LOSS (VGG Perceptual Loss)
# ---------------------------------------------------------
class FeatureLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # I bring in a pre-trained VGG model. 
        # It acts like human eyes to check if the AI's textures look natural.
        self.vgg = vgg19(weights=VGG19_Weights.DEFAULT).features[:36].eval()
        
        # I freeze its brain because I only want to use it for checking, not training.
        for param in self.vgg.parameters():
            param.requires_grad = False
            
        self.loss_math = nn.L1Loss()

    def forward(self, fake_img, real_img):
        # I compare the style and texture of both images
        return self.loss_math(self.vgg(fake_img), self.vgg(real_img))