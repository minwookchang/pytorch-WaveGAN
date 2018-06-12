#Todo
#conv1d / transconv1d
import torch, os
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.utils import save_image
from torchvision.utils import make_grid
from torch.autograd import Variable
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

lr = 0.0001
max_epoch = 20
batch_size = 64
z_dim = 100
image_size = 64
g_conv_dim = 4
d_conv_dim = 64
log_step = 100
sample_step = 500
sample_num = 32
image_path = './data/CelebA/'
sample_path = './output/' 


#

def postProcessing(a,b,c):
    
    return


def phaseShuffle(a,b,c):

    return


def deconv(c_in, c_out, k_size, stride=2, pad=1, bn=True):
    """Custom deconvolutional layer for simplicity."""
    layers = []
    layers.append(nn.ConvTranspose2d(c_in, c_out, k_size, stride, pad))
    if bn:
        layers.append(nn.BatchNorm2d(c_out))
    return nn.Sequential(*layers)

def deconv1d(c_in, c_out, k_size, stride=4, pad=1,bn=True):
    """Custom convolutional 1d lyaer for simplicity."""
    layers = []
    layers.append(nn.ConvTranspose1d(c_in, c_out, k_size,stride,pad))
    if bn:
        layers.append(nn.BatchNorm1d(c_out))
    return nn.Sequential(*layers)

"""
  Input: [None, 100]
  Output: [None, 16384, 1]
"""

class WaveGANGenerator(nn.Module):
    """Generator containing 7 deconvolutional layers."""
    def __init__(self, z_dim=100, image_size=128, conv_dim=g_conv_dim):
        super(WaveGAN_Generator, self).__init__()
        self.fc = nn.Linear(z_dim, 256*conv_dim)
        #self.fc = deconv(z_dim, conv_dim*8, int(image_size/16), 1, 0, bn=False)
        self.deconv1 = deconv1d(conv_dim*16, conv_dim*8, 25)
        self.deconv2 = deconv1d(conv_dim*8, conv_dim*4, 25)
        self.deconv3 = deconv1d(conv_dim*4, conv_dim*2, 25)
        self.deconv4 = deconv1d(conv_dim*2, conv_dim, 25)
        self.deconv5 = deconv1d(conv_dim, 1, 25, bn=False)
        
    def forward(self, z):
        print(z.size())
#        z = z.view(z.size(0), z.size(1))      # If image_size is 64, output shape is as below.
        out = self.fc(z)                 # (?, 256d)
        print(out.size())
        out = out.view(out.size(0), 16*g_conv_dim, 16) # (?,16,16d)
        print(out.size())
        out = F.relu(out)
        print(self.deconv1(out).size())
        out = F.relu(self.deconv1(out))  # (?, 64, 8d)
        out = F.relu(self.deconv2(out))  # (?, 256, 4d)
        out = F.relu(self.deconv3(out))  # (?, 1024, 2d)
        out = F.relu(self.deconv4(out))  # (?, 4096, d)
        out = F.tanh(self.deconv5(out))  # (?, 16384, c)
        return out


"""
  Input: [None, 16384, 1]
  Output: [None] (linear output)
"""

def conv(c_in, c_out, k_size, stride=4, pad=1,bn=True):
    """Custom convolutional 1d lyaer for simplicity."""
    layers = []
    layers.append(nn.Conv2d(c_in, c_out, k_size,stride,pad))
    if bn:
        layers.append(nn.BatchNorm2d(c_out))
    return nn.Sequential(*layers)


def conv1d(c_in, c_out, k_size, stride=4, pad=1,bn=True):
    """Custom convolutional 1d lyaer for simplicity."""
    layers = []
    layers.append(nn.Conv1d(c_in, c_out, k_size,stride,pad))
    if bn:
        layers.append(nn.BatchNorm1d(c_out))
    return nn.Sequential(*layers)

class WaveGAN_Discriminator(nn.Module):
    """Discriminator containing 4 convolutional layers."""
    def __init__(self, image_size=128, conv_dim=d_conv_dim):
        super(WaveGAN_Discriminator, self).__init__()
        self.conv1 = conv1d(1, conv_dim, 25, bn=False)
        self.conv2 = conv1d(conv_dim, conv_dim*2, 25)
        self.conv3 = conv1d(conv_dim*2, conv_dim*4, 25)
        self.conv4 = conv1d(conv_dim*4, conv_dim*8, 25)
        self.conv5 = conv1d(conv_dim*8, conv_dim*16, 25)
        self.fc = nn.Linear(conv_dim*16*16,1)
#            conv(conv_dim*8, 1, int(image_size/16), 1, 0, False)
        
    def forward(self, x):                         # If image_size is 64, output shape is as below.
        out = F.leaky_relu(self.conv1(x), 0.2)    # (?, 64, 32, 32)
        out = F.leaky_relu(self.conv2(out), 0.2)  # (?, 128, 16, 16)
        out = F.leaky_relu(self.conv3(out), 0.2)  # (?, 256, 8, 8)
        out = F.leaky_relu(self.conv4(out), 0.2)  # (?, 512, 4, 4)
        out = F.leaky_relu(self.conv5(out), 0.2)  # (?, 512, 4, 4)
        out = out.view(out.size(0), 256 * d_conv_dim)
        out = F.sigmoid(self.fc(out)).squeeze()
#         out = self.fc(out).squeeze() # Least Square
        return out
