import utils, torch, time, os, pickle
import numpy as np
import torch.nn as nn
import torch.optim as optim
from dataloader import AudioLoader

# Conv2d (Batch_num , Channel, length) 
def deconv(c_in, c_out, k_size, stride=2, pad=1, bn=True):
    """Custom deconvolutional layer for simplicity."""
    layers = []
    layers.append(nn.ConvTranspose2d(c_in, c_out, k_size, stride, pad))
    if bn:
        layers.append(nn.BatchNorm2d(c_out))
    return nn.Sequential(*layers)

#deconv1d
def deconv1d(c_in, c_out, k_size, stride=4, pad=11, out_pad = 1,bn=True):
    """Custom convolutional 1d lyaer for simplicity."""
    layers = []
    layers.append(nn.ConvTranspose1d(c_in, c_out, k_size, stride, pad,out_pad))
    if bn:
        layers.append(nn.BatchNorm1d(c_out))
    return nn.Sequential(*layers)

class WaveGAN_Generator(nn.Module):
    """Generator containing 7 deconvolutional layers."""
    def __init__(self, z_dim=100, image_size=128, conv_dim=g_conv_dim):
        super(WaveGAN_Generator, self).__init__()
        self.fc = nn.Linear(z_dim, 256*conv_dim)
#        self.deconv1 = deconv1d(conv_dim*16, conv_dim*8, (25,2,2), pad = 11)
        self.deconv1 = deconv1d(conv_dim*16, conv_dim*8, k_size = 25, pad = 11, out_pad = 1)
        self.deconv2 = deconv1d(conv_dim*8, conv_dim*4, 25)
        self.deconv3 = deconv1d(conv_dim*4, conv_dim*2, 25)
        self.deconv4 = deconv1d(conv_dim*2, conv_dim, 25)
        self.deconv5 = deconv1d(conv_dim, 1, 25, bn=False)

        utils.initialize_weights(self)
        
    def forward(self, z):
#        z = z.view(z.size(0), z.size(1))      # If image_size is 64, output shape is as below.
        out = self.fc(z)                 # (?, 256d)
        print(out.size())
        out = out.view(out.size(0),16*g_conv_dim,16 ) # (?,16,16d)
        print(out.size())
        out = F.relu(out)
        out = F.relu(self.deconv1(out))  # (?, 64, 8d)
        print("a")
        print(out.size())
        
        out = F.relu(self.deconv2(out))  # (?, 256, 4d)
        
        print(out.size())
        out = F.relu(self.deconv3(out))  # (?, 1024, 2d)
        print(out.size())
        out = F.relu(self.deconv4(out))  # (?, 4096, d)
        print(out.size())
        out = F.tanh(self.deconv5(out))  # (?, 16384, c)
        print(out.size())
        return out

def conv(c_in, c_out, k_size, stride=4, pad=1,bn=True):
    """Custom convolutional 1d lyaer for simplicity."""
    layers = []
    layers.append(nn.Conv2d(c_in, c_out, k_size,stride,pad))
    if bn:
        layers.append(nn.BatchNorm2d(c_out))
    return nn.Sequential(*layers)


def conv1d(c_in, c_out, k_size, stride=4, pad=11, out_pad = 1,bn=True):
    """Custom convolutional 1d lyaer for simplicity."""
    layers = []
    layers.append(nn.Conv1d(c_in, c_out, k_size,stride,pad))
    if bn:
        layers.append(nn.BatchNorm1d(c_out))
    return nn.Sequential(*layers)

def apply_phaseshuffle(x, n_phase, pad_type = 'refelct'):
    (batch, n_channel, x_len) = x.shape
    r = random.randrange(-n_phase, n_phase+1)
    pad_l = np.maximum( r , 0)
    pad_r = np.maximum(-r, 0)
    phase_start = pad_r
    
    padding = nn.ReflectionPad2d((pad_l, pad_r, 0, 0))
#    print("phase : ", r)
    #print("pad_l, pad_r, phase_start, x_len", pad_l, pad_r, phase_start, x_len)
    #print("x.shape", x.shape)
    
    for x_ in x:

        ch_, len_ = x_.shape
        x_ = x_.reshape(1,1,ch_,len_)
        x_ = padding(x_)
        x_ = x_[:, :, :,phase_start:phase_start + len_]
        x_ = x_.reshape(ch_,len_)
    
    return x
    

#Ref DCGAN : https://github.com/InsuJeon/Hello-Generative-Model/blob/master/Day04/DCGAN/dcgan.ipynb
class WaveGAN_Discriminator(nn.Module):
    """Discriminator containing 4 convolutional layers."""
    

    def __init__(self, image_size=128, conv_dim=d_conv_dim, n_phase = 2):
        
        if n_phase > 0:
            self.phaseshuffle = lambda x: apply_phaseshuffle(x,n_phase)
        else:
            self.phaseshuffle = lambda x: x
        
        super(WaveGAN_Discriminator, self).__init__()
        self.conv1 = conv1d(1, conv_dim, 25, bn=False)
        self.conv2 = conv1d(conv_dim, conv_dim*2, 25)
        self.conv3 = conv1d(conv_dim*2, conv_dim*4, 25)
        self.conv4 = conv1d(conv_dim*4, conv_dim*8, 25)
        self.conv5 = conv1d(conv_dim*8, conv_dim*16, 25)
        self.fc = nn.Linear(conv_dim*16*16,1)
        
#            conv(conv_dim*8, 1, int(image_size/16), 1, 0, False)
        
    def forward(self, x):
        # (?, 1, 16384) -> (?, 64, 4096)
        out = F.leaky_relu(self.conv1(x), 0.2)    # (?, 64, 32, 32)
        out = self.phaseshuffle(out)
        
        # (?, 64, 4096) -> (?, 128, 1024)
        out = F.leaky_relu(self.conv2(out), 0.2)  # (?, 128, 16, 16)
        out = self.phaseshuffle(out)

        # (?, 128, 1024) -> (?, 256, 256)
        out = F.leaky_relu(self.conv3(out), 0.2)  # (?, 256, 8, 8)
        out = self.phaseshuffle(out)

        # (?, 256, 256) -> (?, 512, 64)
        out = F.leaky_relu(self.conv4(out), 0.2)  # (?, 512, 4, 4)
        out = self.phaseshuffle(out)

        # (?, 512, 64) -> (?, 1024, 16)
        out = F.leaky_relu(self.conv5(out), 0.2)  # (?, 512, 4, 4)
        out = self.phaseshuffle(out)

        # (?, 1024, 16) -> (?, 16384)
        # (?,16384) -> (?,1)
        out = out.view(out.size(0), 256 * d_conv_dim)
        out = F.sigmoid(self.fc(out))
        out = out.squeeze()
        
        return out

#WGAN-GP : https://github.com/znxlwm/pytorch-generative-model-collections/blob/master/WGAN_GP.py
class WaveGAN(object):
    def __init__(self, args):
        #parameters
        self.epoch = args.epoch
        self.sample_num = 100
        self.batch_size = args.batch_size
        self.save_dir = args.save_dir
        self.result_dir = args.result_dir
        self.dataset = args.dataset
        self.log_dir = args.log_dir
        self.gpu_mode = args.gpu_mode
        self.model_name = args.gan_type
        self.input_size = args.input_size
        self.z_dim = 62
        self.lambda_ = 10
        self.n_critic = 5               # the number of iterations of the critic per generator iteration

        #load dataset
        self.data_loader = AudioLoader(self.dataset, self.input_size, self.batch_size)
        data = self.data_loader.__iter__().__next()[0]

        # networks init
        # arguments needs to be fixed #TODO
        self.G = WaveGAN_Generator(input_dim=self.z_dim, output_dim=data.shape[1], input_size=self.input_size)
        self.D = WaveGAN_Discriminator(input_dim=data.shape[1], output_dim=1, input_size=self.input_size)
        self.G_optimizer = optim.Adam(self.G.parameters(), lr=args.lrG, betas=(args.beta1, args.beta2))
        self.D_optimizer = optim.Adam(self.D.parameters(), lr=args.lrD, betas=(args.beta1, args.beta2))

        


#This is changes













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




# Conv2d (Batch_num , Channel, length) 
def deconv(c_in, c_out, k_size, stride=2, pad=1, bn=True):
    """Custom deconvolutional layer for simplicity."""
    layers = []
    layers.append(nn.ConvTranspose2d(c_in, c_out, k_size, stride, pad))
    if bn:
        layers.append(nn.BatchNorm2d(c_out))
    return nn.Sequential(*layers)

#deconv1d
def deconv1d(c_in, c_out, k_size, stride=4, pad=11, out_pad = 1,bn=True):
    """Custom convolutional 1d lyaer for simplicity."""
    layers = []
    layers.append(nn.ConvTranspose1d(c_in, c_out, k_size, stride, pad,out_pad))
    if bn:
        layers.append(nn.BatchNorm1d(c_out))
    return nn.Sequential(*layers)

class WaveGAN_Generator(nn.Module):
    """Generator containing 7 deconvolutional layers."""
    def __init__(self, z_dim=100, conv_dim=g_conv_dim):
        super(WaveGAN_Generator, self).__init__()
        self.fc = nn.Linear(z_dim, 256*conv_dim)
#        self.deconv1 = deconv1d(conv_dim*16, conv_dim*8, (25,2,2), pad = 11)
        self.deconv1 = deconv1d(conv_dim*16, conv_dim*8, k_size = 25, pad = 11, out_pad = 1)
        self.deconv2 = deconv1d(conv_dim*8, conv_dim*4, 25)
        self.deconv3 = deconv1d(conv_dim*4, conv_dim*2, 25)
        self.deconv4 = deconv1d(conv_dim*2, conv_dim, 25)
        self.deconv5 = deconv1d(conv_dim, 1, 25, bn=False)
        
    def forward(self, z):
#        z = z.view(z.size(0), z.size(1))      # If image_size is 64, output shape is as below.
        out = self.fc(z)                 # (?, 256d)
        print(out.size())
        out = out.view(out.size(0),16*g_conv_dim,16 ) # (?,16,16d)
        print(out.size())
        out = F.relu(out)
        out = F.relu(self.deconv1(out))  # (?, 64, 8d)
        print("a")
        print(out.size())
        
        out = F.relu(self.deconv2(out))  # (?, 256, 4d)
        
        print(out.size())
        out = F.relu(self.deconv3(out))  # (?, 1024, 2d)
        print(out.size())
        out = F.relu(self.deconv4(out))  # (?, 4096, d)
        print(out.size())
        out = F.tanh(self.deconv5(out))  # (?, 16384, c)
        print(out.size())
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


def conv1d(c_in, c_out, k_size, stride=4, pad=11, out_pad = 1,bn=True):
    """Custom convolutional 1d lyaer for simplicity."""
    layers = []
    layers.append(nn.Conv1d(c_in, c_out, k_size,stride,pad))
    if bn:
        layers.append(nn.BatchNorm1d(c_out))
    return nn.Sequential(*layers)

def apply_phaseshuffle(x, n_phase, pad_type = 'refelct'):
    (batch, n_channel, x_len) = x.shape
    r = random.randrange(-n_phase, n_phase+1)
    pad_l = np.maximum( r , 0)
    pad_r = np.maximum(-r, 0)
    phase_start = pad_r
    
    padding = nn.ReflectionPad2d((pad_l, pad_r, 0, 0))
#    print("phase : ", r)
    #print("pad_l, pad_r, phase_start, x_len", pad_l, pad_r, phase_start, x_len)
    #print("x.shape", x.shape)
    
    for x_ in x:

        ch_, len_ = x_.shape
        x_ = x_.reshape(1,1,ch_,len_)
        x_ = padding(x_)
        x_ = x_[:, :, :,phase_start:phase_start + len_]
        x_ = x_.reshape(ch_,len_)
    
    return x
    

#Ref DCGAN : https://github.com/InsuJeon/Hello-Generative-Model/blob/master/Day04/DCGAN/dcgan.ipynb
class WaveGAN_Discriminator(nn.Module):
    """Discriminator containing 4 convolutional layers."""
    

    def __init__(self, image_size=128, conv_dim=d_conv_dim, n_phase = 2):
        
        if n_phase > 0:
            self.phaseshuffle = lambda x: apply_phaseshuffle(x,n_phase)
        else:
            self.phaseshuffle = lambda x: x
        
        super(WaveGAN_Discriminator, self).__init__()
        self.conv1 = conv1d(1, conv_dim, 25, bn=False)
        self.conv2 = conv1d(conv_dim, conv_dim*2, 25)
        self.conv3 = conv1d(conv_dim*2, conv_dim*4, 25)
        self.conv4 = conv1d(conv_dim*4, conv_dim*8, 25)
        self.conv5 = conv1d(conv_dim*8, conv_dim*16, 25)
        self.fc = nn.Linear(conv_dim*16*16,1)
        
#            conv(conv_dim*8, 1, int(image_size/16), 1, 0, False)
        
    def forward(self, x):
        # (?, 1, 16384) -> (?, 64, 4096)
        out = F.leaky_relu(self.conv1(x), 0.2)    # (?, 64, 32, 32)
        out = self.phaseshuffle(out)
        
        # (?, 64, 4096) -> (?, 128, 1024)
        out = F.leaky_relu(self.conv2(out), 0.2)  # (?, 128, 16, 16)
        out = self.phaseshuffle(out)

        # (?, 128, 1024) -> (?, 256, 256)
        out = F.leaky_relu(self.conv3(out), 0.2)  # (?, 256, 8, 8)
        out = self.phaseshuffle(out)

        # (?, 256, 256) -> (?, 512, 64)
        out = F.leaky_relu(self.conv4(out), 0.2)  # (?, 512, 4, 4)
        out = self.phaseshuffle(out)

        # (?, 512, 64) -> (?, 1024, 16)
        out = F.leaky_relu(self.conv5(out), 0.2)  # (?, 512, 4, 4)
        out = self.phaseshuffle(out)

        # (?, 1024, 16) -> (?, 16384)
        # (?,16384) -> (?,1)
        out = out.view(out.size(0), 256 * d_conv_dim)
        out = F.sigmoid(self.fc(out))
        out = out.squeeze()
        
        return out