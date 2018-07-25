import utils, torch, time, os, pickle
import random 
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import grad
from torch.utils import data as torchData
from dataloader import AudioLoader
import librosa

lr = 0.0001
max_epoch = 20
batch_size = 64
z_dim = 100
image_size = 64
g_conv_dim = 64
d_conv_dim = 64
log_step = 100
sample_step = 500
sample_num = 32

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

        self.pp_filter = nn.Conv1d(1, 1, 512) #pp_len : 512
        
        utils.initialize_weights(self)
        
    def forward(self, z):
#        z = z.view(z.size(0), z.size(1))      # If image_size is 64, output shape is as below.
        out = self.fc(z)                 # (?, 256d)
        out = out.view(out.size(0),16*g_conv_dim,16 ) # (?,16,16d)
        out = F.relu(out)
        out = F.relu(self.deconv1(out))  # (?, 64, 8d)
        
        out = F.relu(self.deconv2(out))  # (?, 256, 4d)
        
        out = F.relu(self.deconv3(out))  # (?, 1024, 2d)
        out = F.relu(self.deconv4(out))  # (?, 4096, d)
        out = F.tanh(self.deconv5(out))  # (?, 16384, c)
        out = self.pp_filter(F.pad(out, (256,255)))
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
        #self.dataset = args.dataset
        self.data_dir = args.data_dir
        self.isShuffle= args.isShuffle
        self.log_dir = args.log_dir
        self.gpu_mode = args.gpu_mode
        self.model_name = args.gan_type
        self.input_size = args.input_size
        self.z_dim = 100
        self.lambda_ = 10
        self.n_critic = 3 # the number of iterations of the critic per generator iteration

        #load dataset
        self.dataset = "test0"
        self.trainLoader = torchData.DataLoader(dataset = AudioLoader(self.data_dir,self.isShuffle), batch_size = self.batch_size, shuffle=self.isShuffle)
#        self.data_loader = AudioLoader(self.data_dir, self.isShuffle)#, self.input_size, self.batch_size
        #data = self.trainLoader.__iter__().__next()[0]

        # networks init
        # arguments needs to be fixed #TODO
        self.G = WaveGAN_Generator(z_dim = 100)#input_dim=self.z_dim, output_dim=data.shape[1], input_size=self.input_size)
        self.D = WaveGAN_Discriminator(n_phase = 2)#input_dim=data.shape[1], output_dim=1, input_size=self.input_size)
        self.G_optimizer = optim.Adam(self.G.parameters(), lr=args.lrG, betas=(args.beta1, args.beta2))
        self.D_optimizer = optim.Adam(self.D.parameters(), lr=args.lrD, betas=(args.beta1, args.beta2))
        
        if self.gpu_mode:
            self.G.cuda()
            self.D.cuda()

        print('---------- Networks architecture -------------')
        utils.print_network(self.G)
        utils.print_network(self.D)
        print('-----------------------------------------------')

        # fixed noise
        self.sample_z_ = torch.rand((self.batch_size, self.z_dim))
        if self.gpu_mode:
            self.sample_z_ = self.sample_z_.cuda()

    def train(self):
        self.train_hist = {}
        self.train_hist['D_loss'] = []
        self.train_hist['G_loss'] = []
        self.train_hist['per_epoch_time'] = []
        self.train_hist['total_time'] = []

        self.y_real_, self.y_fake_ = torch.ones(self.batch_size, 1), torch.zeros(self.batch_size, 1)
        if self.gpu_mode:
            self.y_real_, self.y_fake_ = self.y_real_.cuda(), self.y_fake_.cuda()

        self.D.train()
        print('training start!!')
        start_time = time.time()
        for epoch in range(self.epoch):
            self.G.train()
            epoch_start_time = time.time()
            for iter, x_ in enumerate(self.trainLoader):
                
                x_ = x_.view(x_.size()[0],1,x_.size()[1])
                
                if iter == self.trainLoader.__len__() // self.batch_size:
                    break

                z_ = torch.rand((self.batch_size, self.z_dim))
                if self.gpu_mode:
                    x_, z_ = x_.cuda(), z_.cuda()

                # update D network
                self.D_optimizer.zero_grad()


                D_real = self.D(x_)
                D_real_loss = -torch.mean(D_real)

                G_ = self.G(z_)
                D_fake = self.D(G_)
                D_fake_loss = torch.mean(D_fake)

                # gradient penalty
                alpha = torch.rand((self.batch_size, 1, 1))#,1
                if self.gpu_mode:
                    alpha = alpha.cuda()

                x_hat = alpha * x_.data + (1 - alpha) * G_.data
                x_hat.requires_grad = True
                pred_hat = self.D(x_hat)
                if self.gpu_mode:
                    gradients = grad(outputs=pred_hat, inputs=x_hat, grad_outputs=torch.ones(pred_hat.size()).cuda(),
                                 create_graph=True, retain_graph=True, only_inputs=True)[0]
                else:
                    gradients = grad(outputs=pred_hat, inputs=x_hat, grad_outputs=torch.ones(pred_hat.size()),
                                     create_graph=True, retain_graph=True, only_inputs=True)[0]

                gradient_penalty = self.lambda_ * ((gradients.view(gradients.size()[0], -1).norm(2, 1) - 1) ** 2).mean()

                D_loss = D_real_loss + D_fake_loss + gradient_penalty

                D_loss.backward()
                self.D_optimizer.step()

                if ((iter+1) % self.n_critic) == 0:
                    # update G network
                    self.G_optimizer.zero_grad()

                    G_ = self.G(z_)
                    D_fake = self.D(G_)
                    G_loss = -torch.mean(D_fake)
                    self.train_hist['G_loss'].append(G_loss.item())

                    G_loss.backward()
                    self.G_optimizer.step()

                    self.train_hist['D_loss'].append(D_loss.item())

                if ((iter + 1) % 100) == 0:
                    print("Epoch: [%2d] [%4d/%4d] D_loss: %.8f, G_loss: %.8f" %
                          ((epoch + 1), (iter + 1), self.data_loader.dataset.__len__() // self.batch_size, D_loss.item(), G_loss.item()))

            self.train_hist['per_epoch_time'].append(time.time() - epoch_start_time)
            #TODO
            with torch.no_grad():
                self.visualize_results_audio((epoch+1))

        self.train_hist['total_time'].append(time.time() - start_time)
        print("Avg one epoch time: %.2f, total %d epochs time: %.2f" % (np.mean(self.train_hist['per_epoch_time']),
              self.epoch, self.train_hist['total_time'][0]))
        print("Training finish!... save training results")

        self.save()
        #utils.generate_animation(self.result_dir + '/' + self.dataset + '/' + self.model_name + '/' + self.model_name,
        #                         self.epoch)
        print(self.train_hist)
        utils.loss_plot(self.train_hist, os.path.join(self.save_dir, self.dataset, self.model_name), self.model_name)

    def visualize_results_audio(self, epoch, fix=True):
        self.G.eval()

        if not os.path.exists(self.result_dir + '/' + self.dataset + '/' + self.model_name):
            os.makedirs(self.result_dir + '/' + self.dataset + '/' + self.model_name)

        tot_num_samples = min(self.sample_num, self.batch_size)
        image_frame_dim = int(np.floor(np.sqrt(tot_num_samples)))

        if fix:
            """ fixed noise """
            samples = self.G(self.sample_z_)
        else:
            """ random noise """
            sample_z_ = torch.rand((self.batch_size, self.z_dim))
            if self.gpu_mode:
                sample_z_ = sample_z_.cuda()

            samples = self.G(sample_z_)

        print("sample_size", samples.size())
#        print(samples.size())
        #if self.gpu_mode:
        #    samples = samples.cpu().data.numpy().transpose(0, 2, 3, 1)
        #else:
        # samples = samples.data.numpy().transpose(0, 2, 3, 1)
        samples = samples.data.numpy()
        
#        samples = (samples + 1) / 2
        librosa.output.write_wav( self.result_dir + '/' + self.dataset + '/' + self.model_name + '/' + self.model_name + '_epoch%03d' % epoch + '.wav',samples[0][0],sr=16000)
        #utils.save_images(samples[:image_frame_dim * image_frame_dim, :, :, :], [image_frame_dim, image_frame_dim],
        #self.result_dir + '/' + self.dataset + '/' + self.model_name + '/' + self.model_name + '_epoch%03d' % epoch + '.png')

    def visualize_results(self, epoch, fix=True):
        self.G.eval()

        if not os.path.exists(self.result_dir + '/' + self.dataset + '/' + self.model_name):
            os.makedirs(self.result_dir + '/' + self.dataset + '/' + self.model_name)

        tot_num_samples = min(self.sample_num, self.batch_size)
        image_frame_dim = int(np.floor(np.sqrt(tot_num_samples)))

        if fix:
            """ fixed noise """
            samples = self.G(self.sample_z_)
        else:
            """ random noise """
            sample_z_ = torch.rand((self.batch_size, self.z_dim))
            if self.gpu_mode:
                sample_z_ = sample_z_.cuda()

            samples = self.G(sample_z_)

        print(samples.size())
        if self.gpu_mode:
            samples = samples.cpu().data.numpy().transpose(0, 2, 3, 1)
        else:
            samples = samples.data.numpy().transpose(0, 2, 3, 1)

        samples = (samples + 1) / 2
        utils.save_images(samples[:image_frame_dim * image_frame_dim, :, :, :], [image_frame_dim, image_frame_dim],
                          self.result_dir + '/' + self.dataset + '/' + self.model_name + '/' + self.model_name + '_epoch%03d' % epoch + '.png')



    def save(self):
        save_dir = os.path.join(self.save_dir, self.dataset, self.model_name)

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        torch.save(self.G.state_dict(), os.path.join(save_dir, self.model_name + '_G.pkl'))
        torch.save(self.D.state_dict(), os.path.join(save_dir, self.model_name + '_D.pkl'))

        with open(os.path.join(save_dir, self.model_name + '_history.pkl'), 'wb') as f:
            pickle.dump(self.train_hist, f)

    def load(self):
        save_dir = os.path.join(self.save_dir, self.dataset, self.model_name)

        self.G.load_state_dict(torch.load(os.path.join(save_dir, self.model_name + '_G.pkl')))
        self.D.load_state_dict(torch.load(os.path.join(save_dir, self.model_name + '_D.pkl')))