'''Usage

python train_wavegan.py train ./train \--data_dir ./data/customdataset

Ref 
https://github.com/chrisdonahue/wavegan/blob/master/train_wavegan.py

'''

import os
import time

import numpy as np
import pytorch

import loader
from wavegan import WaveGANGenerator, WaveGANDiscriminator

'''
	Constants
'''

_FS = 16000
_WINDOW_LEN = 16384
_D_Z = 100

'''
	Train a WaveGAN
'''

def train():

	# Make z vector
	z = torch.randn(batch_size, z_dim)


'''
	Create Generator
	post-processing not implemented
'''
	G_z = WaveGANGenerator(z, train=True, **args.wavegan_g_kwargs)


'''
	Make Real and Face"
'''





'''
	Create Loss
'''


'''
	Create (recommended optimizer)
'''
	if args.wavegan_loss == 'dcgan':
		g_optimizer = torch.optim.Adam(D.parameters(), lr = 2e-4, betas = (0.5,0.9))
		d_optimizer = torch.optim.Adam(G.parameters(), lr = 2e-4, betas = (0.5,0.9)) 

	elif args.wavegan_loss == 'lsgan':
		g_optimizer = torch.optim.Adam(D.parameters(), lr = 1e-4)
		d_optimizer = torch.optim.Adam(G.parameters(), lr = 1e-4)

	elif args.wavegan_loss == 'wgan':
		g_optimizer = torch.optim.Adam(D.parameters(), lr = 5e-5)
		d_optimizer = torch.optim.Adam(G.parameters(), lr = 5e-5)

	elif args.wavegan_loss == 'wgan-gp':
		g_optimizer = torch.optim.Adam(D.parameters(), lr = 1e-4, betas = (0.5,0.9))
		d_optimizer = torch.optim.Adam(G.parameters(), lr = 1e-4, betas = (0.5,0.9))
	else:
		raise NotImplementedError()

	



def infer():


def preview():



def incept():



if __name__ == '__main__':
	import argparse
	import glob
	import sys

	parser = argparse.ArgumentParser()

	parser.add_argument('mode', type=str, choices=['train', 'preview', 'incept', 'infer'])
	parser.add_argument('train_dir', type=str,
	  help='Training directory')

	wavegan_args = parser.add_argument_group('WaveGAN')
	wavegan_args.add_argument('--wavegan_kernel_len', type=int,
	  help='Length of 1D filter kernels')
	wavegan_args.add_argument('--wavegan_dim', type=int,
	  help='Dimensionality multiplier for model of G and D')
	wavegan_args.add_argument('--wavegan_batchnorm', action='store_true', dest='wavegan_batchnorm',
	  help='Enable batchnorm')
	wavegan_args.add_argument('--wavegan_disc_nupdates', type=int,
	  help='Number of discriminator updates per generator update')
	wavegan_args.add_argument('--wavegan_loss', type=str, choices=['dcgan', 'lsgan', 'wgan', 'wgan-gp'],
	  help='Which GAN loss to use')
	wavegan_args.add_argument('--wavegan_genr_upsample', type=str, choices=['zeros', 'nn', 'lin', 'cub'],
	  help='Generator upsample strategy')
	wavegan_args.add_argument('--wavegan_genr_pp', action='store_true', dest='wavegan_genr_pp',
	  help='If set, use post-processing filter')
	wavegan_args.add_argument('--wavegan_genr_pp_len', type=int,
	  help='Length of post-processing filter for DCGAN')
	wavegan_args.add_argument('--wavegan_disc_phaseshuffle', type=int,
	  help='Radius of phase shuffle operation')

	train_args = parser.add_argument_group('Train')
	train_args.add_argument('--train_batch_size', type=int,
	  help='Batch size')
	train_args.add_argument('--train_save_secs', type=int,
	  help='How often to save model')
	train_args.add_argument('--train_summary_secs', type=int,
	  help='How often to report summaries')

	preview_args = parser.add_argument_group('Preview')
  	preview_args.add_argument('--preview_n', type=int,
      help='Number of samples to preview')

	parser.set_defaults(
	data_dir=None,
	data_first_window=False,
	wavegan_kernel_len=25,
	wavegan_dim=64,
	wavegan_batchnorm=False,
	wavegan_disc_nupdates=5,
	wavegan_loss='wgan-gp',
	wavegan_genr_upsample='zeros',
	wavegan_genr_pp=False,
	wavegan_genr_pp_len=512,
	wavegan_disc_phaseshuffle=2,
	train_batch_size=64,
	train_save_secs=300,
	train_summary_secs=120,
	preview_n=32,
	)

	args = parser.parse_args()

	# Make train dir
	if not os.path.isdir(args.train_dir):
		os.makedirs(args.train_dir)

	# Save args
	with open(os.path.join(args.train_dir, 'args.txt'), 'w') as f:
		f.write('\n'.join([str(k) + ',' + str(v) for k, v in sorted(vars(args).items(), key=lambda x: x[0])]))


	# Make model kwarg dicts
	setattr(args, 'wavegan_g_kwargs', {
		'kernel_len': args.wavegan_kernel_len,
		'dim': args.wavegan_dim,
		'use_batchnorm': args.wavegan_batchnorm,
		'upsample': args.wavegan_genr_upsample
	})
	setattr(args, 'wavegan_d_kwargs', {
		'kernel_len': args.wavegan_kernel_len,
		'dim': args.wavegan_dim,
		'use_batchnorm': args.wavegan_batchnorm,
		'phaseshuffle_rad': args.wavegan_disc_phaseshuffle
	})

	if args.mode == 'train':
		infer(args)
		train(fps, args)
	elif args.mode == 'preview':
		preview(args)
	elif args.mode == 'incept':
		incept(args)
	elif args.mode == 'infer':
		infer(args)
	else:
		raise NotImplementedError()