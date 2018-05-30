import os
import sys
import time
import pickle
import torch
#import matplotlib.pyplot as plt

def getTime():
	# get current time
	# ex) 2017.12.06, 10:30 => 12061030
	now = time.localtime()
	timeText = str(now.tm_year)[-2:] + '%02d%02d%02d%02d_' % (now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min)

	return timeText

def printInfo(a):

    print('mean: ',torch.mean(a.data), ",var:",torch.var(a.data), ",max:",torch.max(a.data),",min:", torch.min(a.data))
    return


# [-1,1] -> [0,1]
def denorm(x):
	out = (x + 1) / 2
	return out.clamp(0, 1)


#Not Used
def divideList(target, size):

	# divide given target list into sub list
	# size of sub lists is 'size'
	return [target[idx:idx + size] for idx in range(0, len(target), size)]

#Not Used
def plotLossHistory(lossHistory, outPath):

	loss1 = list()
	loss2 = list()
	loss3 = list()
	idList = list()
	idx = 0

	for history in lossHistory:

		loss1.append(history[-1][0])
		loss2.append(history[-1][1])
		loss3.append(history[-1][2])
		idList.append(idx)

		idx = idx + 1

	fig, ax = plt.subplots(1, figsize = (15, 12))

	ax.plot(idList, loss1, 'ro', label = 'Loss1')
	ax.plot(idList, loss2, 'go', label = 'Loss2')
	ax.plot(idList, loss3, 'bo', label = 'Loss3')
	ax.set_title('Loss')
	ax.set_xlabel('Iteration')
	ax.set_ylabel('Loss')
	ax.legend()

	fig.savefig(os.path.join(os.path.dirname(outPath), getTime() + 'train_loss.png'))

	return fig, ax

#Not Used
def saveLossHistory(lossHistory, outPath):

	fileName = getTime() + 'loss.pickle'

	with open(os.path.join(os.path.dirname(outPath), fileName), 'wb') as fs:

		pickle.dump(lossHistory, fs)