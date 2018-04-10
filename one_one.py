from __future__ import print_function
import os
import sys
import argparse
import random
import time
import math
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils

import models.dcgan as dcgan
import models.mlp as mlp
import models.mix_dcgan as mix_dcgan
import datetime

startTime = time.time()
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', nargs='+', help='vector of types of input datasets: cifar10/imagenet/folder/lfw')
parser.add_argument('--datapath', nargs='+', help='vector of paths to input datasets')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--nc', type=int, default=3, help='input image channels')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--nshareD', type=int, default=1, help='number of share layer in D(1-4)')
parser.add_argument('--nshareG', type=int, default=1, help='number of share layer in G(1-4)')
parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--lrD', type=float, default=0.00005, help='learning rate for Critic, default=0.00005')
parser.add_argument('--lrG', type=float, default=0.00005, help='learning rate for Generator, default=0.00005')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--clamp_lower', type=float, default=-0.01)
parser.add_argument('--clamp_upper', type=float, default=0.01)
parser.add_argument('--Diters', type=int, default=5, help='number of D iters per each G iter')
parser.add_argument('--noBN', action='store_true', help='use batchnorm or not (only for DCGAN)')
parser.add_argument('--mlp_G', action='store_true', help='use MLP for G')
parser.add_argument('--mlp_D', action='store_true', help='use MLP for D')
parser.add_argument('--n_extra_layers', type=int, default=0, help='Number of extra layers on gen and disc')
parser.add_argument('--experiment', default=None, help='Where to store samples and models')
parser.add_argument('--adam', action='store_true', help='Whether to use adam (default is rmsprop)')
parser.add_argument('--gpu', type=int, default='0', help='which gpu to use')
parser.add_argument('--dropout', action='store_true')

opt = parser.parse_args()
print(opt)

torch.cuda.set_device(opt.gpu)

if opt.experiment is None:
    opt.experiment = 'samples'
elif opt.experiment[-1] == '/':
    opt.experiment = opt.experiment[0:-1]
if not os.path.isdir(opt.experiment):
    os.system('mkdir {0}'.format(opt.experiment))

opt.manualSeed = random.randint(1, 10000) # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

numOfClass = len(opt.dataset)
dataloaderList = []
data_length = sys.maxsize

for datasetType, dataPath in zip(opt.dataset, opt.datapath):
    if datasetType in ['imagenet', 'folder', 'lfw']:
        dataset = dset.ImageFolder(root=dataPath,
                                   transform=transforms.Compose([
                                       transforms.Scale(opt.imageSize),
                                       transforms.CenterCrop(opt.imageSize),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                   ]))
    elif datasetType == 'cifar10':
        dataset = dset.CIFAR10(root=dataPath, download=False,
                               transform=transforms.Compose([
                                   transforms.Scale(opt.imageSize),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
    assert dataset
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                             shuffle=True, num_workers=int(opt.workers), pin_memory=True)
    data_length = min(data_length, len(dataloader))
    dataloaderList.append(dataloader)

ngpu = int(opt.ngpu)
nz = int(opt.nz)
ngf = int(opt.ngf)
ndf = int(opt.ndf)
nc = int(opt.nc)
nshareG = int(opt.nshareG)
nshareD = int(opt.nshareD)
n_extra_layers = int(opt.n_extra_layers)
dropout = opt.dropout

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

netG = mix_dcgan.DCGAN_G(numOfClass, opt.imageSize, nz, nc, ngf, ngpu, nshareG, n_extra_layers)

netG.apply(weights_init)
if opt.netG != '': # load checkpoint if nnceded
    netG.load_state_dict(torch.load(opt.netG))
print(netG)

netD = mix_dcgan.DCGAN_D(numOfClass, opt.imageSize, nz, nc, ndf, ngpu, nshareD, dropout, n_extra_layers)

netD.apply(weights_init)
if opt.netD != '':
    netD.load_state_dict(torch.load(opt.netD))
print(netD)

fixedNoiseList = []
input = torch.FloatTensor(opt.batchSize, nc, opt.imageSize, opt.imageSize)
noise = torch.FloatTensor(opt.batchSize, nz, 1, 1)
fixed_noise = torch.FloatTensor(opt.batchSize, nz, 1, 1).normal_(0, 1)
one = torch.FloatTensor([1])
mone = one * -1
for i in range(numOfClass):
    fixed_noise = torch.FloatTensor(opt.batchSize, nz, 1, 1).normal_(0, 1)
    fixedNoiseList.append(fixed_noise)

if opt.cuda:
    netD.cuda()
    netG.cuda()
    input = input.cuda()
    one, mone = one.cuda(), mone.cuda()
    noise = noise.cuda()
    fixedNoiseList = [fixedNoise.cuda() for fixedNoise in fixedNoiseList]

# setup optimizer
def optimizerD_step(index):
    if opt.adam:
        optimizerD = optim.Adam([
            {'params': netD.discriminators[index].parameters(), 'lr': opt.lrD},
            {'params': netD.main_share.parameters(), 'lr': opt.lrD/numOfClass}
        ], betas=(opt.beta1, 0.999))
    else:
        optimizerD = optim.RMSprop([
            {'params': netD.discriminators[index].parameters(), 'lr': opt.lrD},
            {'params': netD.main_share.parameters(), 'lr': opt.lrD/numOfClass}
        ])
    optimizerD.step()

def optimizerG_step(index):
    if opt.adam:
        optimizerG = optim.Adam([
            {'params': netG.generators[index].parameters(), 'lr': opt.lrG},
            {'params': netG.main_share.parameters(), 'lr': opt.lrG/numOfClass}
        ], betas=(opt.beta1, 0.999))
    else:
        optimizerG = optim.RMSprop([
            {'params': netG.generators[index].parameters(), 'lr': opt.lrG},
            {'params': netG.main_share.parameters(), 'lr': opt.lrG/numOfClass}
        ])
    optimizerG.step()

gen_iterations = 0
for epoch in range(opt.niter):
    dataIterList = [iter(loader) for loader in dataloaderList]
    i = 0
    while i < data_length:
        for index, dataIter in enumerate(dataIterList):
            cur_iterations = gen_iterations // numOfClass
            cur_i = i
            ############################
            # (1) Update D network
            ###########################
            for p in netD.parameters(): # reset requires_grad
                p.requires_grad = True # they are set to False below in netG update

            # train the discriminator Diters times
            if cur_iterations < 25 or cur_iterations % 500 == 0:
                Diters = 100
            else:
                Diters = opt.Diters
            j = 0
            while j < Diters and cur_i < data_length:
                j += 1

                # clamp parameters to a cube
                for p in netD.parameters():
                    p.data.clamp_(opt.clamp_lower, opt.clamp_upper)

                data = dataIter.next()
                cur_i += 1

                # train with real
                real_cpu, _ = data
                netD.zero_grad()

                if opt.cuda:
                    real_cpu = real_cpu.cuda(async=True)
                input.resize_as_(real_cpu).copy_(real_cpu)
                inputv = Variable(input)

                errD_real = netD(inputv, Variable(torch.IntTensor([index])))
                errD_real.backward(one)

                # train with fake
                noise.resize_(opt.batchSize, nz, 1, 1).normal_(0, 1)
                noisev = Variable(noise, volatile = True) # totally freeze netG

                fake = Variable(netG(noisev, Variable(torch.IntTensor([index]))).data)
                errD_fake = netD(fake, Variable(torch.IntTensor([index])))
                errD_fake.backward(mone)

                errD = errD_real - errD_fake
                optimizerD_step(index)

            ############################
            # (2) Update G network
            ###########################
            for p in netD.parameters():
                p.requires_grad = False # to avoid computation
            netG.zero_grad()
            # in case our last batch was the tail batch of the dataloader,
            # make sure we feed a full batch of noise
            noise.resize_(opt.batchSize, nz, 1, 1).normal_(0, 1)
            noisev = Variable(noise)

            fake = netG(noisev, Variable(torch.IntTensor([index])))
            errG = netD(fake, Variable(torch.IntTensor([index])))
            errG.backward(one)
            optimizerG_step(index)
            gen_iterations += 1

            print(f'{datetime.datetime.now()}[{epoch}/{opt.niter}][{cur_i}/{data_length}][{cur_iterations}] class: {index} errD: {errD.data[0]} errG: {errG.data[0]}')

            if index == numOfClass-1:
                i = cur_i

            if cur_iterations % 500 == 0:
                real_cpu = real_cpu.mul(0.5).add(0.5)
                vutils.save_image(real_cpu, '{}/real_samples_{}.png'.format(opt.experiment, index))
                fake = netG(Variable(fixedNoiseList[index], volatile=True), Variable(torch.IntTensor([index])))
                fake.data = fake.data.mul(0.5).add(0.5)
                vutils.save_image(fake.data, '{}/fake_samples_{}_{}.png'.format(opt.experiment, index, cur_iterations))

    # do checkpointing
    if epoch % 500 == 0:
        torch.save(netG.state_dict(), '{0}/netG_epoch_{1}.pth'.format(opt.experiment, epoch))
        torch.save(netD.state_dict(), '{0}/netD_epoch_{1}.pth'.format(opt.experiment, epoch))

endTime = time.time()
minute, second = divmod(endTime - startTime, 60)
hour, minute = divmod(minute, 60)
print('Total time spent:{0}h {1}m {2}s'.format(int(hour), int(minute), round(second, 4)))
