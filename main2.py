from __future__ import print_function
import argparse
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import os
import sys

import models.dcgan as dcgan
import models.mlp as mlp
import models.mixgan as mixgan

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
parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--lrD', type=float, default=0.00005, help='learning rate for Critic, default=0.00005')
parser.add_argument('--lrG', type=float, default=0.00005, help='learning rate for Generator, default=0.00005')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda'  , action='store_true', help='enables cuda')
parser.add_argument('--ngpu'  , type=int, default=1, help='number of GPUs to use')
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

opt = parser.parse_args()
print(opt)

torch.cuda.set_device(opt.gpu)

if opt.experiment is None:
    opt.experiment = 'samples'
elif opt.experiment[-1] == '/':
    opt.experiment = opt.experiment[0:-1]

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
                                            shuffle=True, num_workers=int(opt.workers))
    data_length = min(data_length, len(dataloader))
    dataloaderList.append(dataloader)

ngpu = int(opt.ngpu)
nz = int(opt.nz)
ngf = int(opt.ngf)
ndf = int(opt.ndf)
nc = int(opt.nc)
n_extra_layers = int(opt.n_extra_layers)

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

netG = mixgan.DCGAN_G(numOfClass, opt.imageSize, nz, nc, ngf, ngpu, n_extra_layers)

netG.apply(weights_init)
if opt.netG != '': # load checkpoint if nnceded
    netG.load_state_dict(torch.load(opt.netG))
print(netG)

netD = mixgan.DCGAN_D(numOfClass, opt.imageSize, nz, nc, ndf, ngpu, n_extra_layers)

if opt.netD != '':
    netD.load_state_dict(torch.load(opt.netD))
print(netD)

input = torch.FloatTensor(opt.batchSize, nc, opt.imageSize, opt.imageSize)
noise = torch.FloatTensor(opt.batchSize, nz, 1, 1)
noiseList = [noise for i in range(numOfClass)]
fixed_noise = torch.FloatTensor(opt.batchSize, nz, 1, 1).normal_(0, 1)
fixedNoiseList = [fixed_noise for i in range(numOfClass)]
one = torch.FloatTensor([1])
mone = one * -1

if opt.cuda:
    netD.cuda()
    netG.cuda()
    input = input.cuda()
    one, mone = one.cuda(), mone.cuda()
    noise = noise.cuda()
    fixed_noise = fixed_noise.cuda()

# setup optimizer
if opt.adam:
    optimizerD = optim.Adam(netD.parameters(), lr=opt.lrD, betas=(opt.beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=opt.lrG, betas=(opt.beta1, 0.999))
else:
    optimizerD = optim.RMSprop(netD.parameters(), lr = opt.lrD)
    optimizerG = optim.RMSprop(netG.parameters(), lr = opt.lrG)

gen_iterations = 0
for epoch in range(opt.niter):
    dataIterList = [iter(loader) for loader in dataloaderList]
    i = 0
    while i < data_length:
        ############################
        # (1) Update D network
        ###########################
        for p in netD.parameters(): # reset requires_grad
            p.requires_grad = True # they are set to False below in netG update

        # train the discriminator Diters times
        if gen_iterations < 25 or gen_iterations % 500 == 0:
            Diters = 100
        else:
            Diters = opt.Diters
        j = 0
        while j < Diters and i < data_length:
            j += 1

            # clamp parameters to a cube
            for p in netD.parameters():
                p.data.clamp_(opt.clamp_lower, opt.clamp_upper)

            dataList = [dataIter.next() for dataIter in dataIterList]
            i+=1

            # train with real
            real_cpu_list = [real_cpu for real_cpu, _ in dataList]
            netD.zero_grad()

            inputvList = []
            for real_cpu in real_cpu_list:
                if opt.cuda:
                    real_cpu = real_cpu.cuda()
                input.resize_as_(real_cpu).copy_(real_cpu)
                inputv = Variable(input)
                inputvList.append(inputv)

            errD_realList = netD(inputvList)
            # update one by one or by sum of gradients
            #errD_real.backward(one) for errD_real in errD_realList
            sum(errD_realList).backward(one)

            # train with fake
            noisevList = []
            for noise in noiseList:
                noise.resize_(opt.batchSize, nz, 1, 1).normal_(0, 1)
                noisev = Variable(noise1, volatile = True) # totally freeze netG
                noisevList.append(noisev)

            fakeList = netG(noisevList)
            fakeList = [Variable(fake.data) for fake in fakeList]

            errD_fakeList = netD(fakeList)
            # update one by one or by sum of gradients
            #errD_fake.backward(mone) for errD_fake in errD_fakeList
            sum(errD_fakeList).backward(one)

            errDList = [errD_real - errD_fake for errD_real, errD_fake in zip(errD_realList, errD_fakeList)]
            optimizerD.step()

        ############################
        # (2) Update G network
        ###########################
        for p in netD.parameters():
            p.requires_grad = False # to avoid computation
        netG.zero_grad()
        # in case our last batch was the tail batch of the dataloader,
        # make sure we feed a full batch of noise
        noisevList = []
        for noise in noiseList:
            noise.resize_(opt.batchSize, nz, 1, 1).normal_(0, 1)
            noisev = Variable(noise)
            noisevList.append(noisev)

        fakeList = netG(noisevList)
        errGList = netD(fakeList)
        # update one by one or by sum of gradients
        #errG.backward(one, retrain_variables=True) for errG in errGList
        sum(errGList).backward(one, retrain_variables=True)

        optimizerG.step()
        gen_iterations += 1

        print('[{}/{}][{}/{}][{}] Loss_D: {} Loss_G: {}'.format(epoch, opt.niter, i, data_length, gen_iterations, errDList, errGList))

        if gen_iterations % 500 == 0:
            for index, real_cpu in enumerate(real_cpu_list):
                real_cpu = real_cpu.mul(0.5).add(0.5)
                vutils.save_image(real_cpu, '{}/real_samples_{}.png'.format(opt.experiment, index))
            fixedNoisevList = [Variable(fixed_noise, volatile=True) for fixed_noise in fixedNoiseList]
            fakeList = netG(fixedNoisevList)
            for index, fake in enumerate(fakeList):
                fake.data = fake.data.mul(0.5).add(0.5)
                vutils.save_image(fake.data, '{}/fake_sameples_{}_{}.png'.format(opt.experiment, index, gen_iterations))

    # do checkpointing
    torch.save(netG.state_dict(), '{0}/netG_epoch_{1}.pth'.format(opt.experiment, epoch))
    torch.save(netD.state_dict(), '{0}/netD_epoch_{1}.pth'.format(opt.experiment, epoch))
