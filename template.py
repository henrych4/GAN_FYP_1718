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

import models.dcgan as dcgan
import models.mlp as mlp

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_a', required=True, help='cifar10 | lsun | imagenet | folder | lfw ')
parser.add_argument('--dataset_b')
parser.add_argument('--dataroot_a', required=True, help='path to dataset')
parser.add_argument('--dataroot_b')
parser.add_argument('--data_a')
parser.add_argument('--data_b')
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
parser.add_argument('--share_all', action='store_true')

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

if opt.dataset_a in ['imagenet', 'folder', 'lfw']:
    # folder dataset
    dataset_a = dset.ImageFolder(root=opt.dataroot_a,
                               transform=transforms.Compose([
                                   transforms.Scale(opt.imageSize),
                                   transforms.CenterCrop(opt.imageSize),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
elif opt.dataset_a == 'lsun':
    dataset_a = dset.LSUN(db_path=opt.dataroot_a, classes=[opt.data_a],
                        transform=transforms.Compose([
                            transforms.Scale(opt.imageSize),
                            transforms.CenterCrop(opt.imageSize),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                        ]))
elif opt.dataset_a == 'cifar10':
    dataset_a = dset.CIFAR10(root=opt.dataroot_a, download=False,
                           transform=transforms.Compose([
                               transforms.Scale(opt.imageSize),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
if opt.dataset_b in ['imagenet', 'folder', 'lfw']:
    # folder dataset
    dataset_b = dset.ImageFolder(root=opt.dataroot_b,
                               transform=transforms.Compose([
                                   transforms.Scale(opt.imageSize),
                                   transforms.CenterCrop(opt.imageSize),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
elif opt.dataset_b == 'lsun':
    dataset_b = dset.LSUN(db_path=opt.dataroot_b, classes=[opt.data_b],
                        transform=transforms.Compose([
                            transforms.Scale(opt.imageSize),
                            transforms.CenterCrop(opt.imageSize),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                        ]))
elif opt.dataset_b == 'cifar10':
    dataset_b = dset.CIFAR10(root=opt.dataroot_b, download=False,
                           transform=transforms.Compose([
                               transforms.Scale(opt.imageSize),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))

assert dataset_a
assert dataset_b
dataloader_a = torch.utils.data.DataLoader(dataset_a, batch_size=opt.batchSize,
                                         shuffle=True, num_workers=int(opt.workers))
dataloader_b = torch.utils.data.DataLoader(dataset_b, batch_size=opt.batchSize,
                                         shuffle=True, num_workers=int(opt.workers))

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

if opt.noBN:
    netG = dcgan.DCGAN_G_nobn(opt.imageSize, nz, nc, ngf, ngpu, n_extra_layers)
elif opt.mlp_G:
    netG = mlp.MLP_G(opt.imageSize, nz, nc, ngf, ngpu)
else:
    netG = dcgan.DCGAN_G(opt.imageSize, nz, nc, ngf, ngpu, n_extra_layers, opt.share_all)

netG.apply(weights_init)
if opt.netG != '': # load checkpoint if nnceded
    netG.load_state_dict(torch.load(opt.netG))
print(netG)

if opt.mlp_D:
    netD = mlp.MLP_D(opt.imageSize, nz, nc, ndf, ngpu)
else:
    netD_a = dcgan.DCGAN_D(opt.imageSize, nz, nc, ndf, ngpu, n_extra_layers)
    netD_b = dcgan.DCGAN_D(opt.imageSize, nz, nc, ndf, ngpu, n_extra_layers)
    netD_a.apply(weights_init)
    netD_b.apply(weights_init)

if opt.netD != '':
    netD.load_state_dict(torch.load(opt.netD))
print(netD_a)
print(netD_b)

input_a = torch.FloatTensor(opt.batchSize, nc, opt.imageSize, opt.imageSize)
input_b = torch.FloatTensor(opt.batchSize, nc, opt.imageSize, opt.imageSize)
noise1 = torch.FloatTensor(opt.batchSize, nz, 1, 1)
noise2 = torch.FloatTensor(opt.batchSize, nz, 1, 1)
fixed_noise1 = torch.FloatTensor(opt.batchSize, nz, 1, 1).normal_(0, 1)
fixed_noise2 = torch.FloatTensor(opt.batchSize, nz, 1, 1).normal_(0, 1)
one = torch.FloatTensor([1])
mone = one * -1

if opt.cuda:
    netD_a.cuda()
    netD_b.cuda()
    netG.cuda()
    input_a = input_a.cuda()
    input_b = input_b.cuda()
    one, mone = one.cuda(), mone.cuda()
    noise1, noise2= noise1.cuda(), noise2.cuda()
    fixed_noise1 = fixed_noise1.cuda()
    fixed_noise2 = fixed_noise2.cuda()

# setup optimizer
if opt.adam:
    optimizerD_a = optim.Adam(netD_a.parameters(), lr=opt.lrD, betas=(opt.beta1, 0.999))
    optimizerD_b = optim.Adam(netD_b.parameters(), lr=opt.lrD, betas=(opt.beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=opt.lrG, betas=(opt.beta1, 0.999))
else:
    optimizerD_a = optim.RMSprop(netD_a.parameters(), lr = opt.lrD)
    optimizerD_b = optim.RMSprop(netD_b.parameters(), lr = opt.lrD)
    optimizerG = optim.RMSprop(netG.parameters(), lr = opt.lrG)

gen_iterations = 0
for epoch in range(opt.niter):
    data_iter_a = iter(dataloader_a)
    data_iter_b = iter(dataloader_b)
    data_length = min(len(dataloader_a), len(dataloader_b))
    i = 0
    while i < data_length:
        ############################
        # (1) Update D network
        ###########################
        for p in netD_a.parameters(): # reset requires_grad
            p.requires_grad = True # they are set to False below in netG update
        for p in netD_b.parameters():
            p.requires_grad = True

        # train the discriminator Diters times
        if gen_iterations < 25 or gen_iterations % 500 == 0:
            Diters = 100
        else:
            Diters = opt.Diters
        j = 0
        while j < Diters and i < data_length:
            j += 1

            # clamp parameters to a cube
            for p in netD_a.parameters():
                p.data.clamp_(opt.clamp_lower, opt.clamp_upper)
            for p in netD_b.parameters():
                p.data.clamp_(opt.clamp_lower, opt.clamp_upper)

            data_a = data_iter_a.next()
            data_b = data_iter_b.next()
            i += 1

            # train with real
            real_cpu_a, _ = data_a
            real_cpu_b, _ = data_b
            netD_a.zero_grad()
            netD_b.zero_grad()
            #batch_size = real_cpu.size(0)

            if opt.cuda:
                real_cpu_a = real_cpu_a.cuda()
                real_cpu_b = real_cpu_b.cuda()
            input_a.resize_as_(real_cpu_a).copy_(real_cpu_a)
            input_b.resize_as_(real_cpu_b).copy_(real_cpu_b)
            inputv_a = Variable(input_a)
            inputv_b = Variable(input_b)

            errD_real_a = netD_a(inputv_a)
            errD_real_b = netD_b(inputv_b)
            errD_real_a.backward(one)
            errD_real_b.backward(one)

            # train with fake
            noise1.resize_(opt.batchSize, nz, 1, 1).normal_(0, 1)
            noise2.resize_(opt.batchSize, nz, 1, 1).normal_(0, 1)
            noisev1 = Variable(noise1, volatile = True) # totally freeze netG
            noisev2 = Variable(noise2, volatile = True)
            fake_a, fake_b = netG(noisev1, noisev2)
            fake_a = Variable(fake_a.data)
            fake_b = Variable(fake_b.data)
            #fake_a, fake_b = Variable(netG(noisev).data)
            inputv_a, inputv_b = fake_a, fake_b
            errD_fake_a = netD_a(inputv_a)
            errD_fake_b = netD_b(inputv_b)
            errD_fake_a.backward(mone)
            errD_fake_b.backward(mone)
            errD_a = errD_real_a - errD_fake_a
            errD_b = errD_real_b - errD_fake_b
            optimizerD_a.step()
            optimizerD_b.step()

        ############################
        # (2) Update G network
        ###########################
        for p in netD_a.parameters():
            p.requires_grad = False # to avoid computation
        for p in netD_b.parameters():
            p.requires_grad = False
        netG.zero_grad()
        # in case our last batch was the tail batch of the dataloader,
        # make sure we feed a full batch of noise
        noise1.resize_(opt.batchSize, nz, 1, 1).normal_(0, 1)
        noise2.resize_(opt.batchSize, nz, 1, 1).normal_(0, 1)
        noisev1 = Variable(noise1)
        noisev2 = Variable(noise2)
        fake_a, fake_b = netG(noisev1, noisev2)
        errG_a = netD_a(fake_a)
        errG_b = netD_b(fake_b)
        errG_a.backward(one, retain_variables=True)
        errG_b.backward(one, retain_variables=True)
        optimizerG.step()
        gen_iterations += 1

        print('[%d/%d][%d/%d][%d] Loss_D_a: %f Loss_G_a: %f Loss_D_real_a: %f Loss_D_fake_a %f'
            % (epoch, opt.niter, i, data_length, gen_iterations,
            errD_a.data[0], errG_a.data[0], errD_real_a.data[0], errD_fake_a.data[0]))
        print('[%d/%d][%d/%d][%d] Loss_D_b: %f Loss_G_b: %f Loss_D_real_b: %f Loss_D_fake_b %f'
            % (epoch, opt.niter, i, data_length, gen_iterations,
            errD_b.data[0], errG_b.data[0], errD_real_b.data[0], errD_fake_b.data[0]))

        if gen_iterations % 500 == 0:
            real_cpu_a = real_cpu_a.mul(0.5).add(0.5)
            real_cpu_b = real_cpu_b.mul(0.5).add(0.5)
            vutils.save_image(real_cpu_a, '{0}/real_samples_a.png'.format(opt.experiment))
            vutils.save_image(real_cpu_b, '{0}/real_samples_b.png'.format(opt.experiment))
            fake_a, fake_b = netG(Variable(fixed_noise1, volatile=True), Variable(fixed_noise2, volatile=True))
            fake_a.data = fake_a.data.mul(0.5).add(0.5)
            fake_b.data = fake_b.data.mul(0.5).add(0.5)
            vutils.save_image(fake_a.data, '{0}/fake_samples_a_{1}.png'.format(opt.experiment, gen_iterations))
            vutils.save_image(fake_b.data, '{0}/fake_samples_b_{1}.png'.format(opt.experiment, gen_iterations))

    # do checkpointing
    torch.save(netG.state_dict(), '{0}/netG_epoch_{1}.pth'.format(opt.experiment, epoch))
    torch.save(netD_a.state_dict(), '{0}/netD_a_epoch_{1}.pth'.format(opt.experiment, epoch))
    torch.save(netD_b.state_dict(), '{0}/netD_b_epoch_{1}.pth'.format(opt.experiment, epoch))
