import torch
import torch.nn as nn
import torch.nn.parallel

class ACGAN_D(nn.Module):
    def __init__(self, numOfClass, isize, nz, nc, ndf, ngpu, nshareD, n_extra_layers=0):
        super(ACGAN_D, self).__init__()
        self.ngpu = ngpu
        assert isize % 16 == 0, "isize has to be a multiple of 16"

        acgan_netD = nn.ModuleList()
        conv1 = nn.Sequential(
            nn.Conv2d(nc, 16, 3, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),
        )
        conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, 1, 0, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),
        )
        conv3 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),
        )
        conv4 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 0, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),
        )
        conv5 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),
        )
        conv6 = nn.Sequential(
            nn.Conv2d(256, 512, 3, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),
        )
        fc7 = nn.Sequential(
            nn.Linear(13*13*512, 1),
            nn.Sigmoid()
        )
        acgan_netD.append(conv1)
        acgan_netD.append(conv2)
        acgan_netD.append(conv3)
        acgan_netD.append(conv4)
        acgan_netD.append(conv5)
        acgan_netD.append(conv6)
        acgan_netD.append(fc7)

        main_share = nn.Sequential()
        discriminators = nn.ModuleList()

        # create nshareD shared module
        for i in range(numOfClass):
            main = nn.Sequential()
            for j in range(nshareD, len(acgan_netD)):
                main.add_module('acgan_netD[{}]'.format(j), acgan_netD[j])
            discriminators.append(main)

        # create n discriminators
        for i in range(nshareD):
            main_share.add_module('acgan_netD[{}](share)'.format(i), acgan_netD[i])

        self.discriminators = discriminators
        self.main_share = main_share

    def forward(self, inputList, index=None):
        if index is not None:
            index = index.data[0]
            output = self.discriminators[index](self.main_share(inputList))
            output = output.mean(0).view(1)
            return output
        else:
            outputList = []
            for i, input in enumerate(inputList):
                output = self.discriminators[i](self.main_share(input))
                output = output.mean(0)
            outputList.append(output.view(1))
            return outputList

class ACGAN_G(nn.Module):
    def __init__(self, numOfClass, isize, nz, nc, ngf, ngpu, nshareG, n_extra_layers=0):
        super(ACGAN_G, self).__init__()
        self.ngpu = ngpu
        assert isize % 16 == 0, "isize has to be a multiple of 16"

        acgan_netG = nn.ModuleList()
        fc1 = nn.Sequential(
            nn.Linear(nz, 768),
            nn.ReLU(True)
        )
        tconv2 = nn.Sequential(
            nn.ConvTranspose2d(768, 384, 5, 2, 0, bias=False),
            nn.BatchNorm2d(384),
            nn.ReLU(True)
        )
        tconv3 = nn.Sequential(
            nn.ConvTranspose2d(384, 256, 5, 2, 0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True)
        )
        tconv4 = nn.Sequential(
            nn.ConvTranspose2d(256, 192, 5, 2, 0, bias=False),
            nn.BatchNorm2d(192),
            nn.ReLU(True)
        )
        tconv5 = nn.Sequential(
            nn.ConvTranspose2d(192, 64, 5, 2, 0, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )
        tconv6 = nn.Sequential(
            nn.ConvTranspose2d(64, nc, 8, 2, 0, bias=False),
            nn.Tanh()
        )
        acgan_netG.append(fc1)
        acgan_netG.append(tconv2)
        acgan_netG.append(tconv3)
        acgan_netG.append(tconv4)
        acgan_netG.append(tconv5)
        acgan_netG.append(tconv6)

        main_share = nn.Sequential()
        generators = nn.ModuleList()

        # create n generators
        for i in range(numOfClass):
            main = nn.Sequential()
            for j in range(len(acgan_netG) - nshareG):
                main.add_module('acgan_netG[{}]'.format(j), acgan_netG[j])
            generators.append(main)

        # create nshareG shared module
        for i in range(len(acgan_netG) - nshareG, nshareG):
            main_share.add_module('acgan_netG[{}](share)'.format(i), acgan_netG[i])

        self.generators = generators
        self.main_share = main_share

    def forward(self, inputList, index=None):
        if index is not None:
            index = index.data[0]
            output = self.main_share(self.generators[index](inputList))
            return output
        else:
            outputList = []
            for i, input in enumerate(inputList):
                output = self.main_share(self.generators[i](input))
                outputList.append(output)
            return outputList
