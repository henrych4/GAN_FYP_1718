import torch
import torch.nn as nn
import torch.nn.parallel

class DCGAN_D(nn.Module):
    def __init__(self, numOfClass, isize, nz, nc, ndf, ngpu, nshareD, dropout, n_extra_layers=0):
        super(DCGAN_D, self).__init__()
        self.ngpu = ngpu
        assert isize % 16 == 0, "isize has to be a multiple of 16"
        osize, ondf = isize//2, ndf

        main_share = nn.Sequential()
        discriminators = nn.ModuleList()

        # create nshareD shared module
        if nshareD != 0:
            main_share.add_module('initial.conv.{}-{}(share)'.format(nc, ndf),
                    nn.Conv2d(nc, ndf, 4, 2, 1, bias=False))
            main_share.add_module('initial.relu.{}(share)'.format(ndf),
                    nn.LeakyReLU(0.2, inplace=True))
            if dropout:
                main_share.add_module('initial.dropout.{}(share)'.format(ndf),
                        nn.Dropout(0.5, inplace=True))

        while osize > isize//pow(2, nshareD):
            main_share.add_module('pyarmid{}-{}.conv(share)'.format(ondf, ondf*2),
                    nn.Conv2d(ondf, ondf*2, 4, 2, 1, bias=False))
            main_share.add_module('pyramid.{}.batchnorm(share)'.format(ondf*2),
                    nn.BatchNorm2d(ondf*2))
            main_share.add_module('pyramid.{}.relu(share)'.format(ondf*2),
                    nn.LeakyReLU(0.2, inplace=True))
            if dropout:
                main_share.add_module('pyramid.{}.dropout(share)'.format(ondf*2),
                        nn.Dropout(0.5, inplace=True))
            ondf = ondf * 2
            osize = osize / 2

        # create n discriminators
        for i in range(numOfClass):
            csize, cndf = osize, ondf
            main = nn.Sequential()

            if nshareD == 0:
                main.add_module('initial.conv.{}-{}'.format(nc, ndf),
                    nn.Conv2d(nc, ndf, 4, 2, 1, bias=False))
                main.add_module('initial.relu.{}'.format(ndf),
                    nn.LeakyReLU(0.2, inplace=True))
                if dropout:
                    main.add_module('initial.dropout.{}'.format(ndf),
                        nn.Dropout(0.5, inplace=True))

            # Extra layers
            for t in range(n_extra_layers):
                main.add_module('extra-layers{}-{}.{}.conv'.format(i, t, cndf),
                                nn.Conv2d(cndf, cndf, 3, 1, 1, bias=False))
                main.add_module('extra-layers{}-{}.{}.batchnorm'.format(i, t, cndf),
                                nn.BatchNorm2d(cndf))
                main.add_module('extra-layers{}-{}.{}.relu'.format(i, t, cndf),
                                nn.LeakyReLU(0.2, inplace=True))

            while csize > 4:
                main.add_module('pyramid{}.{}-{}.conv'.format(i, cndf, cndf*2),
                                nn.Conv2d(cndf, cndf*2, 4, 2, 1, bias=False))
                main.add_module('pyramid{}.{}.batchnorm'.format(i, cndf*2),
                                nn.BatchNorm2d(cndf*2))
                main.add_module('pyramid{}.{}.relu'.format(i, cndf*2),
                                nn.LeakyReLU(0.2, inplace=True))
                if dropout:
                    main.add_module('pyramid{}.{}.dropout'.format(i, cndf*2),
                                    nn.Dropout(0.5, inplace=True))
                cndf = cndf * 2
                csize = csize / 2

            main.add_module('final{}.{}-{}.conv'.format(i, cndf, 1),
                            nn.Conv2d(cndf, 1, 4, 1, 0, bias=False)) 

            discriminators.append(main)

        self.discriminators = discriminators
        self.main_share = main_share

    def forward(self, inputList, index=None):
        if index is not None:
            index = index.data[0]
            if self.ngpu > 1:
                output = nn.parallel.data_parallel(self.main_share, inputList, range(self.ngpu))
                output = nn.parallel.data_parallel(self.discriminators[index], output, range(self.ngpu))
                output = output.mean(0).view(1)
            else:
                output = self.discriminators[index](self.main_share(inputList))
                output = output.mean(0).view(1)
            return output
        else:
            outputList = []
            for i, input in enumerate(inputList):
                if self.ngpu > 1:
                    output = nn.parallel.data_parallel(self.main_share, input, range(self.ngpu))
                    output = nn.parallel.data_parallel(self.discriminators[i], output, range(self.ngpu))
                    output = output.mean(0).view(1)
                else:
                    output = self.discriminators[i](self.main_share(input))
                    output = output.mean(0).view(1)
                outputList.append(output)
            return outputList

class DCGAN_G(nn.Module):
    def __init__(self, numOfClass, isize, nz, nc, ngf, ngpu, nshareG, n_extra_layers=0):
        super(DCGAN_G, self).__init__()
        self.ngpu = ngpu
        assert isize % 16 == 0, "isize has to be a multiple of 16"

        main_share = nn.Sequential()
        generators = nn.ModuleList()

        # create n generators
        for i in range(numOfClass):
            cngf, tisize = ngf//2, 4
            while tisize != isize:
                cngf = cngf * 2
                tisize = tisize * 2

            main = nn.Sequential()
            main.add_module('initial{}.{}.{}.convt'.format(i, nz, cngf),
                    nn.ConvTranspose2d(nz, cngf, 4, 1, 0, bias=False))
            main.add_module('initial{}.{}.batchnorm'.format(i, cngf),
                nn.BatchNorm2d(cngf))
            main.add_module('initial{}.{}.relu'.format(i, cngf),
                    nn.ReLU(True))

            csize, cngf = 4, cngf
            while csize < isize//pow(2, nshareG):
                main.add_module('pyramid{}.{}-{}.convt'.format(i, cngf, cngf//2),
                        nn.ConvTranspose2d(cngf, cngf//2, 4, 2, 1, bias=False))
                main.add_module('pyramid{}.{}.batchnorm'.format(i, cngf//2),
                        nn.BatchNorm2d(cngf//2))
                main.add_module('pyramid{}.{}.relu'.format(i, cngf//2),
                        nn.ReLU(True))
                cngf = cngf // 2
                csize = csize * 2

            for t in range(n_extra_layers):
                main.add_module('extra-layers{}-{}.{}.conv'.format(i, t, cngf),
                        nn.Conv2d(cngf, cngf, 3, 1, 1, bias=False))
                main.add_module('extra-layers{}-{}.{}.batchnorm'.format(i, t, cngf),
                        nn.BatchNorm2d(cngf))
                main.add_module('extra-layers{}-{}.{}.relu'.format(i, t, cngf),
                        nn.ReLU(True))

            generators.append(main)

        # create nshareG shared module
        while csize < isize//2:
            main_share.add_module('pyramid.{}-{}.convt(share)'.format(cngf, cngf//2),
                    nn.ConvTranspose2d(cngf, cngf//2, 4, 2, 1, bias=False))
            main_share.add_module('pyramid{}.batchnorm(share)'.format(cngf//2),
                    nn.BatchNorm2d(cngf//2))
            main_share.add_module('pyramid{}.relu'.format(cngf//2),
                    nn.ReLU(True))
            cngf = cngf // 2
            csize = csize * 2

        if nshareG != 0:
            main_share.add_module('final.{}-{}.convt(share)'.format(cngf, nc),
                    nn.ConvTranspose2d(cngf, nc, 4, 2, 1, bias=False))
            main_share.add_module('final.{}.tanh(share)'.format(nc),
                    nn.Tanh())
        else:
            for main in generators:
                main.add_module('final.{}-{}.convt'.format(cngf, nc),
                    nn.ConvTranspose2d(cngf, nc, 4, 2, 1, bias=False))
                main.add_module('final.{}.tanh'.format(nc),
                    nn.Tanh())

        self.generators = generators
        self.main_share = main_share

    def forward(self, inputList, index=None):
        if index is not None:
            index = index.data[0]
            if self.ngpu > 1:
                output = nn.parallel.data_parallel(self.generators[index], inputList, range(self.ngpu))
                output = nn.parallel.data_parallel(self.main_share, output, range(self.ngpu))
            else:
                output = self.main_share(self.generators[index](inputList))
            return output
        else:
            outputList = []
            for i, input in enumerate(inputList):
                if self.ngpu > 1:
                    output = nn.parallel.data_parallel(self.generators[i], input, range(self.ngpu))
                    output = nn.parallel.data_parallel(self.main_share, output, range(self.ngpu))
                else:
                    output = self.main_share(self.generators[i](input))
                    outputList.append(output)
            return outputList
