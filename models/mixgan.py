import torch
import torch.nn as nn
import torch.nn.parallel

class DCGAN_D(nn.Module):
    def __init__(self, numOfClass, isize, nz, nc, ndf, ngpu, n_extra_layers=0):
        super(DCGAN_D, self).__init__()
        self.ngpu = ngpu
        assert isize % 16 == 0, "isize has to be a multiple of 16"

        main_share = nn.Sequential()
        discriminators = nn.ModuleList()

        main_share.add_module('initial.conv.{}-{}(share)'.format(nc, ndf),
                              nn.Conv2d(nc, ndf, 4, 2, 1, bias=False))
        main_share.add_module('initial.relu.{}(share)'.format(ndf),
                              nn.LeakyReLU(0.2, inplace=True))

        # create n discriminators
        for i in range(numOfClass):
            csize, cndf = isize / 2, ndf
            main = nn.Sequential()

            # Extra layers
            for t in range(n_extra_layers):
                main.add_module('extra-layers{}-{}.{}.conv'.format(i, t, cndf),
                                nn.Conv2d(cndf, cndf, 3, 1, 1, bias=False))
                main.add_module('extra-layers{}-{}.{}.batchnorm'.format(i, t, cndf),
                                nn.BatchNorm2d(cndf))
                main.add_module('extra-layers{}-{}.{}.relu'.format(i, t, cndf),
                                nn.LeakyReLU(0.2, inplace=True))

            while csize > 4:
                in_feat = cndf
                out_feat = cndf * 2
                main.add_module('pyramid{}.{}-{}.conv'.format(i, in_feat, out_feat),
                                nn.Conv2d(in_feat, out_feat, 4, 2, 1, bias=False))
                main.add_module('pyramid{}.{}.batchnorm'.format(i, out_feat),
                                nn.BatchNorm2d(out_feat))
                main.add_module('pyramid{}.{}.relu'.format(i, out_feat),
                                nn.LeakyReLU(0.2, inplace=True))
                cndf = cndf * 2
                csize = csize / 2

            main.add_module('final{}.{}-{}.conv'.format(i, cndf, 1),
                            nn.Conv2d(cndf, 1, 4, 1, 0, bias=False)) 

            discriminators.append(main)

        self.discriminators = discriminators
        self.main_share = main_share

    def forward(self, inputList):
        outputList = []

        for i, input in enumerate(inputList):
            output = self.discriminators[i](self.main_share(input))
            output = output.mean(0)
            outputList.append(output.view(1))

        return outputList

class DCGAN_G(nn.Module):
    def __init__(self, numOfClass, isize, nz, nc, ngf, ngpu, n_extra_layers=0):
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
            while csize < isize//2:
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
        
        main_share.add_module('final.{}-{}.convt(share)'.format(cngf, nc),
                nn.ConvTranspose2d(cngf, nc, 4, 2, 1, bias=False))
        main_share.add_module('final.{}.tanh(share)'.format(nc),
                nn.Tanh())
        
        self.generators = generators
        self.main_share = main_share

    def forward(self, inputList):
        outputList = []

        for i, input in enumerate(inputList):
            output = self.main_share(self.generators[i](input))
            outputList.append(output)

        # outputList = [self.main_share(self.generators[i](input)) for i, input in enumerate(inputList)]
        return outputList
