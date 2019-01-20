##################################################################
# Reproducing the paper                                          #
# ENet - Real Time Semantic Segmentation                         #
# Paper: https://arxiv.org/pdf/1606.02147.pdf                    #
#                                                                #
# Copyright (c) 2019                                             #
# Authors: @iArunava <iarunavaofficial@gmail.com>                #
#          @AvivSham <mista2311@gmail.com>                       #
#                                                                #
# License: BSD License 3.0                                       #
#                                                                #
# The Code in this file is distributed for free                  #
# usage and modification with proper credits                     #
# directing back to this repository.                             #
##################################################################

class ENet(nn.Module):
    def __init__(self, C):
        super().__init__()
        
        # Define class variables
        self.C = C
        
        self.init = InitialBlock()
        
        self.b10 = RDDNeck(256, 256, 1, 16, 64, True, 0.01)
        self.b11 = RDDNeck(128, 128, 1, 64, 64, False, 0.01)
        self.b12 = RDDNeck(128, 128, 1, 64, 64, False, 0.01)
        self.b13 = RDDNeck(128, 128, 1, 64, 64, False, 0.01)
        self.b14 = RDDNeck(128, 128, 1, 64, 64, False, 0.01)
        
        self.b20 = RDDNeck(128, 128, 1, 64, 128, True)
        self.b21 = RDDNeck(64, 64, 1, 128, 128, False)
        self.b22 = RDDNeck(64, 64, 2, 128, 128, False)
        self.b23 = ASNeck(64, 64, 128, 128)
        self.b24 = RDDNeck(64, 64, 4, 128, 128, False)
        self.b25 = RDDNeck(64, 64, 1, 128, 128, False)
        self.b26 = RDDNeck(64, 64, 8, 128, 128, False)
        self.b27 = ASNeck(64, 64, 128, 128)
        self.b28 = RDDNeck(64, 64, 16, 128, 128, False)
        
        self.b31 = RDDNeck(64, 64, 1, 128, 128, False)
        self.b32 = RDDNeck(64, 64, 2, 128, 128, False)
        self.b33 = ASNeck(64, 64, 128, 128)
        self.b34 = RDDNeck(64, 64, 4, 128, 128, False)
        self.b35 = RDDNeck(64, 64, 1, 128, 128, False)
        self.b36 = RDDNeck(64, 64, 8, 128, 128, False)
        self.b37 = ASNeck(64, 64, 128, 128)
        self.b38 = RDDNeck(64, 64, 16, 128, 128, False)
        
        self.b40 = UBNeck(64, 64, 128, 64)
        self.b41 = RDDNeck(128, 128, 1, 64, 64, False)
        self.b42 = RDDNeck(128, 128, 1, 64, 64, False)
        
        self.b50 = UBNeck(128, 128, 64, 16)
        self.b51 = RDDNeck(256, 256, 1, 16, 16, False)
        
        self.fullconv = nn.ConvTranspose2d(in_channels=16, 
                                           out_channels=self.C, 
                                           kernel_size=3, 
                                           stride=2, 
                                           padding=1, 
                                           output_padding=1)
        
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        
        x = self.init(x)
        
        x, i1 = self.b10(x)
        x = self.b11(x)
        x = self.b12(x)
        x = self.b13(x)
        x = self.b14(x)
        
        x, i2 = self.b20(x)
        x = self.b21(x)
        x = self.b22(x)
        x = self.b23(x)
        x = self.b24(x)
        x = self.b25(x)
        x = self.b26(x)
        x = self.b27(x)
        x = self.b28(x)
        
        x = self.b31(x)
        x = self.b32(x)
        x = self.b33(x)
        x = self.b34(x)
        x = self.b35(x)
        x = self.b36(x)
        x = self.b37(x)
        x = self.b38(x)
        
        x = self.b40(x, i2)
        x = self.b41(x)
        x = self.b42(x)
        
        x = self.b50(x, i1)
        x = self.b51(x)
        
        x = self.fullconv(x)
        x = self.sigmoid(x)
        
        return x