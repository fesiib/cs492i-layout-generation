from functools import cmp_to_key
from typing import ForwardRef
from numpy.core.numeric import convolve, outer
from numpy.core.shape_base import stack

from torch.autograd import backward
from torch.nn.functional import pad

from utils_CNN import get_args, get_bb_types, get_device

import torch
import torch.nn as nn

# Basic settings
torch.manual_seed(470)
torch.cuda.manual_seed(470)

args = get_args()
BB_TYPES = get_bb_types()
device = get_device()

class ConvBlock(nn.Module):
    """Convolutiona block with activation and batch normalization

    Args:
        nn ([type]): [description]
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=True,
        batch_norm = False, activation="ReLU"):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, 
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias = bias
            )

        self.batchnorm = None
        self.activation = None
        if batch_norm:
            self.batchnorm = nn.BatchNorm2d(out_channels)
        if activation == "ReLU":
            self.activation = nn.ReLU()
        elif activation == "LeakyReLU":
            self.activation = nn.LeakyReLU(0.1, True)


    def forward(self, x):
        print(x.shape)
        output = self.conv(x)
        
        if self.batchnorm is not None:
            output = self.batchnorm(output)
        if self.activation is not None:
            output = self.activation(output)
        
        return output


class TransposedConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, 
        stride, dilation=1, padding=0, bias=True, batch_norm = False, activation="ReLU"):
        super(TransposedConvBlock, self).__init__()
        self.deconv = nn.ConvTranspose2d(
            in_channels=in_channels, 
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
            dilation=dilation)
        self.batchnorm = None
        self.activation = None

        if batch_norm:
            self.batchnorm = nn.BatchNorm2d(out_channels)
        if activation == "ReLU":
            self.activation = nn.ReLU()
        elif activation == "LeakyReLU":
            self.activation = nn.LeakyReLU(0.1, True)
        elif activation == "Tanh":
            self.activation = nn.Tanh()
        
        
    def forward(self, x):
        output = self.deconv(x)
        
        if self.batchnorm is not None:
            output = self.batchnorm(output)
        if self.activation is not None:
            output = self.activation(output)
        
        return output



class ConditionedLayoutEncoder(nn.Module):
    
    def __init__(self, in_channels = 3, y_dimension = 128):
        super(ConditionedLayoutEncoder, self).__init__()
        self.convs = nn.ModuleList()
        self.y_dimension = y_dimension

        conv1 = ConvBlock(in_channels = in_channels, 
            out_channels=64, 
            kernel_size=5, 
            stride=2, 
            padding=2, 
            bias=True, 
            batch_norm=False, 
            activation="LeakyReLU")

        self.convs.append(conv1)
        in_c = 64
        out_c = in_c * 2
        for i in range(3): 
            conv = ConvBlock(in_channels = in_c, 
                out_channels=out_c, 
                kernel_size=5, 
                stride=2, 
                padding=2, 
                bias=True, 
                batch_norm=True, 
                activation="LeakyReLU")
            self.convs.append(conv)
            in_c = out_c
            out_c = in_c * 2
        
        self.final_conv1 =  ConvBlock(in_channels = y_dimension + 512, 
            out_channels= args.cond_layout_encoder_dim, 
            kernel_size=4, 
            stride=1, 
            padding=0, 
            bias=True, 
            batch_norm=False, 
            activation=None)
        
        self.final_conv2 =  ConvBlock(in_channels=y_dimension + 512, 
            out_channels=args.cond_layout_encoder_dim, 
            kernel_size=4, 
            stride=1, 
            padding=0, 
            bias=True, 
            batch_norm=False, 
            activation=None)
        

    
    def forward(self, x, y):
        
        output = x
        batch_size = x.size(0)

        for i, layer in enumerate(self.convs):
            output = layer(output)

        y = y.repeat(4, 4, 1, 1)
        y = y.transpose(1, 3).transpose(0, 2)

        output = torch.cat((output, y), 1)
        output_mu = self.final_conv1(output)
        logVar = self.final_conv2(output)
        output_mu = torch.squeeze(torch.squeeze(output_mu, dim=3), dim=2)
        logVar = torch.squeeze(torch.squeeze(logVar, dim=3), dim=2)

        return output_mu, logVar

class LayoutEncoder(nn.Module):
    """
        Slide Encoder that creates embedding of the slide structure

    """
    def __init__(self, in_channels=3):
        super(LayoutEncoder, self).__init__()
        self.convs = nn.ModuleList()

        conv1 = ConvBlock(in_channels = in_channels, 
            out_channels=64, 
            kernel_size=5, 
            stride=2, 
            padding=2, 
            bias=True, 
            batch_norm=False, 
            activation="LeakyReLU")

        self.convs.append(conv1)
        in_c = 64
        out_c = in_c * 2
        for i in range(3): 
            conv = ConvBlock(in_channels = in_c, 
                out_channels=out_c, 
                kernel_size=5, 
                stride=2, 
                padding=2, 
                bias=True, 
                batch_norm=True, 
                activation="LeakyReLU")
            self.convs.append(conv)
            in_c = out_c
            out_c = in_c * 2
        
        self.final_conv1 =  ConvBlock(in_channels = in_c, 
            out_channels= args.layout_encoder_dim, 
            kernel_size=4, 
            stride=1, 
            padding=0, 
            bias=True, 
            batch_norm=False, 
            activation=None)
        
        self.final_conv2 =  ConvBlock(in_channels=in_c, 
            out_channels=args.layout_encoder_dim, 
            kernel_size=4, 
            stride=1, 
            padding=0, 
            bias=True, 
            batch_norm=False, 
            activation=None)

    def forward(self, x):
        output = x

        for i, layer in enumerate(self.convs):
            output = layer(output)

        output_mu = self.final_conv1(output)
        logVar = self.final_conv2(output)

        output_mu = torch.squeeze(torch.squeeze(output_mu, dim=3), dim=2)
        logVar = torch.squeeze(torch.squeeze(logVar, dim=3), dim=2)
        return output_mu, logVar


class SlideDeckEncoder(nn.Module):

    def __init__(self, in_channels=3):
        super(SlideDeckEncoder, self).__init__()
        self.slide_encoder = LayoutEncoder(in_channels)
        self.linear = nn.Linear(args.layout_encoder_dim, args.slide_deck_embedding_output_dim, 
            bias=False)
        self.dropout = nn.Dropout(0.5)
        self.activation = nn.LeakyReLU(0.1)

    def load_weights(self, path):
        try:
            checkpoint = torch.load(path)
        except:
            print("Couldn't load the last checkpoint!")
            return 

        self.slide_encoder.load_state_dict(checkpoint['model_encoder_state_dict'])

    def forward(self, x):
        """

        Args:
            x ([list[Tensors]]): Tensor size is [b, 3, W, H]
            
        """
        outputs = []

        for slide in x:
            output = self.slide_encoder(slide)
            output.detach()
            output = self.activation(self.dropout(self.linear(output)))
            outputs.append(output)
        
        features = torch.stack(outputs, dim=1)
        features = torch.max(features, dim=1, keepdim=False)[0]
        
        return features

        
class Generator(nn.Module):

    def __init__(self, input_dim=128):
        super(Generator, self).__init__()

        self.fc1 = nn.Linear(input_dim, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(1024, 8192)
        self.bn2 = nn.BatchNorm1d(8192)
        self.act2 = nn.ReLU()
        
        self.deconvs = nn.ModuleList()
        in_c = 512
        out_c = in_c // 2
        for i in range(3):
            deconv = TransposedConvBlock(in_c, out_c, 5, 2,
                padding=1, 
                bias=True, 
                batch_norm = True, 
                activation="ReLU")
            self.deconvs.append(deconv)
            in_c = out_c
            out_c = in_c // 2
        
        self.last_deconv = TransposedConvBlock(64, 3, 5, 2,
            padding=1, 
            bias=True, 
            batch_norm = False, 
            activation="Tanh")
    
    def forward(self, x):
        batch_size = x.size(0)
        output = self.act1(self.bn1(self.fc1(x)))
        output = self.act2(self.bn2(self.fc2(output)))

        output = torch.reshape(output, (batch_size, 512, 4, 4))

        resolution = 8
        for i, layer in enumerate(self.deconvs):
            output = layer(output)
            output = output[:,:,:resolution, :resolution]
            resolution *= 2
        
        output = self.last_deconv(output)
        output = output[:,:,:resolution, :resolution]
        
        return output
        
    
class ConditionedGenerator(nn.Module):

    def __init__(self, input_dim=128, cond_dim=128):
        super(ConditionedGenerator, self).__init__()

        self.fc1 = nn.Linear(input_dim + cond_dim, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(1024, 8192)
        self.bn2 = nn.BatchNorm1d(8192)
        self.act2 = nn.ReLU()
        
        self.deconvs = nn.ModuleList()
        in_c = 512
        out_c = in_c // 2
        for i in range(3):
            deconv = TransposedConvBlock(in_c, out_c, 5, 2,
                padding=1, 
                dilation=1,
                bias=True, 
                batch_norm = True, 
                activation="ReLU")
            self.deconvs.append(deconv)
            in_c = out_c
            out_c = in_c // 2
        
        self.last_deconv = TransposedConvBlock(64, 3, 5, 2,
            padding=1, 
            bias=True, 
            batch_norm = False, 
            activation="Tanh")
    
    def forward(self, x, y):
        batch_size = x.size(0)
        output = torch.cat((x, y), dim = 1)
        output = self.act1(self.bn1(self.fc1(output)))
        output = self.act2(self.bn2(self.fc2(output)))

        output = torch.reshape(output, (batch_size, 512, 4, 4))
        resolution = 8
        for i, layer in enumerate(self.deconvs):
            output = layer(output)
            output = output[:,:,:resolution, :resolution]
            resolution *= 2

        
        output = self.last_deconv(output)
        output = output[:,:,:resolution, :resolution]

        
        return output

class Discriminator(nn.Module):

    def __init__(self, input_channels=3):
        super(Discriminator, self).__init__()
        self.convs = nn.ModuleList()

        conv1 = ConvBlock(in_channels = input_channels, 
            out_channels=64, 
            kernel_size=5, 
            stride=2, 
            padding=2, 
            bias=True, 
            batch_norm=False, 
            activation="LeakyReLU")

        self.convs.append(conv1)
        in_c = 64
        out_c = in_c * 2
        for i in range(3): 
            conv = ConvBlock(in_channels = in_c, 
                out_channels=out_c, 
                kernel_size=5, 
                stride=2, 
                padding=2, 
                bias=True, 
                batch_norm=True, 
                activation="LeakyReLU")
            self.convs.append(conv)
            in_c = out_c
            out_c = in_c * 2
        
        self.final_conv1 =  ConvBlock(in_channels = in_c, 
            out_channels=128, 
            kernel_size=4, 
            stride=1, 
            padding=0, 
            bias=True, 
            batch_norm=False, 
            activation=None)
        
        self.fc = nn.Linear(256, 1)
        self.act = nn.LeakyReLU(0.1, True)
    

    def forward(self, x, z):

        output = x
        for i, layer in enumerate(self.convs):
            output = layer(output)

        output = self.final_conv1(output)
        output = torch.squeeze(output, dim=3)
        output = torch.squeeze(output, dim=2)
        
        output = torch.cat((output, z), dim=1)
        output = self.act(self.fc(output))
        
        return output


        

class ConditionedDiscriminator(nn.Module):
    
    def __init__(self, input_channels=3, cond_dim=128):
        super(ConditionedDiscriminator, self).__init__()
        self.convs = nn.ModuleList()

        conv1 = ConvBlock(in_channels = input_channels + cond_dim, 
            out_channels=64, 
            kernel_size=5, 
            stride=2, 
            padding=2, 
            bias=True, 
            batch_norm=False, 
            activation="LeakyReLU")
        
        self.convs.append(conv1)
        in_c = 64
        out_c = in_c * 2
        for i in range(3): 
            conv = ConvBlock(in_channels = in_c, 
                out_channels=out_c, 
                kernel_size=5, 
                stride=2, 
                padding=2, 
                bias=True, 
                batch_norm=True, 
                activation="LeakyReLU")
            self.convs.append(conv)
            in_c = out_c
            out_c = in_c * 2
        
        self.final_conv1 =  ConvBlock(in_channels = in_c, 
            out_channels=128, 
            kernel_size=4, 
            stride=1, 
            padding=0, 
            bias=True, 
            batch_norm=False, 
            activation=None)
        
        self.fc = nn.Linear(256, 1)
        self.act = nn.LeakyReLU(0.1, True)
    

    def forward(self, x, y, z):

        y = y.repeat(64, 64, 1, 1)
        y = y.transpose(1, 3).transpose(0, 2)
        output = torch.cat((x, y), dim=1)  

        for i, layer in enumerate(self.convs):
            output = layer(output)

        output = self.final_conv1(output)
        output = torch.squeeze(output, dim=3)
        output = torch.squeeze(output, dim=2)
        
        output = torch.cat((output, z), dim=1)
        output = self.act(self.fc(output))
        
        return output


class Discriminator(nn.Module):

    def __init__(self, input_channels=3):
        super(Discriminator, self).__init__()
        self.convs = nn.ModuleList()

        conv1 = ConvBlock(in_channels = input_channels, 
            out_channels=64, 
            kernel_size=5, 
            stride=2, 
            padding=2, 
            bias=True, 
            batch_norm=False, 
            activation="LeakyReLU")

        self.convs.append(conv1)
        in_c = 64
        out_c = in_c * 2
        for i in range(3): 
            conv = ConvBlock(in_channels = in_c, 
                out_channels=out_c, 
                kernel_size=5, 
                stride=2, 
                padding=2, 
                bias=True, 
                batch_norm=True, 
                activation="LeakyReLU")
            self.convs.append(conv)
            in_c = out_c
            out_c = in_c * 2
        
        self.final_conv1 =  ConvBlock(in_channels = in_c, 
            out_channels=128, 
            kernel_size=4, 
            stride=1, 
            padding=0, 
            bias=True, 
            batch_norm=False, 
            activation=None)
        
        self.fc = nn.Linear(128 + args.layout_encoder_dim, 1)
        self.act = nn.LeakyReLU(0.1, True)


    def forward(self, x, z):

        output = x
        for i, layer in enumerate(self.convs):
            output = layer(output)

        output = self.final_conv1(output)
        output = torch.squeeze(output, dim=3)
        output = torch.squeeze(output, dim=2)
        
        output = torch.cat((output, z), dim=1)
        output = self.act(self.fc(output))
        
        return output


