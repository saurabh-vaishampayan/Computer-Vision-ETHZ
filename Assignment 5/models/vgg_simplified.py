import torch
import torch.nn as nn
import math

class Vgg(nn.Module):
    def __init__(self, fc_layer=512, classes=10):
        super(Vgg, self).__init__()
        """ Initialize VGG simplified Module
        Args: 
            fc_layer: input feature number for the last fully MLP block
            classes: number of image classes
        """
        self.fc_layer = fc_layer
        self.classes = classes

        #TODO: construct the simplified VGG network blocks
        # input shape: [bs, 3, 32, 32]
        # layers and output feature shape for each block:
        # # conv_block1 (Conv2d, ReLU, MaxPool2d) --> [bs, 64, 16, 16]
        # # conv_block2 (Conv2d, ReLU, MaxPool2d) --> [bs, 128, 8, 8]
        # # conv_block3 (Conv2d, ReLU, MaxPool2d) --> [bs, 256, 4, 4]
        # # conv_block4 (Conv2d, ReLU, MaxPool2d) --> [bs, 512, 2, 2]
        # # conv_block5 (Conv2d, ReLU, MaxPool2d) --> [bs, 512, 1, 1]
        # # classifier (Linear, ReLU, Dropout2d, Linear) --> [bs, 10] (final output)

        # hint: stack layers in each block with nn.Sequential, e.x.:
        # # self.conv_block1 = nn.Sequential(
        # #     layer1,
        # #     layer2,
        # #     layer3,
        # #     ...)

        
        self.conv_block1 = nn.Sequential(nn.Conv2d(in_channels=3, 
                                                   out_channels=64, 
                                                   kernel_size=3,
                                                   stride=1,
                                                   padding=1),
                                         nn.ReLU(),
                                         nn.MaxPool2d(kernel_size=2,
                                                      stride=2))
        
        self.conv_block2 = nn.Sequential(nn.Conv2d(in_channels=64, 
                                                   out_channels=128, 
                                                   kernel_size=3,
                                                   stride=1,
                                                   padding=1),
                                         nn.ReLU(),
                                         nn.MaxPool2d(kernel_size=2,
                                                      stride=2))
        
        self.conv_block3 = nn.Sequential(nn.Conv2d(in_channels=128, 
                                                   out_channels=256, 
                                                   kernel_size=3,
                                                   stride=1,
                                                   padding=1),
                                         nn.ReLU(),
                                         nn.MaxPool2d(kernel_size=2,
                                                      stride=2))
        
        self.conv_block4 = nn.Sequential(nn.Conv2d(in_channels=256, 
                                                   out_channels=512, 
                                                   kernel_size=3,
                                                   stride=1,
                                                   padding=1),
                                         nn.ReLU(),
                                         nn.MaxPool2d(kernel_size=2, 
                                                      stride=2))
        
        self.conv_block5 = nn.Sequential(nn.Conv2d(in_channels=512, 
                                                   out_channels=512, 
                                                   kernel_size=3,
                                                   stride=1,
                                                   padding=1),
                                         nn.ReLU(),
                                         nn.MaxPool2d(kernel_size=2,
                                                      stride=2))
        
        self.final_classifier = nn.Sequential(nn.Linear(in_features=512, out_features=128),
                                              nn.ReLU(),
                                              nn.Dropout2d(p=0.75),
                                              nn.Linear(in_features=128, out_features=10))
        

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()


    def forward(self, x):
        """
        :param x: input image batch tensor, [bs, 3, 32, 32]
        :return: score: predicted score for each class (10 classes in total), [bs, 10]
        """
        score = None
        #TODO
        y = x.clone()
        y = self.conv_block1(y)
        y = self.conv_block2(y)
        y = self.conv_block3(y)
        y = self.conv_block4(y)
        y = self.conv_block5(y)
        y = y.squeeze(-1).squeeze(-1) #The linear layer only accepts a 2D tensor.
        score = self.final_classifier(y)
        return score

