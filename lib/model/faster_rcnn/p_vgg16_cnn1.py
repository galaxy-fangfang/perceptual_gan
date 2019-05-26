# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen
# --------------------------------------------------------
# Written by Lanfang Kong
# p-vgg16.py
# 1.extract conv1 layer.  Done
# 2.define Generator network. 
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import torchvision.models as models
from model.faster_rcnn.p_faster_rcnn_cnn1 import _fasterRCNN
#from model.faster_rcnn.faster_rcnn import _netD
import pdb
def weight_init(m):
    #for layer in m:
    if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()

############################################
# define the basic block in generator network
# 3x3->bn->relu->3x3 + ------>  out
#    |_______________|
############################################
class BasicBlock(nn.Module):
    def __init__(self, inplanes, outplanes, stride = 1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, outplanes, kernel_size = 3, stride = stride, padding = 1, bias = False)
        weight_init(self.conv1)
        self.bn1 = nn.BatchNorm2d(outplanes)
        weight_init(self.bn1)
        self.relu = nn.ReLU(inplace = True)
        self.conv2 = nn.Conv2d(outplanes, inplanes, kernel_size = 3, stride = stride, padding = 1, bias = False)
        weight_init(self.conv2)
        self.bn2 = nn.BatchNorm2d(inplanes)
        weight_init(self.bn2)
       #for m in self.modules():
        #    if isinstance(m,nn.Conv2d):
        #        n = m.kernel_size[0] * m.kernel_size[1]* m.out_channels
        #        m.weight.data.normal_(0,math.sqrt(2./n))
        #    elif isinstance(m,nn.BatchNorm2d):
        #        m.weight.data.fill_(1)
        #        m.bias.data.zero_()

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)
        #out = self.relu(out)
        return out

################################################
# define B BasicBlock in the generator network
# defalut: B = 3
################################################
class ResidualNet(nn.Module):
    def __init__(self,layers = [3]):
    
        self.inplanes = 512
        self.outplanes = 128
        super(ResidualNet,self).__init__()

        self.layer1 = self._make_layers(BasicBlock, layers[0])


    def _make_layers(self,block,blocks,stride = 1):
        
        layers=[]
        layers.append(block(self.inplanes,self.outplanes,stride))
        #self.inplanes = planes * block.expansion
        for i in range (1,blocks):
            layers.append(block(self.inplanes,self.outplanes, stride = 1))
        
        return nn.Sequential(*layers)


    def forward(self,x):
        #import ipdb
        #ipdb.set_trace()
        x = self.layer1(x)

        
        return x

class vgg16(_fasterRCNN):
  def __init__(self, classes, pretrained=False, class_agnostic=False):
    self.model_path = 'data/pretrained_model/vgg16_caffe.pth'
    self.dout_base_model = 512
    self.pretrained = pretrained
    self.class_agnostic = class_agnostic

    _fasterRCNN.__init__(self, classes, class_agnostic)

  def _init_modules(self):
    vgg = models.vgg16()
    if self.pretrained:
        print("Loading pretrained weights from %s" %(self.model_path))
        state_dict = torch.load(self.model_path)
        vgg.load_state_dict({k:v for k,v in state_dict.items() if k in vgg.state_dict()})

    vgg.classifier = nn.Sequential(*list(vgg.classifier._modules.values())[:-1])
    
    #import ipdb
    #ipdb.set_trace()
    # define the conv1 layer
    #self.Conv1 = nn.Sequential(*list(vgg.features._modules.values())[:5])
    # not using the last maxpool layer
    self.RCNN_base = nn.Sequential(*list(vgg.features._modules.values())[:-1])

    # Fix all the layers in backbone:
 #   for layer in range(5):
#        for p in self.Conv1[layer].parameters(): p.requires_grad = False
    for layer in range(30):
        for p in self.RCNN_base[layer].parameters(): p.requires_grad = False

    # self.RCNN_base = _RCNN_base(vgg.features, self.classes, self.dout_base_model)

    self.RCNN_top = vgg.classifier

    # not using the last maxpool layer
    self.RCNN_cls_score = nn.Linear(4096, self.n_classes)

    #############################################
    # Generator Network
    #############################################

    # add 3x3 convolutional layer and 1x1 convolutional layer to increase the feature dimension
    self.downsample = nn.Sequential(
            nn.Conv2d(in_channels = 64, out_channels= 512, kernel_size = 3, stride = 2, padding = 0, bias = False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace = True),
            nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = 1, stride = 2, padding = 0, bias = False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(2,ceil_mode = False)
            )
    # when train RPN, do not need the gradients
    #for p in self.downsample.parameters(): p.requires_grad = False
    # define the roi pooling layer in the generator network
        
    ########################################
    # define the residual block,  B number
    ########################################
    # residual block has been initialized
    ########################################
    self.residualblock = ResidualNet()
    # when train rpn, do not need the gradients
    #for p in self.residualblock.parameters(): p.requires_grad = False

    def set_bn_fix(m):
        classname = m.__class__.__name__
        if classname.find('BatchNorm') != -1:
            for p in m.parameters(): p.requires_grad=False
    # when testing, fix the bn layer
    self.residualblock.apply(set_bn_fix)


    if self.class_agnostic:
      self.RCNN_bbox_pred = nn.Linear(4096, 4)
    else:
      self.RCNN_bbox_pred = nn.Linear(4096, 4 * self.n_classes)      
    
    ###################################
    # edited  by fangfang
    # 2019/1/10
    # def train(self,mode = True):
    #     # override train so that the training mode is set as we want
    #     # set RPN as eval mode when training GAN
    #     nn.Module.train(self,mode)
    #     if mode:
    #         # set fixed block to be in eval mode
    #         self.Conv1.eval()
    #         self.RCNN_base.eval()
    #         self.RCNN_RPN.eval()
    #         self.downsample.train()
    #         self.residualblock.train()

  def _head_to_tail(self, pool5):
    
    pool5_flat = pool5.view(pool5.size(0), -1)
    fc7 = self.RCNN_top(pool5_flat)

    return fc7
  
