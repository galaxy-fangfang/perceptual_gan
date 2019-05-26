#!/usr/bin/python
# -*- coding: UTF-8 -*-
import cv2
import os
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dir',help = "image diretory",type = str)
parser.add_argument('--fac',help = "resize factor",type = float)
parser.add_argument('--save',help = "save directory",type = str)

args = parser.parse_args()
imgdir = args.dir
factor = args.fac
savedir = args.save

imgs = os.listdir(imgdir)
imgs.sort()
for img in imgs:
    a = cv2.imread(os.path.join(imgdir,img))
    oh,ow = a.shape[:2]
    nh = int(oh * factor)
    nw = int(ow * factor)
    b = cv2.resize(a,(nw,nh))
    cv2.imwrite(os.path.join(savedir,'{}'.format(img)),b)

