#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: edward
"""

from PIL import Image, ImageDraw
from matplotlib.pyplot import imshow, figure
from matplotlib import animation, pyplot
import numpy as np
import random
import copy

from Shapes import Polygon
from skimage.measure import compare_ssim
random.seed(123)
#random.seed(321)

refImg = Image.open('x-ray2.jpg')
refImg.load()
refArray = np.asarray(refImg).astype(np.int16)

if len(refArray.shape) < 3:
    refArray = np.dstack([refArray]*3)
    
if refArray.shape[2] > 3:
    refArray = refArray[:,:,0:3]

imageX = refArray.shape[1]
imageY = refArray.shape[0]
nPoly = 50
nV = 3
grayscale = True

def fitness(image):
    if grayscale:
        imArray = np.asarray(image).astype(np.int16)[:,:,0]
        diff = np.sum(np.abs(refArray[:,:,0] - imArray))
        diff = diff/(255*imageX*imageY*1)
        #diff = diff + 1-compare_ssim(refArray[:,:,0], np.asarray(image).astype(np.int16)[:,:,0])
    else:
        imArray = np.asarray(image).astype(np.int16)
        diff = np.sum(np.abs(refArray - imArray))
    return diff

def DrawImage(polyList, imageX, imageY):
    im = Image.new('RGB', (imageX, imageY))
    draw = ImageDraw.Draw(im, 'RGBA')
    
    for poly in polyList:
        poly.DrawOnImage(draw)
    
    return im

allPoly = []
for x in range(nPoly):
    initVertices = [[random.randint(0, imageX), random.randint(0, imageY)] for _ in range(nV) ]
    initFill = random.randint(0, 255)
    initRGBA = [initFill, initFill, initFill, random.randint(0,255)]
    allPoly.append(Polygon(initVertices, initRGBA, imageX, imageY, nV))

fig = pyplot.figure()
im = DrawImage(allPoly, imageX, imageY)
prevFit = fitness(im)
myobj = pyplot.imshow(im)


def updateImg(i):
    global allPoly, prevFit
    
    mutatePoly = random.randint(0, nPoly-1)
    
    allPoly2 = copy.deepcopy(allPoly)
    allPoly2[mutatePoly].MediumMutate()
    
    
    im = DrawImage(allPoly2, imageX, imageY)
    newFit = fitness(im)
    if newFit < prevFit:
        prevFit = newFit
        allPoly = copy.deepcopy(allPoly2)
        print(1-fitness(im))
        #print(1 - fitness(im)/(255*imageX*imageY*1))
        myobj.set_data(im)
    
    
    return myobj

ani = animation.FuncAnimation(fig, updateImg, interval = 0)
pyplot.show()
    