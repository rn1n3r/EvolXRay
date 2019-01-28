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

from skimage.measure import compare_ssim
from Shapes import Polygon
random.seed(123)
#random.seed(321)

nPoly = 2000
nV = 3 # Triangles
grayscale = True
headless = False

refImg = Image.open('x-ray2.jpg')
refImg.load()
refArray = np.asarray(refImg).astype(np.int16)

if len(refArray.shape) < 3:
    refArray = np.dstack([refArray]*3)
    
if refArray.shape[2] >= 3 and grayscale:
    refArray = refArray[:,:,0]
    
if grayscale:
    uniqueRGB = 1
else:
    uniqueRGB = 3
    
imageX = refArray.shape[1]
imageY = refArray.shape[0]


def clamp (x, minVal, maxVal):
    return max(minVal, min(x, maxVal))

def fitness (indiv):
    
    image = DrawImage(indiv, imageX, imageY)
    
    if grayscale:
        imArray = np.asarray(image).astype(np.int16)[:,:,0]
        delta = np.abs(refArray - imArray)
        diff = np.sum(delta)/(255*imageX*imageY*uniqueRGB)
        
    else:
        imArray = np.asarray(image).astype(np.int16)
        diff = np.sum(np.abs(refArray - imArray))

    return diff, image

# The version that takes an image and avoids the DrawImage call
def fitness2(image):

    if grayscale:
        imArray = np.asarray(image).astype(np.int16)[:,:,0]
        delta = np.abs(refArray - imArray)
        diff = np.sum(delta)/(255*imageX*imageY*uniqueRGB)
     #   diff = 1-compare_ssim(refArray, np.asarray(image).astype(np.int16)[:,:,0]) #np.sum(delta)
       
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

def updateImg(i):
    global allPoly, prevFit, gen, im
    
    
    if len(allPoly) < nPoly:
        initVertex1 = [random.randint(0,imageX-1), random.randint(0,imageY-1)]
        
        maxX = clamp(initVertex1[0]+30, 0, imageX)
        minX = clamp(initVertex1[0]-30, 0, imageX)
        
        maxY = clamp(initVertex1[1]+30, 0, imageY)
        minY = clamp(initVertex1[1]-30, 0, imageY)
        
        initVertex2 = [random.randint(minX, maxX), random.randint(minY,maxY)]
        initVertex3 = [random.randint(minX, maxX), random.randint(minY,maxY)]
        
        initVertices = [initVertex1, initVertex2, initVertex3]
        
         
        initFill = 0
        #initFill = random.randint(0,255)
        
        avgX = sum([x[0] for x in initVertices])//3
        avgY = sum([y[1] for y in initVertices])//3

        if avgX > 0 and avgX < imageX-1 and avgY > 0 and avgY < imageY-1:
            initFill = refArray[avgX, avgY] + refArray[avgX+1, avgY] + \
                        refArray[avgX-1, avgY] + refArray[avgX, avgY+1] + \
                        refArray[avgX, avgY-1]
            
            initFill = initFill//5
           
        else:
            initFill = refArray[avgX, avgY]
        
        initRGBA = [initFill, initFill, initFill, random.randint(0,255)]
        
        allPoly.append(Polygon(initVertices, initRGBA, imageX, imageY, nV))
        
        im2 = copy.deepcopy(im)
        draw = ImageDraw.Draw(im2, 'RGBA')
        allPoly[-1].DrawOnImage(draw)
        
        newFit = fitness2(im2)
        #newFit = fitness(allPoly)
        if prevFit > newFit:
            
            for x in [-1, 1]:
                while(True):
                    
                    initColor = allPoly[-1].fill[0:3]
                    allPoly[-1].fill[0:3] = [clamp(initColor[0]+ x*5, 0, 255) for _ in range(0,3)]
                    
                    im2 = im.copy()
                    draw = ImageDraw.Draw(im2, 'RGBA')
                    allPoly[-1].DrawOnImage(draw)
                    
                    currFit = fitness2(im2)
                    #print(str(currFit) + " " + str(initFit))
                    
                    if currFit >= newFit:
                        allPoly[-1].fill[0:3] = initColor
                        break
                    else:
                        newFit = currFit
                    
            # Optimize alpha
            for x in [-1, 1]:
                while(True):
                        
                    initAlpha = allPoly[-1].fill[3]
                    allPoly[-1].fill[3] = clamp(initAlpha + x*5, 0, 255)
                    
                    im2 = im.copy()
                    draw = ImageDraw.Draw(im2, 'RGBA')
                    allPoly[-1].DrawOnImage(draw)
                    
                    currFit = fitness2(im2)
                    #print(str(currFit) + " " + str(initFit))
                    
                    if currFit >= newFit:
                        allPoly[-1].fill[3] = initAlpha
                        break
                    else:
                        newFit = currFit
                        
            # Optimize the 3 vertices
            for x in [0,2]:
                for y in [0, 1]:
                    for z in [-1,1]:
                        while(True):
                            initVal = allPoly[-1].vertices[x][y]
                            
                            allPoly[-1].vertices[x][y] = clamp(initVal + z*15, 0, 255)
                            
                            im2 = im.copy()
                            draw = ImageDraw.Draw(im2, 'RGBA')
                            allPoly[-1].DrawOnImage(draw)
                            
                            currFit = fitness2(im2)
                            
                            if currFit >= newFit:
                                allPoly[-1].vertices[x][y] = initVal
                                break
                            else:
                                newFit = currFit
                        
            currFit = newFit
            im = im2.copy()
        else:
            del(allPoly[-1])
            currFit,_ = prevFit
                     
    # Mutation
    mutatePoly = random.randint(0, len(allPoly)-1)
    origPoly = copy.deepcopy(allPoly[mutatePoly])
    
    allPoly[mutatePoly].SoftMutate()
    
    currFit,_ = fitness(allPoly)
    
    if currFit >= prevFit:
        allPoly[mutatePoly] = copy.deepcopy(origPoly)
    else:
        prevFit = currFit
     
    gen = gen + 1
    print(str(1-prevFit) + " " + str(gen) + " " + str(len(allPoly)))
    im = DrawImage(allPoly, imageX, imageY)
    myobj.set_data(im)
    
    return myobj

allPoly = []
#for x in range(nPoly):
#    initVertices = [[random.randint(0, imageX), random.randint(0, imageY)] for _ in range(nV) ]
#    initFill = random.randint(0, 255)
#    initRGBA = [initFill, initFill, initFill, random.randint(0,255)]
#    allPoly.append(Polygon(initVertices, initRGBA, imageX, imageY, nV))

#im = DrawImage(allPoly, imageX, imageY)
prevFit,im = fitness(allPoly)

gen = 1


if not headless:    
    fig = pyplot.figure()
    myobj = pyplot.imshow(im)
    
    ani = animation.FuncAnimation(fig, updateImg, interval = 0)
    pyplot.show()
else:
    

    for loops in range(0,699):
  #  while True:
        if len(allPoly) < nPoly:
            #initVertices = [[random.randint(0, imageX-1), random.randint(0, imageY-1)] for _ in range(nV) ]
            initVertex1 = [random.randint(0,imageX-1), random.randint(0,imageY-1)]
            
            maxX = clamp(initVertex1[0]+30, 0, imageX)
            minX = clamp(initVertex1[0]-30, 0, imageX)
            
            maxY = clamp(initVertex1[1]+30, 0, imageY)
            minY = clamp(initVertex1[1]-30, 0, imageY)
            
            initVertex2 = [random.randint(minX, maxX), random.randint(minY,maxY)]
            initVertex3 = [random.randint(minX, maxX), random.randint(minY,maxY)]
            
            initVertices = [initVertex1, initVertex2, initVertex3]
            
            initFill = random.randint(0, 255)
            
            initRGBA = [initFill, initFill, initFill, random.randint(0,255)]
            
            allPoly.append(Polygon(initVertices, initRGBA, imageX, imageY, nV))
            
            im2 = copy.deepcopy(im)
            draw = ImageDraw.Draw(im2, 'RGBA')
            allPoly[-1].DrawOnImage(draw)
            
            newFit = fitness2(im2)
            #newFit = fitness(allPoly)
            if prevFit > newFit:
                
                for x in [-1, 1]:
                    while(True):
                        
                        initColor = allPoly[-1].fill[0:3]
                        allPoly[-1].fill[0:3] = [clamp(initColor[0]+ x*5, 0, 255) for _ in range(0,3)]
                        
                        im2 = im.copy()
                        draw = ImageDraw.Draw(im2, 'RGBA')
                        allPoly[-1].DrawOnImage(draw)
                        
                        currFit = fitness2(im2)
                        #print(str(currFit) + " " + str(initFit))
                        
                        if currFit >= newFit:
                            allPoly[-1].fill[0:3] = initColor
                            break
                        else:
                            newFit = currFit
                        
                # Optimize alpha
                for x in [-1, 1]:
                    while(True):
                            
                        initAlpha = allPoly[-1].fill[3]
                        allPoly[-1].fill[3] = clamp(initAlpha + x*5, 0, 255)
                        
                        im2 = im.copy()
                        draw = ImageDraw.Draw(im2, 'RGBA')
                        allPoly[-1].DrawOnImage(draw)
                        
                        currFit = fitness2(im2)
                        #print(str(currFit) + " " + str(initFit))
                        
                        if currFit >= newFit:
                            allPoly[-1].fill[3] = initAlpha
                            break
                        else:
                            newFit = currFit
                            
                # Optimize the 3 vertices
                for x in [0,2]:
                    for y in [0, 1]:
                        for z in [-1,1]:
                            while(True):
                                initVal = allPoly[-1].vertices[x][y]
                                
                                allPoly[-1].vertices[x][y] = clamp(initVal + z*15, 0, 255)
                                
                                im2 = im.copy()
                                draw = ImageDraw.Draw(im2, 'RGBA')
                                allPoly[-1].DrawOnImage(draw)
                                
                                currFit = fitness2(im2)
                                
                                if currFit >= newFit:
                                    allPoly[-1].vertices[x][y] = initVal
                                    break
                                else:
                                    newFit = currFit
                            
                currFit = newFit
                im = im2.copy()
            else:
                del(allPoly[-1])
                currFit = prevFit
        prevFit = currFit
        
        # Mutation
        mutatePoly = random.randint(0, len(allPoly)-1)
        origPoly = copy.deepcopy(allPoly[mutatePoly])
        
        allPoly[mutatePoly].SoftMutate()
        
        currFit, im = fitness(allPoly)
        
        if currFit >= prevFit:
            allPoly[mutatePoly] = copy.deepcopy(origPoly)
            im = DrawImage(allPoly, imageX, imageY)
        else:
            prevFit = currFit
            
        #im = DrawImage(allPoly, imageX, imageY)
        #myobj.set_data(im)
        gen = gen + 1
        print(str(1-prevFit) + " " + str(gen) + " " + str(len(allPoly)))
       # im = DrawImage(allPoly, imageX, imageY)
    



