#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Genetic alg version
@author: edward
"""

import utils

from PIL import Image, ImageDraw
from matplotlib.pyplot import imshow, figure
from matplotlib import animation, pyplot
import numpy as np
import random
import copy
import sys
import math

from Shapes import Polygon
#random.seed(123)
random.seed(321)

# Load image 
refImg = Image.open('../img/NIH-1.png')
refImg.load()
refArrayOrig = np.asarray(refImg).astype(np.int16)


nPoly = 15000
nV = 3 # Triangle
popSize = 30
grayscale = True
hotStart = False

headless = False
if len(sys.argv) > 1:
    headless = True

selectionPercent = 0.25
popSize = popSize - (popSize % math.floor(popSize*selectionPercent))
popSize = int(popSize)

imageX = refArrayOrig.shape[1]
imageY = refArrayOrig.shape[0]

#if len(refArray.shape) < 3:
#    refArray = np.dstack([refArray]*3)
#    
#if refArray.shape[2] > 3:
#    refArray = refArray[:,:,0:3]
if len(refArrayOrig.shape) < 3:
    refArrayOrig = np.dstack([refArrayOrig]*3)
    
if refArrayOrig.shape[2] >= 3 and grayscale:
    refArrayOrig = refArrayOrig[:,:,0]


if grayscale:
    uniqueRGB = 1
else:
    uniqueRGB = 3


# y, x
roi1 = [0,0]
roi2 = [254,254]

#roi1 = [75,75]
#roi2 = [170,170]

imageY = roi2[0]+1
imageX = roi2[1]+1

imageMinY = roi1[0]
imageMinX = roi1[1]

refArray = refArrayOrig[roi1[0]:roi2[0]+1, roi1[1]:roi2[1]+1]


def clamp (x, minVal, maxVal):
    return max(minVal, min(x, maxVal))


# Sum of differences
def fitness (indiv):
    
    image = DrawImage(indiv, imageX-imageMinX, imageY-imageMinY)
    
    if grayscale:
        imArray = np.asarray(image).astype(np.int16)[:,:,0]
        #imArray = np.asarray(image).astype(np.int16)[roi1[0]:roi2[0]+1, roi1[1]:roi2[1]+1, 0]
        delta = np.abs(refArray - imArray)
        diff = np.sum(delta)/(255.0*(imageX-imageMinX)*(imageY-imageMinY)*uniqueRGB)
        
        
    else:
        imArray = np.asarray(image).astype(np.int16)
        diff = np.sum(np.abs(refArray - imArray))

    return diff, image

# The version that takes an image and avoids the DrawImage call
def fitness2(image):

    if grayscale:
        #imArray = np.asarray(image).astype(np.int16)[:,:,0]
        imArray = np.asarray(image).astype(np.int16)[roi1[0]:roi2[0]+1, roi1[1]:roi2[1]+1, 0]
        
        delta = np.abs(refArray - imArray)
        diff = np.sum(delta)/(255.0*(imageX-imageMinX)*(imageY-imageMinY)*uniqueRGB)
     #   diff = 1-compare_ssim(refArray, np.asarray(image).astype(np.int16)[:,:,0]) #np.sum(delta)
       
    else:
        imArray = np.asarray(image).astype(np.int16)
        diff = np.sum(np.abs(refArray - imArray))
    
    return diff


def DrawImage(polyList, lenX, lenY):
    im = Image.new('RGB', (lenX, lenY))
    draw = ImageDraw.Draw(im, 'RGBA')
    
    for poly in polyList:
        poly.DrawOnImage(draw)
    
    return im

# Returns list of fitness values for each image in population
def GetPopFitness(population, popSize, imageX, imageY, p):
    popFitness = [0]*popSize
    
        
    for i, indiv in enumerate(population):
        popFitness[i],_ = fitness(indiv)
        
    
    return popFitness


def copyPolyList(polyList):
    newPolyList = [Polygon([[z[0], z[1]] for z in [y for y in x.vertices]], [y for y in x.fill], imageX, imageY) for x in polyList]
    return newPolyList

def CreateNewGen(population, popFitness):
    # population and popFitness are presorted

    newPop = []
    newPopFitness = []
    parents = int(math.floor(popSize*selectionPercent))
    
    # Calculate fittest individual from previous generation
    bestPrevFitness,_ = fitness(population[0])
    
    for i in range(0,parents):
        prevFit = popFitness[i]
        im = DrawImage(population[i], 255, 255)
        for j in range(0, popSize//parents):
            firstParent = i
            randomParent = random.randint(0, parents)
            #newPop.append(copy.deepcopy(population[firstParent]))
            newPop.append(copyPolyList(population[firstParent]))
            
            
            if len(newPop[-1]) < nPoly:
                
                initVertex1 = [random.randint(0,imageX-1-imageMinX), random.randint(0,imageY-1-imageMinY)]
                
                maxX = clamp(initVertex1[0]+30, 0, imageX-imageMinX)
                minX = clamp(initVertex1[0]-30, 0, imageX-imageMinX)
                
                maxY = clamp(initVertex1[1]+30, 0, imageY-imageMinY)
                minY = clamp(initVertex1[1]-30, 0, imageY-imageMinY)
                
                initVertex2 = [random.randint(minX, maxX), random.randint(minY,maxY)]
                initVertex3 = [random.randint(minX, maxX), random.randint(minY,maxY)]
                
                initVertices = [initVertex1, initVertex2, initVertex3]
                
                 
                initFill = 0
                
                avgX = sum([x[0] for x in initVertices])//3
                avgY = sum([y[1] for y in initVertices])//3
        
                if avgX > 0 and avgX < imageX-1 and avgY > 0 and avgY < imageY-1:
                    initFill = refArray[avgY, avgX] + refArray[avgY+1, avgX] + \
                                refArray[avgY-1, avgX] + refArray[avgY, avgX+1] + \
                                refArray[avgY, avgX-1]
                    
                    initFill = initFill//5
                   
                else:
                    initFill = refArray[avgX, avgY]
                
                initRGBA = [initFill, initFill, initFill, random.randint(0,255)]
                newPop[-1].append(Polygon(initVertices, initRGBA, imageX, imageY, nV))
                
                
                newFit,_ = fitness(newPop[-1])
                

                for x in [-1, 1]:
                    while(True):
                        
                        initColor = newPop[-1][-1].fill[0:3]
                        newPop[-1][-1].fill[0:3] = [clamp(initColor[0]+ x*5, 0, 255) for _ in range(0,3)]
                        
                        im2 = im.copy()
                        draw = ImageDraw.Draw(im2, 'RGBA')
                        newPop[-1][-1].DrawOnImage(draw)
                        
                        currFit = fitness2(im2)

                        if currFit >= newFit:
                            newPop[-1][-1].fill[0:3] = initColor
                            break
                        else:
                            newFit = currFit
                        
                # Optimize alpha
                for x in [-1, 1]:
                    while(True):
                            
                        initAlpha = newPop[-1][-1].fill[3]
                        newPop[-1][-1].fill[3] = clamp(initAlpha + x*5, 0, 255)
                        
                        im2 = im.copy()
                        draw = ImageDraw.Draw(im2, 'RGBA')
                        newPop[-1][-1].DrawOnImage(draw)
                        
                        currFit = fitness2(im2)
                        
                        if currFit >= newFit:
                            newPop[-1][-1].fill[3] = initAlpha
                            break
                        else:
                            newFit = currFit
                          
                # Optimize the 3 vertices
                for x in [0,2]:
                    for y in [0, 1]:
                        for z in [-1,1]:
                            while(True):
                                initVal = newPop[-1][-1].vertices[x][y]
                                
                                if y == 0:
                                    limit = maxX
                                else:
                                    limit = maxY
                                    
                                newPop[-1][-1].vertices[x][y] = clamp(initVal + z*15, 0, limit)
                                
                                im2 = im.copy()
                                draw = ImageDraw.Draw(im2, 'RGBA')
                                newPop[-1][-1].DrawOnImage(draw)
                                
                                currFit = fitness2(im2)
                                
                                if currFit >= newFit:
                                    newPop[-1][-1].vertices[x][y] = initVal
                                    break
                                else:
                                    newFit = currFit
                    
            else:
                newFit = prevFit
                                    
            
            # Random Mutation
            for k in range(0, len(newPop[-1])):   
                if len(newPop[-1]) > 0:
                    if random.randint(0, 100) < 1:
                        newPop[-1][k].SoftMutate()
                    
            # Crossover
            if random.randint(0,100) < 100:
                
                for x in range(0, min(len(population[randomParent]), len(population[i]))-1):
                    if random.randint(0,1) == 1:
                        newPop[-1][x].vertices = population[randomParent][x].vertices
                        newPop[-1][x].fill = population[randomParent][x].fill
                    
            newFit,_ = fitness(newPop[-1])
            newPopFitness.append(newFit)                           
                                        
            

    
    newPop = sorted(list(zip(newPopFitness, newPop)), key=lambda x: x[0])
    newPopFitness = [x[0] for x in newPop]
    newPop = [x[1] for x in newPop]
   

    
    # If the fittest in the new generation is less fit than the best from before, replace it
    if bestPrevFitness < min(newPopFitness):
        newPop[0] = copyPolyList(population[0])
        newPopFitness[newPopFitness.index(min(newPopFitness))] = bestPrevFitness
        

            
    return newPop, newPopFitness

def updateImg(i):
    global population, popFitness, gen, roi1, roi2, imageY, imageX, imageMinY, imageMinX, refArray
    
    
#    roi1 = [75,75]
#    roi2 = [170,170]
    
    population, popFitness = CreateNewGen(population, popFitness)
    
    best = copy.deepcopy(population[0])
    
    ## ?? How do I adjust if I'm going from full frame to focused??
#    for x in best:
#        x.vertices = [[a[0] + roi1[0], a[1] + roi1[1]] for a in x.vertices]
#                    
    im = DrawImage(best, 255, 255)

    #maxDiff = 255*imageX*imageY*uniqueRGB
    #print(1 - min(popFitness)/maxDiff)
    gen += 1
    print(str(1-min(popFitness)) + " " + str(gen))
    myobj.set_data(im)
    
#    if gen == 100:
#        for x in population:
#            for y in x:
#                y.vertices = [[a[0] + roi1[0], a[1] + roi1[1]] for a in y.vertices]
#                
#                        
#        roi1 = [75,75]
#        roi2 = [170,170]
#        
#        imageY = roi2[0]+1
#        imageX = roi2[1]+1
#        
#        imageMinY = roi1[0]
#        imageMinX = roi1[1]
#        
#        refArray = refArrayOrig[roi1[0]:roi2[0]+1, roi1[1]:roi2[1]+1]
#        
#        popFitness = [1 for x in popFitness]

    
    
    return myobj
        
# Initialize first population
    
if not hotStart:
    population = list()
    
    for x in range(popSize):
#        initVertices = [[[random.randint(0,imageX), random.randint(0,imageY)] for _ in range(nV)] for _ in range(nPoly)] 
#        
#        initRGBA = []
#        for i in range(nPoly):
#            if grayscale:
#                initGray = random.randint(0,255)
#                initRGBA.append([initGray, initGray, initGray, random.randint(0,255)])
#            else:
#                initRGBA.append([random.randint(0,255), random.randint(0,255), random.randint(0,255), random.randint(0,255)])
#        
#        allPoly = [Polygon(initVertices[i], initRGBA[i], imageX, imageY, nV, grayscale) for i in range(nPoly)]
        allPoly = []
        population.append(allPoly)

#p = Pool(4)
p = 1
#popFitness = GetPopFitness(population, popSize, imageX, imageY, p)
popFitness =  GetPopFitness(population, popSize, imageX, imageY, p)
population = sorted(list(zip(popFitness, population)), key=lambda x: x[0])
popFitness = [x[0] for x in population]
population = [x[1] for x in population]

imList = []

gen = 0

if not headless:
    fig = pyplot.figure()
    myobj = pyplot.imshow(DrawImage(population[0], 255, 255))
        
    ani = animation.FuncAnimation(fig, updateImg, interval = 0)
    pyplot.show()

else:
    for x in range(300):
#    while True:
        try:
            
            population, popFitness = CreateNewGen(population, popFitness)
            
#            if x % 300 == 0:
#                DrawImage(population[0], imageX, imageY).save("img/"+str(x)+".png")
#                imList.append(DrawImage(population[0], imageX, imageY))
            gen += 1
            print(str(1-min(popFitness)) + " " + str(gen))
            
        except KeyboardInterrupt:
            #p.close()
            print("Bye!")
            #im.save("imageGenetic.png")
            #print("Writing pickle to disk")
            #utils.WriteObjPickle(imList, "imList-every300.pickle")
            break
    #print("Writing pickle to disk")
    #utils.WriteObjPickle(imList, "imList-every300.pickle")
