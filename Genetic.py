#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Genetic alg version
@author: edward
"""

from PIL import Image, ImageDraw
from matplotlib.pyplot import imshow, figure
from matplotlib import animation, pyplot
import numpy as np
import random
import copy
import math
from skimage.measure import compare_ssim

from multiprocessing import Pool

from Shapes import Polygon
#random.seed(123)
random.seed(321)

# Load image 
refImg = Image.open('x-ray2.jpg')
refImg.load()
refArray = np.asarray(refImg).astype(np.int16)

if len(refArray.shape) < 3:
    refArray = np.dstack([refArray]*3)
    
if refArray.shape[2] > 3:
    refArray = refArray[:,:,0:3]


nPoly = 50
nV = 3 # Triangle
popSize = 30
grayscale = True
headless = False
hotStart = False

selectionPercent = 0.25
popSize = popSize - (popSize % math.floor(popSize*selectionPercent))

imageX = refArray.shape[1]
imageY = refArray.shape[0]

if grayscale:
    uniqueRGB = 1
else:
    uniqueRGB = 3


def clamp (x, minVal, maxVal):
    return max(minVal, min(x, maxVal))


# Sum of differences
def fitness(indiv):
    
    image = DrawImage(indiv, imageX, imageY)
    
    if grayscale:
        imArray = np.asarray(image).astype(np.int16)[:,:,0]
        delta = np.abs(refArray[:,:,0] - imArray)
        diff = np.sum(delta)/(255*imageX*imageY*uniqueRGB)
        diff = diff# + (1-compare_ssim(refArray[:,:,0], np.asarray(image).astype(np.int16)[:,:,0]))#np.sum(delta)
   
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

# Returns list of fitness values for each image in population
def GetPopFitness(population, popSize, imageX, imageY, p):
    popFitness = [0]*popSize
    
    #if __name__ == '__main__':
        
        #popFitness = [p.apply_async(fitness2, args=(x,)) for x in population]
        #popFitness = [a.get() for a in popFitness]
        
        
    for i, indiv in enumerate(population):
        popFitness[i] = fitness(indiv)
        
    
    return popFitness


def CreateNewGen(population):
    # Sort by fitness
#    population = sorted(list(zip(popFitness, population)), key=lambda x: x[0])
#    population = [x[1] for x in population]
#    
    newPop = []
    
    parents = int(math.floor(popSize*selectionPercent))
    
    # Calculate fittest individual from previous generation
    bestCandidate = copy.deepcopy(population[0])
    bestPrevFitness = fitness(bestCandidate)
    
    for i in range(0,parents):
        for j in range(0, popSize//parents):
            firstParent = i
            randomParent = random.randint(0, parents)
            newPop.append(copy.deepcopy(population[firstParent]))
            for k in range(0, nPoly-1):
               # if random.randint(0, 100) < 50:
                    #newPop[-1][k] = copy.deepcopy(population[randomParent][k])
                    
                if random.randint(0, 100) < 1:
                    newPop[-1][k].SoftMutate()
                    
    newPopFitness = GetPopFitness(newPop, popSize, imageX, imageY, 1)
    
    newPop = sorted(list(zip(newPopFitness, newPop)), key=lambda x: x[0])
    newPop = [x[1] for x in newPop]
    
    # If the fittest in the new generation is less fit than the best from before, replace it
    if bestPrevFitness < min(newPopFitness):
        newPop[0] = copy.deepcopy(bestCandidate)
        newPopFitness[newPopFitness.index(min(newPopFitness))] = bestPrevFitness
        
#  
#    elite = 5
#    newPop[0:elite] = copy.deepcopy(population[0:elite]) # ELITISM
#    
##    for j in range(2,8):
##        if random.randint(0,100) < 10:
##            for x in newPop[j]:
##                x.SoftMutate()
#    
#    maxDiff = 255*imageX*imageY*uniqueRGB
#    
#    # For the rest of the new population, use roulette wheel selection to select two parents
#    for j in range(elite,popSize):
#        
#        S = popSize*maxDiff - sum(popFitness)
#    
#        parentIndex = []
#        for k in range(2):
#            r = random.randint(0, S)
#            s = 0
#            index = 0
#            while s <= r:
#                s = s + maxDiff - popFitness[index]
#                if s > r:
#                    parentIndex.append(index)
#                index = index + 1
#                
#        # CROSSOVER!!
#        random.shuffle(parentIndex)
#        newPop[j] = copy.deepcopy(population[parentIndex[1]])
#        
#        if random.randint(0,100) < 100:
#            
#            for i in range(0, nPoly-1):
#                if random.randint(0,1) == 1:
#                    newPop[j][i] = copy.deepcopy(population[parentIndex[0]][i])
##            a = random.randint(0, nPoly-1)
##            b = random.randint(0, nPoly-1)
##            if a > b:
##                a,b = b,a
##                
##            for i in range(a,b):
##                newPop[j][i] = copy.deepcopy(population[parentIndex[0]][i])
#        
#        # Mutation rate
#        if random.randint(0,100) < 1:
#            for x in newPop[j]:
#                x.SoftMutate()
            
            
    return newPop, newPopFitness

def updateImg(i):
    global population, popFitness
    
    population, popFitness = CreateNewGen(population)
            
    im = DrawImage(population[0], imageX, imageY)

       
    maxDiff = 255.0*imageX*imageY*uniqueRGB
  

    #maxDiff = 255*imageX*imageY*uniqueRGB
    #print(1 - min(popFitness)/maxDiff)
    print(1-min(popFitness))
    myobj.set_data(im)
    
    
    return myobj
        
# Initialize first population
    
if not hotStart:
    population = list()
    
    for x in range(popSize):
        initVertices = [[[random.randint(0,imageX), random.randint(0,imageY)] for _ in range(nV)] for _ in range(nPoly)] 
        
        initRGBA = []
        for i in range(nPoly):
            if grayscale:
                initGray = random.randint(0,255)
                initRGBA.append([initGray, initGray, initGray, random.randint(0,255)])
            else:
                initRGBA.append([random.randint(0,255), random.randint(0,255), random.randint(0,255), random.randint(0,255)])
        #initRGBA = [[random.randint(0,255) for _ in range(4)]for _ in range(nPoly)]
        allPoly = [Polygon(initVertices[i], initRGBA[i], imageX, imageY, nV, grayscale) for i in range(nPoly)]
        population.append(allPoly)

#p = Pool(4)
p = 1
popFitness = GetPopFitness(population, popSize, imageX, imageY, p)


im = DrawImage(population[0], imageX, imageY)

if not headless:
    fig = pyplot.figure()
    myobj = pyplot.imshow(im)
        
    ani = animation.FuncAnimation(fig, updateImg, interval = 0)
    pyplot.show()

else:
    
    while True:
        try:
            
            population, popFitness = CreateNewGen(population)
            
            im = DrawImage(population[0], imageX, imageY)

               
            maxDiff = 255.0*imageX*imageY*uniqueRGB
            print(1 - min(popFitness)/maxDiff)
            
        except KeyboardInterrupt:
            #p.close()
            print("Bye!")
            im.save("imageGenetic.png")
            break
    