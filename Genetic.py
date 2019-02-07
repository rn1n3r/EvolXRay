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


nPoly = 50
nV = 3 # Triangle
popSize = 30
grayscale = True
headless = True
hotStart = False

selectionPercent = 0.25
popSize = popSize - (popSize % math.floor(popSize*selectionPercent))
popSize = int(popSize)

imageX = refArray.shape[1]
imageY = refArray.shape[0]

#if len(refArray.shape) < 3:
#    refArray = np.dstack([refArray]*3)
#    
#if refArray.shape[2] > 3:
#    refArray = refArray[:,:,0:3]
if len(refArray.shape) < 3:
    refArray = np.dstack([refArray]*3)
    
if refArray.shape[2] >= 3 and grayscale:
    refArray = refArray[:,:,0]


if grayscale:
    uniqueRGB = 1
else:
    uniqueRGB = 3


def clamp (x, minVal, maxVal):
    return max(minVal, min(x, maxVal))


# Sum of differences
def fitness (indiv):
    
    image = DrawImage(indiv, imageX, imageY)
    
    if grayscale:
        imArray = np.asarray(image).astype(np.int16)[:,:,0]
        delta = np.abs(refArray - imArray)
        diff = np.sum(delta)/(255.0*imageX*imageY*uniqueRGB)
        
    else:
        imArray = np.asarray(image).astype(np.int16)
        diff = np.sum(np.abs(refArray - imArray))

    return diff, image

# The version that takes an image and avoids the DrawImage call
def fitness2(image):

    if grayscale:
        imArray = np.asarray(image).astype(np.int16)[:,:,0]
        delta = np.abs(refArray - imArray)
        diff = np.sum(delta)/(255.0*imageX*imageY*uniqueRGB)
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

# Returns list of fitness values for each image in population
def GetPopFitness(population, popSize, imageX, imageY, p):
    popFitness = [0]*popSize
    
    #if __name__ == '__main__':
        
        #popFitness = [p.apply_async(fitness2, args=(x,)) for x in population]
        #popFitness = [a.get() for a in popFitness]
        
        
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
    nPoly = 1000
    parents = int(math.floor(popSize*selectionPercent))
    
    # Calculate fittest individual from previous generation
    bestPrevFitness,_ = fitness(population[0])
    
    for i in range(0,parents):
        prevFit = popFitness[i]
        im = DrawImage(population[i], imageX, imageY)
        for j in range(0, popSize//parents):
            firstParent = i
            randomParent = random.randint(0, parents)
            #newPop.append(copy.deepcopy(population[firstParent]))
            newPop.append(copyPolyList(population[firstParent]))
            
            if len(newPop[-1]) < nPoly:
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
                newPop[-1].append(Polygon(initVertices, initRGBA, imageX, imageY, nV))
                
                
                newFit,_ = fitness(newPop[-1])
                

                if prevFit > newFit:

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
                            #print(str(currFit) + " " + str(initFit))
                            
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
                                    
                                    newPop[-1][-1].vertices[x][y] = clamp(initVal + z*15, 0, 255)
                                    
                                    im2 = im.copy()
                                    draw = ImageDraw.Draw(im2, 'RGBA')
                                    newPop[-1][-1].DrawOnImage(draw)
                                    
                                    currFit = fitness2(im2)
                                    
                                    if currFit >= newFit:
                                        newPop[-1][-1].vertices[x][y] = initVal
                                        break
                                    else:
                                        newFit = currFit
                                
                    
                   # im = im2.copy()
                else: 
                    del(newPop[-1][-1])
                    newFit = prevFit
                                    
            
            
            for k in range(0, len(population[i])):
               # if random.randint(0, 100) < 50:
                    #newPop[-1][k] = copy.deepcopy(population[randomParent][k])
                    # Mutation
                if len(allPoly) > 0:
                    
                    if random.randint(0, 100) < 1:
                        origPoly = copyPolyList(newPop[-1][k])
                        newPop[-1][k].SoftMutate()
                
                        currFit,_ = fitness(newPop[-1])
                    
                        if currFit >= newFit:
                            newPop[-1][k] = copyPolyList(origPoly)
                            #currFit = prevFit
                        else:
                            newFit = currFit
                            
            newPopFitness.append(newFit)                           
                                        

    
    
    newPop = sorted(list(zip(newPopFitness, newPop)), key=lambda x: x[0])
    newPopFitness = [x[0] for x in newPop]
    newPop = [x[1] for x in newPop]
   

    
    # If the fittest in the new generation is less fit than the best from before, replace it
    if bestPrevFitness < min(newPopFitness):
        newPop[0] = copyPolyList(population[0])
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
    global population, popFitness, gen
    
    population, popFitness = CreateNewGen(population, popFitness)
            
    im = DrawImage(population[0], imageX, imageY)

    #maxDiff = 255*imageX*imageY*uniqueRGB
    #print(1 - min(popFitness)/maxDiff)
    gen += 1
    print(str(1-min(popFitness)) + " " + str(gen))
    myobj.set_data(im)
    
    
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
    myobj = pyplot.imshow(im)
        
    ani = animation.FuncAnimation(fig, updateImg, interval = 0)
    pyplot.show()

else:
    for x in range(1000):
#    while True:
        try:
            
            population, popFitness = CreateNewGen(population, popFitness)
            
            imList.append(DrawImage(population[0], imageX, imageY))
            gen += 1
            print(str(1-min(popFitness)) + " " + str(gen))
            
        except KeyboardInterrupt:
            #p.close()
            print("Bye!")
            im.save("imageGenetic.png")
            break
    
