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

#random.seed(123)
random.seed(321)

# Load image 
refImg = Image.open('x-ray.jpg')
refImg.load()
refArray = np.asarray(refImg).astype(np.int16)

if len(refArray.shape) < 3:
    refArray = np.dstack([refArray]*3)
    
if refArray.shape[2] > 3:
    refArray = refArray[:,:,0:3]


imageX = refArray.shape[1]
imageY = refArray.shape[0]
nPoly =50
nV = 3 # Triangle
grayscale = True
popSize = 70


def clamp (x, minVal, maxVal):
    return max(minVal, min(x, maxVal))

class Polygon:
    def __init__(self, vertices, fill, nVertices=nV):
        self.nVertices = nVertices
        self.vertices = vertices
        self.fill = fill
        
    def DrawOnImage(self, ImageDrawer):
        ImageDrawer.polygon(tuple(tuple(x) for x in self.vertices), tuple(self.fill))

    def SoftMutate(self):
        
        delta = random.choice(range(-20,20))
        
        if not grayscale:
            mutateType = random.randint(0,5)
            
            if mutateType < 4:
                self.fill[mutateType] = clamp(self.fill[mutateType] + delta, 0, 255)
                
            else:
                vertex = random.randint(0, self.nVertices-1)
                if mutateType == 4:
                    limit = imageX
                else:
                    limit = imageY
                    
                self.vertices[vertex][mutateType-4] = clamp(self.vertices[vertex][mutateType-4] + delta, 0, limit)
        
        else:
            mutateType = random.randint(0,3)
            
            if mutateType == 0:
                self.fill[0:3] = [clamp(self.fill[0] + delta, 0, 255) for _ in range(3)]
            elif mutateType == 1:
                self.fill[3] = clamp(self.fill[3] + delta, 0, 255)
            else:
                vertex = random.randint(0, self.nVertices-1)
                if mutateType == 2:
                    limit = imageX
                else:
                    limit = imageY
                    
                self.vertices[vertex][mutateType-2] = clamp(self.vertices[vertex][mutateType-2] + delta, 0, limit)
            
    def MediumMutate(self):
        if not grayscale:
            mutateType = random.randint(0,5)
            
            if mutateType < 4:
                self.fill[mutateType] = random.randint(0,255)
                
            else:
                vertex = random.randint(0, self.nVertices-1)
                if mutateType == 4:
                    limit = imageX
                else:
                    limit = imageY
                    
                self.vertices[vertex][mutateType-4] = random.randint(0,limit)

        else: # If the color is selected for change, change ALL RGB channels
            mutateType = random.randint(0,3)
            
            if mutateType == 0:
                val = random.randint(0,255)
                self.fill[0:3] = [val,val,val]
            elif mutateType == 1:
                self.fill[3] = random.randint(0,255)
            else:
                vertex = random.randint(0, self.nVertices-1)
                if mutateType == 2:
                    limit = imageX
                else:
                    limit = imageY
                    
                self.vertices[vertex][mutateType-2] = random.randint(0,limit)


# Sum of differences
def fitness(image):
    
    if grayscale:
        imArray = np.asarray(image).astype(np.int16)[:,:,0]
        diff = np.sum(np.abs(refArray[:,:,0] - imArray))
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
def GetPopFitness(population, popSize, imageX, imageY):
    popFitness = [0]*popSize
    for i, indiv in enumerate(population):
        im = DrawImage(indiv, imageX, imageY)
        
        popFitness[i] = fitness(im)
    
    return popFitness


def CreateNewGen(population, popFitness):
    # Sort by fitness
    population = sorted(list(zip(popFitness, population)), key=lambda x: x[0])
    population = [x[1] for x in population]
    
    newPop = [0]*len(population)
    
    elite = 5
    newPop[0:elite] = copy.deepcopy(population[0:elite]) # ELITISM
    
#    for j in range(2,8):
#        if random.randint(0,100) < 10:
#            for x in newPop[j]:
#                x.SoftMutate()
    
    maxDiff = 255*256*256*3
    
    # For the rest of the new population, use roulette wheel selection to select two parents
    for j in range(elite,popSize):
        
        S = popSize*maxDiff - sum(popFitness)
    
        parentIndex = []
        for k in range(2):
            r = random.randint(0, S)
            s = 0
            index = 0
            while s <= r:
                s = s + maxDiff - popFitness[index]
                if s > r:
                    parentIndex.append(index)
                index = index + 1
                
        # CROSSOVER!!
        random.shuffle(parentIndex)
        newPop[j] = copy.deepcopy(population[parentIndex[1]])
        
        if random.randint(0,100) < 100:
            
            for i in range(0, nPoly-1):
                if random.randint(0,1) == 1:
                    newPop[j][i] = copy.deepcopy(population[parentIndex[0]][i])
#            a = random.randint(0, nPoly-1)
#            b = random.randint(0, nPoly-1)
#            if a > b:
#                a,b = b,a
#                
#            for i in range(a,b):
#                newPop[j][i] = copy.deepcopy(population[parentIndex[0]][i])
        
        # Mutation rate
        if random.randint(0,100) < 1:
            for x in newPop[j]:
                x.SoftMutate()
            
            
    return newPop
        
# Initialize first population
population = list()

for x in range(popSize):
    initVertices = [[[random.randint(0,imageX), random.randint(0,imageY)] for _ in range(nV)] for _ in range(nPoly)] 
    
    initRGBA = []
    for i in range(nPoly):    
        initGray = random.randint(0,255)
        #initRGBA.append([initGray, initGray, initGray, random.randint(0,255)])
        initRGBA.append([initGray, initGray, initGray, random.randint(0,255)])
    #initRGBA = [[random.randint(0,255) for _ in range(4)]for _ in range(nPoly)]
    allPoly = [Polygon(initVertices[i], initRGBA[i]) for i in range(nPoly)]
    population.append(allPoly)


popFitness = GetPopFitness(population, popSize, imageX, imageY)


#fig = pyplot.figure()
im = DrawImage(population[0], imageX, imageY)

#myobj = pyplot.imshow(im)

while True:
    try:
        population = CreateNewGen(population, popFitness)
        popFitness = GetPopFitness(population, popSize, imageX, imageY)
        population = sorted(list(zip(popFitness, population)), key=lambda x: x[0])
        population = [x[1] for x in population]
        popFitness.sort()
        
        
        im = DrawImage(population[0], imageX, imageY)
        
        print(1 - min(popFitness)/(255.0*imageX*imageY*1))
        #myobj.set_data(im)
        
    except KeyboardInterrupt:
        print("Bye!")
        im.save("imageGenetic.png")
        break
    
def updateImg(i):
    global population, popFitness
    
    population = CreateNewGen(population, popFitness)
    popFitness = GetPopFitness(population, popSize, imageX, imageY)
    population = sorted(list(zip(popFitness, population)), key=lambda x: x[0])
    population = [x[1] for x in population]
    popFitness.sort()
    
    
    im = DrawImage(population[0], imageX, imageY)
    
    print(1 - min(popFitness)/(255.0*imageX*imageY*1))
    myobj.set_data(im)
    
    
    return myobj

#ani = animation.FuncAnimation(fig, updateImg, interval = 0)
#pyplot.show()
    