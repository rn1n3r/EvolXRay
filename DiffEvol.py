#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Differential Evolution
@author: edward
"""

from PIL import Image, ImageDraw
from matplotlib.pyplot import imshow, figure
from matplotlib import animation, pyplot
import numpy as np
import random
import copy

from Shapes import Polygon
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


nPoly = 250
nV = 3 # Triangle
popSize = 30
grayscale = True
headless = True

imageX = refArray.shape[1]
imageY = refArray.shape[0]

if grayscale:
    uniqueRGB = 1
else:
    uniqueRGB = 3


def clamp (x, minVal, maxVal):
    return max(minVal, min(x, maxVal))


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

# Differential Evolution
def CreateNewGen(population, popFitness):
    # Sort by fitness
    population = sorted(list(zip(popFitness, population)), key=lambda x: x[0])
    population = [x[1] for x in population]
    
    newPop = [0]*len(population)
    F = 0.5
    
    # For each individual in the population
    for i,x in enumerate(population):
        
        indivIDcurr = [ids for ids in range(0, popSize)]
        indivIDcurr.remove(i)
       
        # Pick three distinct individuals (vectors)
        r1, r2, r3 = random.sample(indivIDcurr, 3)

        # v = r1 + F*(r2 - r3)
        # v = r1 + F*(rbest - r2)
        v = copy.deepcopy(population[r1])
        randInd = random.randint(0,nPoly)
        
        # Iterate through the triangles in the donor vector
        for j,y in enumerate(v):
            # Add the vertices (?)
            y.vertices = np.ndarray.tolist(np.array(y.vertices) + \
                                           np.round(0.5*F*(np.array(population[0][j].vertices) - \
                                           np.array(population[r2][j].vertices))))
            y.vertices = [[clamp(t, 0, imageX) for t in verts] for verts in y.vertices]
            
            y.fill = np.ndarray.tolist(np.array(y.fill) + \
                                           np.round(F*(np.array(population[0][j].fill) - \
                                           np.array(population[r2][j].fill))))
#            y.fill = np.ndarray.tolist(np.array(y.fill) + \
#                                           np.round(F*(np.array(population[r2][j].fill) - \
#                                           np.array(population[r3][j].fill))))
#            
            y.fill = [int(clamp(fills, 0, 255)) for fills in y.fill]
            
            
            # Crossover
            # Cross individual vertices over
            for q in range(0, len(y.vertices)):
                if random.randint(0, 100) < 40 or j == randInd:
                    y.vertices[q] = x[j].vertices[q]
            
            # Cross the fill over (assume grayscale for now, so the first three values should be kept the same)
            if random.randint(0,100) < 40 or j == randInd:
                y.fill[0:3] = x[j].fill[0:3]
                
            # Cross the transparency over (last element of y.fill)
            if random.randint(0,100) < 40 or j == randInd:
                y.fill[3] = x[j].fill[3]
                
        if popFitness[i] > fitness(DrawImage(v, imageX, imageY)):
            newPop[i] = copy.deepcopy(v)
        else:
            newPop[i] = copy.deepcopy(x)    
            
    
    return newPop


def updateImg(i):
    global population, popFitness
    
    population = CreateNewGen(population, popFitness)
    popFitness = GetPopFitness(population, popSize, imageX, imageY)
    population = sorted(list(zip(popFitness, population)), key=lambda x: x[0])
    population = [x[1] for x in population]
    popFitness.sort()
    
    
    im = DrawImage(population[0], imageX, imageY)

    maxDiff = 255.0*imageX*imageY*uniqueRGB
    print(1 - min(popFitness)/maxDiff)
    myobj.set_data(im)
    
    
    return myobj
        
# Initialize first population
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


popFitness = GetPopFitness(population, popSize, imageX, imageY)


im = DrawImage(population[0], imageX, imageY)

if not headless:
    fig = pyplot.figure()
    myobj = pyplot.imshow(im)
        
    ani = animation.FuncAnimation(fig, updateImg, interval = 0)
    pyplot.show()

else:
    while True:
        try:
            population = CreateNewGen(population, popFitness)
            popFitness = GetPopFitness(population, popSize, imageX, imageY)
            population = sorted(list(zip(popFitness, population)), key=lambda x: x[0])
            population = [x[1] for x in population]
            popFitness.sort()
            
            
            im = DrawImage(population[0], imageX, imageY)

               
            maxDiff = 255.0*imageX*imageY*uniqueRGB
            print(1 - min(popFitness)/maxDiff)
            
        except KeyboardInterrupt:
            print("Bye!")
            im.save("imageDiffEvol.png")
            break
    