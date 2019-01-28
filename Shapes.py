#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 21:31:16 2018

@author: edward
"""
import random

def clamp (x, minVal, maxVal):
    return max(minVal, min(x, maxVal))

class Polygon:
    def __init__(self, vertices, fill, imageX, imageY, nVertices=3, grayscale=True):
        self.nVertices = nVertices
        self.vertices = vertices
        self.fill = fill
        self.imageX = imageX
        self.imageY = imageY
        self.grayscale = grayscale
        
    def DrawOnImage(self, ImageDrawer):
        ImageDrawer.polygon(tuple(tuple(x) for x in self.vertices), tuple(self.fill))

    def SoftMutate(self):
        
        delta = random.choice(range(-10,10))
        
        if not self.grayscale:
            mutateType = random.randint(0,5)
            
            if mutateType < 4:
                self.fill[mutateType] = clamp(self.fill[mutateType] + delta, 0, 255)
                
            else:
                vertex = random.randint(0, self.nVertices-1)
                if mutateType == 4:
                    limit = self.imageX
                else:
                    limit = self.imageY
                    
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
                    limit = self.imageX
                else:
                    limit = self.imageY
                    
                self.vertices[vertex][mutateType-2] = clamp(self.vertices[vertex][mutateType-2] + delta, 0, limit)
  
    def MediumMutate(self):
        if not self.grayscale:
            mutateType = random.randint(0,5)
            
            if mutateType < 4:
                self.fill[mutateType] = random.randint(0,255)
                
            else:
                vertex = random.randint(0, self.nVertices-1)
                if mutateType == 4:
                    limit = self.imageX
                else:
                    limit = self.imageY
                    
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
                    limit = self.imageX
                else:
                    limit = self.imageY
                    
                self.vertices[vertex][mutateType-2] = random.randint(0,limit)
                

class Circle:
    def __init__(self, centre, radius, fill, imageX, imageY, grayscale=True):
        self.centre = centre
        self.radius = radius
        self.fill = fill
        self.imageX = imageX
        self.imageY = imageY
        self.grayscale = grayscale
        
    def DrawOnImage(self, ImageDrawer):
        ImageDrawer.ellipse([self.centre[0]-self.radius, self.centre[1]-self.radius, self.centre[0]+self.radius, self.centre[1]+self.radius], tuple(self.fill), outline=None)

            
    def MediumMutate(self):
        if not self.grayscale:
            mutateType = random.randint(0,5)
            
            if mutateType < 4:
                self.fill[mutateType] = random.randint(0,255)
                
            else:
                vertex = random.randint(0, self.nVertices-1)
                if mutateType == 4:
                    limit = self.imageX
                else:
                    limit = self.imageY
                    
                self.vertices[vertex][mutateType-4] = random.randint(0,limit)

        else: # If the color is selected for change, change ALL RGB channels
            mutateType = random.randint(0,3)
            
            if mutateType == 0:
                val = random.randint(0,255)
                self.fill[0:3] = [val,val,val]
            elif mutateType == 1:
                self.fill[3] = random.randint(0,255)
            elif mutateType == 2:
                self.radius = random.randint(0, self.imageX)
            else:
                self.centre = [random.randint(0,self.imageX), random.randint(0,self.imageY)]
    