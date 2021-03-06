# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 10:56:22 2019

@author: user
"""

import cv2
import numpy as np

from PIL import ImageDraw
import pickle
    

def WriteFrames (imList, filename):
    
    imageX, imageY = imList[0].size
    
    video = cv2.VideoWriter(filename, -1, 20, (imageX, imageY))
    

    for i,im in enumerate(imList):
        imCopy = im.copy()
        draw = ImageDraw.Draw(imCopy)
        string = "Generation: " + str(i+1)
        draw.text((5,5), string, (255,255,255))
        video.write(np.asarray(imCopy))
        
    video.release()
    
    
def WriteObjPickle (obj, filename):
    pickle.dump(obj, open(filename, mode="wb"))
    
def ReadObjPickle (filename):
    return pickle.load(open(filename, mode="rb"))