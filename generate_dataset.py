# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 10:58:41 2017

@author: aakanksha

Code to create training samples from videos by clicking on images
"""


import numpy as np
import cv2
import sys
import pandas as pd
import pickle
import math as m
import tkinter
import csv, glob
import yaml
import os, sys
from tkinter.filedialog import askopenfilename
import ntpath

with open("config.yml", "r") as stream:
    config_data= yaml.safe_load(stream)
path = config_data["root_dir"]
grabSize = int(config_data["annotation_size"]) #int(m.ceil((100.0/alt)*12))

ix,iy = -1,-1

def click_bb(event, x, y,flags,param):
    # grab references to the global variables
    global ref #PtX,PtY 
    # if the left mouse button was clicked, record the starting
    # (x, y) coordinates and indicate that cropping is being
    # performed
    if event == cv2.EVENT_LBUTTONDOWN:
          ref = [(x,y)]
          cv2.circle(fr, ref[0],32, (0, 255, 0), 2)
          
    else:
          ref = None

#Category of training data  
tr_set = input("Enter \"yes\" if creating animal class samples, \"no\"  if creating background class samples ")   
step = int(sys.argv[1])    #Number of frames to skip

#Open the video file which needs to be processed     
root = tkinter.Tk()
movieName =  askopenfilename(filetypes=[("Video files","*")])#filedialog.askopenfilename(filetypes=[("Video files","*")])
head, tail = ntpath.split(movieName)
cap = cv2.VideoCapture(movieName)
nframe =cap.get(cv2.CAP_PROP_FRAME_COUNT)
   
nx = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
ny = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

frName = "Click on blackbucks/background, press n for next frame and press q when done"
cv2.destroyAllWindows()


i=1
df_bb = pd.DataFrame(columns=['x_px','y_px'])     #Store blackbuck coordinates

#cv2.namedWindow(frName)

counter=0

while(cap.isOpened()):

    if (cv2.waitKey(1) & 0xFF == ord('q')) or i > nframe:
       break
    cap.set(cv2.CAP_PROP_POS_FRAMES,i)
    ret, fr = cap.read()
    i= i+step
    cv2.namedWindow(frName, flags =  cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(frName,click_bb)
    #Capture clicks on the image, fclick on blackbuck
    df_bb=df_bb[0:0]
    ref=None
    while(1):
        
        cv2.imshow(frName,fr)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('n'):
            break
        elif key == ord('q'):
            i = nframe+100
            break
        elif not(not(ref)):
             df_bb = df_bb.append({'x_px':ref[0][0],'y_px':ref[0][1]},ignore_index=True)
             
    
     
    df_bb=df_bb.drop_duplicates()           #Drop duplicates from the data frame
    df_bb = df_bb.reset_index(drop=True)    #Reindexing the new data frame after dropped values   


    #save training data in yes or no folder

    #Write images
    
    for k in range(0,(len(df_bb)-1)):
            
            counter = counter+1
            ix = int(df_bb.loc[k][0])
            iy = int(df_bb.loc[k][1])
            tmpImg =  fr[max(0,iy-grabSize):min(ny,iy+grabSize), max(0,ix-grabSize):min(nx,ix+grabSize)].copy()
            if tr_set == "yes":
              cv2.imwrite(path + "/yes/"+ tail[0:(len(tail)-4)] + '_' + str(counter) + ".png", cv2.resize(tmpImg,(40,40)))
            elif tr_set == "no":
              cv2.imwrite(path + "/no/no{}.png".format(counter), cv2.resize(tmpImg,(40,40)))

cv2.destroyAllWindows()

  
