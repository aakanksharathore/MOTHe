import numpy as np
import cv2
import sys
import pandas as pd
import pickle
import math as m
import tkinter
import csv, glob
import os, sys
import yaml
import argparse as ap
import io
from tkinter.filedialog import askopenfilename


# initialize the list of reference points and boolean indicating
# whether cropping is being performed or not
refPt = []
cropping = False

def click_and_crop(event, x, y, flags, param):
	# grab references to the global variables
	global refPt, cropping

	# if the left mouse button was clicked, record the starting
	# (x, y) coordinates and indicate that cropping is being
	# performed
	if event == cv2.EVENT_LBUTTONDOWN:
		refPt = [(x, y)]
		cropping = True

	# check to see if the left mouse button was released
	elif event == cv2.EVENT_LBUTTONUP:
		# record the ending (x, y) coordinates and indicate that
		# the cropping operation is finished
		refPt.append((x, y))
		cropping = False

		# draw a rectangle around the region of interest
		cv2.rectangle(image, refPt[0], refPt[1], (0, 255, 0), 2)
		cv2.imshow("image", image)
# Add the configuration items for the user required to run MOTHe into a yaml file
specifications= {}
root_dir= input("[REQUIRED.....] Enter the path to the MOTHe repository on your local machine (Ex: /home/user/Desktop/MOTHe): ")
specifications.__setitem__('root_dir', root_dir)
vid_dir= input("[REQUIRED.....] Enter the path to the videos directory on your local machine (Ex: /home/user/Desktop/Videos): ")
specifications.__setitem__('vid_dir', vid_dir)
runtype= input("[REQUIRED.....] Are you running for a new data or testing on blackbuck/wasp? Enter 1 for new data 0 for blackbuck or wasp: ")
specifications.__setitem__('run', runtype)

# Add the dimensions of the bounding box based on user requirement
root = tkinter.Tk()
movieName =  askopenfilename(filetypes=[("Video files","*")])
cap = cv2.VideoCapture(movieName)
i=0
steps=50
nframe =cap.get(cv2.CAP_PROP_FRAME_COUNT)
while(cap.isOpened() & (i<(nframe-steps))):
    i = i + steps
    print("[REQUIRED.....] Click and drag the mouse across the area of interest. In case you want to navigate to a different frame, press 'k'")
    cap.set(cv2.CAP_PROP_POS_FRAMES,i)

    ret, image = cap.read()
    clone = image.copy()
    cv2.namedWindow("image")
    cv2.setMouseCallback("image", click_and_crop)
    while True:
        # display the image and wait for a keypress
        cv2.imshow("image", image)
        key = cv2.waitKey(1) & 0xFF

        # if the 'r' key is pressed, reset the cropping region
        if key == ord("r"):
            frame = clone.copy()

        # if the 'c' key is pressed, break from the loop
        elif key == ord("c"):
            break

    # if there are two reference points, then crop the region of interest
    # from teh image and display it
    if len(refPt) == 2:
        roi = clone[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]]
        if (refPt[0][0]-refPt[0][1]) > (refPt[1][0]-refPt[1][1]):
            specifications.__setitem__('annotation_size', ((refPt[0][0]-refPt[0][1])/4))
        else:
            specifications.__setitem__('annotation_size', ((refPt[1][0]-refPt[1][1])/4))
        cv2.imshow("ROI", roi)
        cv2.waitKey(0)

    # close all open windows
    cv2.destroyAllWindows()
    break
with io.open(specifications["root_dir"]+"/config.yml","r") as yamlfile:
        cur_yaml = yaml.safe_load(yamlfile)
        cur_yaml = {} if cur_yaml is None else cur_yaml
        cur_yaml.update(specifications)
        print(cur_yaml)
with io.open("config.yml", "w", encoding= "utf8") as outfile:
    yaml.safe_dump(cur_yaml, outfile, default_flow_style= False, allow_unicode= True)

