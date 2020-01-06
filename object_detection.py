import numpy as np
import cv2
import sys
import pandas as pd
import pickle
import math as m
import tkinter
import yaml
import csv

with open("config.yml", "r") as stream:
    config_data= yaml.safe_load(stream)
path = config_data["root_dir"]
grabsize = (int(config_data["annotation_size"]))
minTh = int(config_data["Tmin"])
maxTh = int(config_data["Tmax"])
runtype=int(config_data["run"])
#Open video fileimport Tkinter
from tkinter.filedialog import askopenfilename
#Open the video file which needs to be processed
root = tkinter.Tk()

#get screen resolution
screen_width = int(root.winfo_screenwidth())
screen_height = int(root.winfo_screenheight())

movieName =  askopenfilename(filetypes=[("Video files","*")])
cap = cv2.VideoCapture(movieName)

nframe =cap.get(cv2.CAP_PROP_FRAME_COUNT)
nx = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
ny = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


#Create data frames to store x and y coords of the identified blobs, rows for each individual column for each frame
df = pd.DataFrame(columns=['c_id','x_px','y_px','frame'])
data = pd.DataFrame([])
i=0
row = 0
steps= int(input("Enter the number of frames to skip, eg 5 if you want to track every 6th frame: ")  )

#Load model
from keras.models import load_model
if runtype == 1:
  bb_model = load_model(path + "/classifiers/mothe_model.h5py")
else if runtype == 0:
  modelname = input("Enter data name (for wasp data enter wasp, for blackbuck enter bb):")
  if modelname == "wasp":
     bb_model = load_model(path + "/classifiers/wasp_model.h5py")
  else if modelname == "bb":
     bb_model = load_model(path + "/classifiers/bb_model.h5py")
#Video writer object
out = cv2.VideoWriter(path + '/mothe_detect.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 30, (nx,ny))

while(cap.isOpened() & (i<(nframe-steps))):

  i = i + steps
  print("[UPDATING.....]{}th frame detected and stored".format(i))
  cap.set(cv2.CAP_PROP_POS_FRAMES,i)
  ret, frame = cap.read()
  grayF = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  #Equalize image
  #gray = cv2.equalizeHist(gray)
  #remove noise
  gray = cv2.medianBlur(grayF,5)
  #Invert image
  gray = cv2.bitwise_not(gray)

  # Blob detection
  # Setup SimpleBlobDetector parameters.
  params = cv2.SimpleBlobDetector_Params()

  # Change thresholds
  params.minThreshold = minTh;
  params.maxThreshold = maxTh;

  # Filter by Circularity
  params.filterByCircularity = False
  #params.minCircularity = 0.1

  # Filter by Convexity
  params.filterByConvexity = False
  #params.minConvexity = 0.87

  # Filter by Inertia
  params.filterByInertia = False

  # Create a detector with the parameters
  ver = (cv2.__version__).split('.')
  if int(ver[0]) < 3 :
    detector = cv2.SimpleBlobDetector(params)
  else :
    detector = cv2.SimpleBlobDetector_create(params)

  # Detect blobs.
  keypoints = detector.detect(gray)
  import csv
  testX = np.ndarray(shape=(len(keypoints),40,40,3), dtype='uint8', order='C')
  j = 0
  for keyPoint in keypoints:

    ix = keyPoint.pt[0]
    iy = keyPoint.pt[1]
    tmpImg=frame[max(0,int(iy-grabsize)):min(ny,int(iy+grabsize)), max(0,int(ix-grabsize)):min(nx,int(ix+grabsize))].copy()

    tmpImg1=cv2.resize(tmpImg,(40,40))
    testX[j,:,:,:]=tmpImg1
    j = j + 1
  testX = testX.reshape(-1, 40,40, 3)
  testX = testX.astype('float32')
  testX = testX / 255.
  pred = bb_model.predict(testX)
  Pclass = np.argmax(np.round(pred),axis=1)
  j=0
  indx=[]
  FKP = []
  for pr in Pclass:
      if pr == 1:
          row = row + 1
          df.loc[row] = [j, keypoints[j].pt[0],keypoints[j].pt[1], i]
          FKP.append(keypoints[j])
          indx.append(j)

      j=j+1
  pts=[(m.floor(i.pt[0]), m.floor(i.pt[1])) for i in FKP]

  for item in pts:
    data = data.append(pd.DataFrame({'frame': i, 'x': item[0], 'y': item[1],}, index=[0]), ignore_index=True)
    cv2.rectangle(frame,(item[0]-grabsize, item[1]-grabsize), (item[0]+grabsize, item[1]+grabsize),(0,255,0),thickness = 2)
  out.write(frame)
data.to_csv(path + "detect.csv")
cap.release()
out.release()
