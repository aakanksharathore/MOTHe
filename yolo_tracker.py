"""
    SORT: A Simple, Online and Realtime Tracker
    Copyright (C) 2016 Alex Bewley alex@dynamicdetection.com

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
from __future__ import print_function

import sys
import os.path
import scipy
import numpy as np
#import matplotlib.pyplot as plt
#import matplotlib.patches as patches
#from skimage import io
from sklearn.utils.linear_assignment_ import linear_assignment
from scipy.optimize import linear_sum_assignment
import glob
import time
import argparse
from filterpy.kalman import KalmanFilter
from filterpy.stats import mahalanobis
import filterpy

def _interval_overlap(interval_a, interval_b):
    x1, x2 = interval_a
    x3, x4 = interval_b

    if x3 < x1:
        if x4 < x1:
            return 0
        else:
            return min(x2,x4) - x1
    else:
        if x2 < x3:
             return 0
        else:
            return min(x2,x4) - x3

def bbox_iou(box1, box2):

    intersect_w = _interval_overlap([box1[0], box1[2]], [box2[0], box2[2]])
    intersect_h = _interval_overlap([box1[1], box1[3]], [box2[1], box2[3]])

    intersect = intersect_w * intersect_h

    w1, h1 = box1[2]-box1[0], box1[3]-box1[1]
    w2, h2 = box2[2]-box2[0], box2[3]-box2[1]

    union = w1*h1 + w2*h2 - intersect

    return float(intersect) / union

def do_nms(new_boxes, nms_thresh):
    # do nms
    sorted_indices = np.argsort(-new_boxes[:,4])
    boxes=new_boxes.tolist()

    for i in range(len(sorted_indices)):

        index_i = sorted_indices[i]

        if new_boxes[index_i,4] == 0: continue

        for j in range(i+1, len(sorted_indices)):
            index_j = sorted_indices[j]
            # anything with certainty above 1 is untouchable
            if boxes[index_j][4]>1:
                continue
            if bbox_iou(boxes[index_i][0:4], boxes[index_j][0:4]) > nms_thresh:
                new_boxes[index_j,4] = 0

    return

def convert_bbox_to_kfx(bbox):
    """
    Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
    [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
    the aspect ratio
    """
    w = bbox[2]-bbox[0]
    h = bbox[3]-bbox[1]
    x = bbox[0]+w/2.
    y = bbox[1]+h/2.
    return np.array([x,y,w,h]).reshape((4,1))

def convert_kfx_to_bbox(x):
    """
    Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
    [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
    """
    w=max(0.0,x[2])
    h=max(0.0,x[3])
    return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.]).reshape((1,4))


class KalmanBoxTracker(object):
    """
    This class represents the internel state of individual tracked objects observed as bbox.
    """
    count = 0
    def __init__(self,bbox):
        """
        Initialises a tracker using initial bounding box.
        """
        #define constant velocity model
        self.kf = KalmanFilter(dim_x=8, dim_z=4)
        self.kf.F = np.array([[1,0,0,0,1,0,0.5,0],[0,1,0,0,0,1,0,0.5],[0,0,1,0,0,0,0,0],[0,0,0,1,0,0,0,0],[0,0,0,0,1,0,1,0],[0,0,0,0,0,1,0,1],[0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,1]])
        self.kf.H = np.array([[1,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0],[0,0,1,0,0,0,0,0],[0,0,0,1,0,0,0,0]])

        self.kf.R[:,:] *= 25.0 # set measurement uncertainty for positions
        self.kf.Q[:2,:2] = 0.0 # process uncertainty for positions is zero - only moves due to velocity, leave process for width height as 1 to account for turning
        self.kf.Q[2:4,2:4] *= 0.1 # process uncertainty for width/height for turning
        self.kf.Q[4:6,4:6] = 0.0 # process uncertainty for velocities is zeros - only accelerates due to accelerations
        self.kf.Q[6:,6:] *= 0.01 # process uncertainty for acceleration
        self.kf.P[4:,4:] *= 5.0 # maximum speed

        z=convert_bbox_to_kfx(bbox)
        self.kf.x[:4] = z
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.hits = 1
        self.hit_streak = 1
        self.age = 1
        self.score = bbox[4]

    def update(self,bbox):
        """
        Updates the state vector with observed bbox.
        """
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        self.score = (self.score*(self.hits-1.0)/float(self.hits)) + (bbox[4]/float(self.hits))
        z = convert_bbox_to_kfx(bbox)
        self.kf.update(z)

    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        self.kf.predict()
        self.age += 1
        if(self.time_since_update>0):
            self.hit_streak = 0
        self.time_since_update += 1

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        return convert_kfx_to_bbox(self.kf.x)

    def get_distance(self, y):
        """
        Returns the mahalanobis distance to the given point.
        """
        b1 = convert_kfx_to_bbox(self.kf.x[:4])[0]
        return (bbox_iou(b1,y))

def associate_detections_to_trackers(detections,trackers,iou_threshold = 0.3):
    """
    Assigns detections to tracked object (both represented as bounding boxes)
    Returns 3 lists of matches, unmatched_detections and unmatched_trackers
    """
    if(len(trackers)==0):
        return np.empty((0,2),dtype=int), np.arange(len(detections)), np.empty((0,5),dtype=int)

    iou_matrix = np.zeros((len(detections),len(trackers)),dtype=np.float32)
    id_matrix = np.zeros((len(detections),len(trackers)),dtype=np.float32)
    scale_id = 0.5

    for d,det in enumerate(detections):
        for t,trk in enumerate(trackers):
            trackBox = convert_kfx_to_bbox(trk.kf.x[:4])[0]
            iou_matrix[d,t] = bbox_iou(trackBox, det)
            id_matrix[d,t] = scale_id*det[4]

    matched_indices = linear_assignment(-iou_matrix-id_matrix)

    unmatched_detections = []
    for d,det in enumerate(detections):
        if(d not in matched_indices[:,0]):
            unmatched_detections.append(d)

    unmatched_trackers = []
    for t,trk in enumerate(trackers):
        if(t not in matched_indices[:,1]):
            unmatched_trackers.append(t)

    #filter out matched with low probability
    matches = []
    for m in matched_indices:
        if(iou_matrix[m[0],m[1]]<iou_threshold):
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1,2))

    if(len(matches)==0):
        matches = np.empty((0,2),dtype=int)
    else:
        matches = np.concatenate(matches,axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)



class yoloTracker(object):
    def __init__(self,max_age=1,track_threshold=0.5, init_threshold=0.9, init_nms=0.0,link_iou=0.3 ):
        """
        Sets key parameters for YOLOtrack
        """
        self.max_age = max_age # time since last detection to delete track
        self.trackers = []
        self.frame_count = 0
        self.track_threshold = track_threshold # only return tracks with average confidence above this value
        self.init_threshold = init_threshold # threshold confidence to initialise a track, note this is much higher than the detection threshold
        self.init_nms = init_nms # threshold overlap to initialise a track - set to 0 to only initialise if not overlapping another tracked detection
        self.link_iou = link_iou # only link tracks if the predicted box overlaps detection by this amount
        KalmanBoxTracker.count = 0

    def update(self,dets):
        """
        Params:
          dets - a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
        Requires: this method must be called once for each frame even with empty detections.
        Returns a similar array, where the last column is the object ID.

        NOTE: The number of objects returned may differ from the number of detections provided.
        """
        self.frame_count += 1
        #get predicted locations from existing trackers.
        trks = np.zeros((len(self.trackers),5))

        ret = []
        for t,trk in enumerate(self.trackers):
            self.trackers[t].predict()


        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets,self.trackers, self.link_iou)

        #update matched trackers with assigned detections
        for t,trk in enumerate(self.trackers):
            if(t not in unmatched_trks):
                d = matched[np.where(matched[:,1]==t)[0],0]
                trk.update(dets[d,:][0])
                dets[d,4]=2.0 # once assigned we set it to full certainty

        #add tracks to detection list
        for t,trk in enumerate(self.trackers):
            if(t in unmatched_trks):

                d = convert_kfx_to_bbox(trk.kf.x)[0]
                d = np.append(d,np.array([2]), axis=0)
                d = np.expand_dims(d,0)
                dets = np.append(dets,d, axis=0)

        if len(dets)>0:
            dets = dets[dets[:,4]>self.init_threshold]
            do_nms(dets,self.init_nms)
            dets= dets[dets[:,4]<1.1]
            dets= dets[dets[:,4]>0]

        for det in dets:
            trk = KalmanBoxTracker(det[:])
            self.trackers.append(trk)

        i = len(self.trackers)
        for trk in reversed(self.trackers):
            i -= 1
            #remove dead tracklet
            if(trk.time_since_update > self.max_age):
                self.trackers.pop(i)

        for trk in (self.trackers):
            d = convert_kfx_to_bbox(trk.kf.x)[0]
            if ((trk.time_since_update < 1) and (trk.score>self.track_threshold)):
                ret.append(np.concatenate((d,[trk.id])).reshape(1,-1))

        if(len(ret)>0):
            return np.concatenate(ret)
        return np.empty((0,5))
