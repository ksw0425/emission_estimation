#from __future__ import print_function

from numba import jit
import numpy as np
#from sklearn.utils.linear_assignment_ import linear_assignment
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter
import os

GPU_set = str(input("Set GPU : "))

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=GPU_set


import argparse
import imutils
import time


import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer, Lambda, LeakyReLU
from tensorflow.keras import backend as K
import random
import sys
import h5py
from itertools import compress

from PIL import Image, ImageFont, ImageDraw
from matplotlib import pyplot as plt
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb

import math
import cv2
import json
import ast

import warnings
import pickle

class_num = 5
num_classes = 7
Ls_1 = 1 - 0.2 + 0.2/num_classes
Ls_0 = 0.2/num_classes

h_, w_ = 512, 512
input_shape = (h_, w_)

line_color = [(0,0,255),(0,255,0),(255,0,0),(0,0,0),(255,128,128),(255,255,0),(128,128,128),(50,150,150),
             (0,0,255),(0,255,0),(255,0,0),(0,0,0),(255,128,128),(255,255,0),(128,128,128),(50,150,150),
             (0,0,255),(0,255,0),(255,0,0),(0,0,0),(255,128,128),(255,255,0),(128,128,128),(50,150,150)]


while True:
    PATH = os.path.abspath(str(input("Set Dir path : ")))
    if os.path.isdir(PATH) == True:
        break

log_dir = PATH + os.sep + 'Detect' + os.sep
print('Weights were saved here : ' + log_dir)
tr_dir = PATH + os.sep + 'Track' + os.sep
print('Tracking ouputs were saved here : ' + tr_dir)

while True:
    try:
        if os.path.isfile(log_dir + os.sep + 'trained_weights.h5') == True:
            weight_PATH = log_dir + 'trained_weights.h5'
            break
    except:
        weight_PATH = os.path.abspath(log_dir + str(input("Set weight : ")))
        if os.path.isfile(weight_PATH) == True:
            break
print(weight_PATH)

while True:
    vid_path = os.path.abspath(tr_dir + os.sep + 'input' + os.sep + str(input("Set video name : ")))
    if os.path.isfile(vid_path) == True:
        break


while True:
    speed_check = str(input("Turn on speed estimation [y/n] : "))
    if speed_check == 'y':
        speed_check = True
        break
    elif speed_check == 'n':
        speed_check = False
        break

while True:
    video_write = str(input("write the result video [y/n] : "))
    if video_write == 'y':
        video_write = True
        break
    elif video_write == 'n':
        video_write = False
        break

line_path = os.path.join(tr_dir, 'input', 'line.txt')  # 또는 절대 경로 지정
line_list = []

with open(line_path, 'r') as f:
    for line in f:
        try:
            line = line.strip()
            # 한 줄이 "(x1, y1), (x2, y2)" 형식일 때 그대로 튜플 2개로 나누기
            if line.count('(') == 2 and line.count(')') == 2:
                left, right = line.split('),')
                left = ast.literal_eval(left.strip() + ')')  # 보정해서 닫기
                right = ast.literal_eval(right.strip())
                line_list.append([left, right])
            else:
                print(f"⚠️ 무시된 줄: {line}")
        except Exception as e:
            print(f"⚠️ 파싱 실패: {line} → {e}")

line_num = len(line_list)
counter_t = np.zeros((line_num,class_num))

if speed_check == True:
    t_list = []
    wco_path = os.path.join(tr_dir, 'input', 'warp_coordinates.txt')
    with open(os.path.abspath(wco_path), 'r') as f:
        for line in f:
            try:
                # 안전하게 ')', 줄바꿈 처리까지 보정
                parts = [pt.strip() for pt in line.strip().split('),') if pt.strip()]
                coords = []
                for pt in parts:
                    if not pt.endswith(')'):
                        pt += ')'
                    coords.append(ast.literal_eval(pt))
                t_list.append(coords)
            except Exception as e:
                print(f"⚠️ warp_coordinates.txt 파싱 실패: {line.strip()} → {e}")

    pts_src = np.array(t_list[0])
    pts_dst = np.array(t_list[1])
    f_, status = cv2.findHomography(pts_src, pts_dst)

    mask_path = os.path.join(tr_dir, 'input', 'mask.png')
    MASK = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)




@jit
def iou(bb_test,bb_gt):
  """
  Computes IUO between two bboxes in the form [x1,y1,x2,y2]
  """
  xx1 = np.maximum(bb_test[0], bb_gt[0])
  yy1 = np.maximum(bb_test[1], bb_gt[1])
  xx2 = np.minimum(bb_test[2], bb_gt[2])
  yy2 = np.minimum(bb_test[3], bb_gt[3])
  w = np.maximum(0., xx2 - xx1)
  h = np.maximum(0., yy2 - yy1)
  wh = w * h
  o = wh / ((bb_test[2]-bb_test[0])*(bb_test[3]-bb_test[1])
    + (bb_gt[2]-bb_gt[0])*(bb_gt[3]-bb_gt[1]) - wh)
  return(o)


def bbox_count(bbox):
  cla = bbox[5]
  return np.array(cla).reshape((1,))


def convert_bbox_to_z(bbox):
  """
  Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
    [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
    the aspect ratio
  """
  w = bbox[2]-bbox[0]
  h = bbox[3]-bbox[1]
  x = bbox[0]+w/2.
  y = bbox[1]+h/2.
  s = w*h    #scale is just area
  r = w/float(h)
  return np.array([x,y,s,r]).reshape((4,1))

def convert_x_to_bbox(x,score=None):
  """
  Takes a bounding box in the center form [x,y,s,r] and returns it in the form
    [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
  """
  w = np.sqrt(x[2]*x[3])
  h = x[2]/w
  if(score==None):
    return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.]).reshape((1,4))
  else:
    return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.,score]).reshape((1,5))

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
    self.kf = KalmanFilter(dim_x=7, dim_z=4)
    self.kf.F = np.array([[1,0,0,0,1,0,0],[0,1,0,0,0,1,0],[0,0,1,0,0,0,1],[0,0,0,1,0,0,0],  [0,0,0,0,1,0,0],[0,0,0,0,0,1,0],[0,0,0,0,0,0,1]])
    self.kf.H = np.array([[1,0,0,0,0,0,0],[0,1,0,0,0,0,0],[0,0,1,0,0,0,0],[0,0,0,1,0,0,0]])

    self.kf.R[2:,2:] *= 5.
    self.kf.R[3:,3:] *= 0.01
    self.kf.P[4:,4:] *= 1000. #give high uncertainty to the unobservable initial velocities
    self.kf.P *= 10.
    self.kf.Q *= 10
    self.kf.Q[3:4,3:4] *= 0.01
    self.kf.Q[-1,-1] *= 0.002
    self.kf.Q[4:,4:] *= 0.002

    self.kf.x[:4] = convert_bbox_to_z(bbox)
    self.time_since_update = 0
    self.id = KalmanBoxTracker.count
    KalmanBoxTracker.count += 1
    self.cla = bbox_count(bbox)
    self.history = []
    self.hits = 0
    self.hit_streak = 0
    self.age = 0

  def update(self,bbox):
    """
    Updates the state vector with observed bbox.
    """
    self.time_since_update = 0
    self.history = []
    self.hits += 1
    self.hit_streak += 1
    self.kf.update(convert_bbox_to_z(bbox))
    self.cla = bbox_count(bbox)

  def predict(self):
    """
    Advances the state vector and returns the predicted bounding box estimate.
    """
    if((self.kf.x[6]+self.kf.x[2])<=0):
      self.kf.x[6] *= 0.0
    self.kf.predict()
    self.age += 1
    if(self.time_since_update>0):
      self.hit_streak = 0
    self.time_since_update += 1
    self.history.append(convert_x_to_bbox(self.kf.x))
    return self.history[-1]

  def get_state(self):
    """
    Returns the current bounding box estimate.
    """
    return convert_x_to_bbox(self.kf.x)

def associate_detections_to_trackers(detections,trackers,iou_threshold = 0.1):
  """
  Assigns detections to tracked object (both represented as bounding boxes)
  Returns 3 lists of matches, unmatched_detections and unmatched_trackers
  """
  if(len(trackers)==0) or (len(detections)==0):
    return np.empty((0,2),dtype=int), np.arange(len(detections)), np.empty((0,5),dtype=int)
  iou_matrix = np.zeros((len(detections),len(trackers)),dtype=np.float32)

  for d,det in enumerate(detections):
    for t,trk in enumerate(trackers):
      iou_matrix[d,t] = iou(det,trk)
  row_ind, col_ind = linear_sum_assignment(-iou_matrix)
  matched_indices = np.array(list(zip(row_ind, col_ind)))

  unmatched_detections = []
  for d,det in enumerate(detections):
    if(d not in matched_indices[:,0]):
      unmatched_detections.append(d)
  unmatched_trackers = []
  for t,trk in enumerate(trackers):
    if(t not in matched_indices[:,1]):
      unmatched_trackers.append(t)

  #filter out matched with low IOU
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

class Sort(object):
  def __init__(self,max_age=1,min_hits=3):
    """
    Sets key parameters for SORT
    """
    self.max_age = max_age
    self.min_hits = min_hits
    self.trackers = []
    self.frame_count = 0

  def update(self,dets):
    """
    Params:
      dets - a numpy array of detections in the format [[x,y,w,h,score],[x,y,w,h,score],...]
    Requires: this method must be called once for each frame even with empty detections.
    Returns the a similar array, where the last column is the object ID.
    NOTE: The number of objects returned may differ from the number of detections provided.
    """

    # prevent "too many indices for array" error
    if len(dets)==0:
      return np.empty((0,5))

    self.frame_count += 1
    #get predicted locations from existing trackers.
    trks = np.zeros((len(self.trackers),5))
    to_del = []
    ret = []
    for t,trk in enumerate(trks):
      pos = self.trackers[t].predict()[0]
      trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
      if(np.any(np.isnan(pos))):
        to_del.append(t)
    trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
    for t in reversed(to_del):
      self.trackers.pop(t)
    matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets,trks)

    #update matched trackers with assigned detections
    for t,trk in enumerate(self.trackers):
      if(t not in unmatched_trks):
        d = matched[np.where(matched[:,1]==t)[0],0]
        trk.update(dets[d,:][0])

    #create and initialise new trackers for unmatched detections
    for i in unmatched_dets:
        trk = KalmanBoxTracker(dets[i,:])
        self.trackers.append(trk)
    i = len(self.trackers)
    #for trk in reversed(self.trackers):
    for k,trk in enumerate(reversed(self.trackers)):
        d = trk.get_state()[0]
        if((trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits)):
          ret.append(np.concatenate((d,[trk.id+1],trk.cla)).reshape(1,-1)) # +1 as MOT benchmark requires positive
        i -= 1
        #remove dead tracklet
        if(trk.time_since_update > self.max_age):
          self.trackers.pop(i)
    if(len(ret)>0):
      return np.concatenate(ret)
    return np.empty((0,6))



def intersect(A,B,C,D):
    return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

def ccw(A,B,C):
    return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])



def focal_loss(hm_pred, hm_true, Ls_1):
    Ls_1 = tf.convert_to_tensor(Ls_1, tf.float32)
    pos_mask = tf.cast(tf.equal(hm_true, Ls_1), tf.float32)#0.8285714285714286
    neg_mask = tf.cast(tf.less(hm_true, Ls_1), tf.float32)
    neg_weights = tf.pow(1 - hm_true, 4)

    pos_loss = -tf.log(tf.clip_by_value(hm_pred, 1e-4, 1. - 1e-4)) * tf.pow(1 - hm_pred, 2) * pos_mask
    neg_loss = -tf.log(tf.clip_by_value(1 - hm_pred, 1e-4, 1. - 1e-4)) * tf.pow(hm_pred, 2) * neg_weights * neg_mask

    num_pos = tf.reduce_sum(pos_mask)
    pos_loss = tf.reduce_sum(pos_loss)
    neg_loss = tf.reduce_sum(neg_loss)

    cls_loss = tf.cond(tf.greater(num_pos, 0), lambda: (pos_loss + neg_loss) / num_pos, lambda: neg_loss)
    return cls_loss

def reg_l1_loss(y_pred, y_true, indices, mask):
    b = tf.shape(y_pred)[0]
    k = tf.shape(indices)[1]
    c = tf.shape(y_pred)[-1]
    y_pred = tf.reshape(y_pred, (b, -1, c))
    indices = tf.cast(indices, tf.int32)
    y_pred = tf.gather(y_pred, indices, batch_dims=1)
    mask = tf.tile(tf.expand_dims(mask, axis=-1), (1, 1, 2))
    total_loss = tf.reduce_sum(tf.abs(y_true * mask - y_pred * mask))
    reg_loss = total_loss / (tf.reduce_sum(mask) + 1e-4)
    return reg_loss

def centernet_loss(args):
    hm_pred, wh_pred, reg_pred, hm_true, wh_true, reg_true, reg_mask, indices, Ls_1 = args
    hm_loss = focal_loss(hm_pred, hm_true,Ls_1)
    wh_loss = 0.1 * reg_l1_loss(wh_pred, wh_true, indices, reg_mask)
    reg_loss = reg_l1_loss(reg_pred, reg_true, indices, reg_mask)
    total_loss = hm_loss + wh_loss + reg_loss
    return total_loss

class Mish(Layer):
    def call(self, inputs):
        return inputs * tf.math.tanh(tf.math.softplus(inputs))

def hourglass_module(bottom, cnv_dim, hgid, dims):
    lfs = left_features(bottom, hgid, dims)

    rf1 = right_features(lfs, hgid, dims)
    rf1 = convolution(rf1, 3, cnv_dim, name='cnvs.%d' % hgid)

    return rf1


def convolution(_x, k, out_dim, name, stride=1):
    padding = (k - 1) // 2
    _x = tf.keras.layers.ZeroPadding2D(padding=padding, name=name + '.pad')(_x)
    _x = tf.keras.layers.Conv2D(out_dim, k, strides=stride, use_bias=False, name=name + '.conv')(_x)
    _x = tf.keras.layers.BatchNormalization(epsilon=1e-5, name=name + '.bn')(_x)
    _x = Mish()(_x)
    return _x


def residual(_x, out_dim, name, stride=1):
    shortcut = _x
    num_channels = tf.keras.backend.int_shape(shortcut)[-1]
    _x = tf.keras.layers.ZeroPadding2D(padding=1, name=name + '.pad1')(_x)
    _x = tf.keras.layers.Conv2D(out_dim, 3, strides=stride, use_bias=False, name=name + '.conv1')(_x)
    _x = tf.keras.layers.BatchNormalization(epsilon=1e-5, name=name + '.bn1')(_x)
    _x = Mish()(_x)

    _x = tf.keras.layers.Conv2D(out_dim, 3, padding='same', use_bias=False, name=name + '.conv2')(_x)
    _x = tf.keras.layers.BatchNormalization(epsilon=1e-5, name=name + '.bn2')(_x)

    if num_channels != out_dim or stride != 1:
        shortcut = tf.keras.layers.Conv2D(out_dim, 1, strides=stride, use_bias=False, name=name + '.shortcut.0')(
                    shortcut)
        shortcut = tf.keras.layers.BatchNormalization(epsilon=1e-5, name=name + '.shortcut.1')(shortcut)

    _x = tf.keras.layers.Add(name=name + '.add')([_x, shortcut])
    _x = Mish()(_x)
    return _x


def pre(_x, num_channels):
    _x = convolution(_x, 3, 128, name='pre.0', stride=2)
    _x = residual(_x, num_channels, name='pre.1', stride=2)
    return _x


def left_features(bottom, hgid, dims):
    features = [bottom]
    for kk, nh in enumerate(dims):
        pow_str = ''
        for _ in range(kk):
            pow_str += '.center'
        _x = residual(features[-1], nh, name='kps.%d%s.down.0' % (hgid, pow_str), stride=2)
        _x = tf.keras.layers.Dropout(0.1)(_x)
        _x = residual(_x, nh, name='kps.%d%s.down.1' % (hgid, pow_str))
        features.append(_x)
    return features

def connect_left_right(left, right, num_channels, num_channels_next, name):
    left = residual(left, num_channels_next, name=name + 'skip.0')
    left = tf.keras.layers.Dropout(0.1)(left)
    left = residual(left, num_channels_next, name=name + 'skip.1')
    left = tf.keras.layers.Dropout(0.1)(left)

    out = residual(right, num_channels, name=name + 'out.0')
    out = tf.keras.layers.Dropout(0.1)(out)
    out = residual(out, num_channels_next, name=name + 'out.1')
    out = tf.keras.layers.Dropout(0.1)(out)
    out = tf.keras.layers.UpSampling2D(name=name + 'out.upsampleNN')(out)
    out = tf.keras.layers.Add(name=name + 'out.add')([left, out])
    return out


def bottleneck_layer(_x, num_channels, hgid):
    pow_str = 'center.' * 5
    _x = residual(_x, num_channels, name='kps.%d.%s0' % (hgid, pow_str))
    _x = tf.keras.layers.Dropout(0.1)(_x)
    _x = residual(_x, num_channels, name='kps.%d.%s1' % (hgid, pow_str))
    _x = tf.keras.layers.Dropout(0.1)(_x)
    _x = residual(_x, num_channels, name='kps.%d.%s2' % (hgid, pow_str))
    _x = tf.keras.layers.Dropout(0.1)(_x)
    _x = residual(_x, num_channels, name='kps.%d.%s3' % (hgid, pow_str))
    _x = tf.keras.layers.Dropout(0.1)(_x)
    _x = residual(_x, num_channels, name='kps.%d.%s4' % (hgid, pow_str))
    _x = tf.keras.layers.Dropout(0.1)(_x)
    return _x


def right_features(leftfeatures, hgid, dims):
    rf = bottleneck_layer(leftfeatures[-1], dims[-1], hgid)
    for kk in reversed(range(len(dims))):
        pow_str = ''
        for _ in range(kk):
            pow_str += 'center.'
        rf = connect_left_right(leftfeatures[kk], rf, dims[kk], dims[max(kk - 1, 0)], name='kps.%d.%s' % (hgid, pow_str))
    return rf

def create_heads(heads, rf1, hgid):
    _heads = []
    for head in sorted(heads):
        num_channels = heads[head]
        _x = tf.keras.layers.Conv2D(256, 3, use_bias=True, padding='same', name=head + '.%d.0.conv' % hgid)(rf1)
        _x = Mish()(_x)
        _x = tf.keras.layers.Conv2D(num_channels, 1, use_bias=True, name=head + '.%d.1' % hgid)(_x)
        _heads.append(_x)
    return _heads

def centernet(image_input, input_size=512, max_objects=140, num_classes = 7, score_threshold=0.1, nms=False, Ls_1=1):

    num_stacks = 2
    cnv_dim=96
    dims=[128, 128, 192, 256]

    output_size = input_size // 4

    hm_input = tf.keras.layers.Input(shape=(output_size,output_size,num_classes))
    wh_input = tf.keras.layers.Input(shape=(max_objects, 2))
    reg_input = tf.keras.layers.Input(shape=(max_objects, 2))
    reg_mask_input = tf.keras.layers.Input(shape=(max_objects,))
    index_input = tf.keras.layers.Input(shape=(max_objects,))

    inter = pre(image_input, cnv_dim)
    prev_inter = None
    for i in range(num_stacks):
        prev_inter = inter
        inter = hourglass_module(inter, cnv_dim, i, dims)
        if i < num_stacks - 1:
            inter_ = tf.keras.layers.Conv2D(cnv_dim, 1, use_bias=False, name='inter_.%d.0' % i)(prev_inter)
            inter_ = tf.keras.layers.BatchNormalization(epsilon=1e-5, name='inter_.%d.1' % i)(inter_)

            cnv_ = tf.keras.layers.Conv2D(cnv_dim, 1, use_bias=False, name='cnv_.%d.0' % i)(inter)
            cnv_ = tf.keras.layers.BatchNormalization(epsilon=1e-5, name='cnv_.%d.1' % i)(cnv_)

            inter = tf.keras.layers.Add(name='inters.%d.inters.add' % i)([inter_, cnv_])
            inter = Mish()(inter)
            inter = residual(inter, cnv_dim, 'inters.%d' % i)

    feature_maps = inter

    L2 = tf.keras.regularizers.l2(5e-4)

    y0 = tf.keras.layers.Conv2D(output_size, 3, padding='same', use_bias=False, kernel_initializer='he_normal', kernel_regularizer=L2)(feature_maps)
    y0 = tf.keras.layers.BatchNormalization()(y0)
    y0 = LeakyReLU(alpha=0.1)(y0)

    y1 = tf.keras.layers.Conv2D(num_classes, 1, kernel_initializer='he_normal', kernel_regularizer=L2, activation='sigmoid')(y0)

    y2 = tf.keras.layers.Conv2D(2, 1, kernel_initializer='he_normal', kernel_regularizer=L2)(y0)

    y3 = tf.keras.layers.Conv2D(2, 1, kernel_initializer='he_normal', kernel_regularizer=L2)(y0)

    cnt_loss = tf.keras.layers.Lambda(lambda x: centernet_loss(x + [Ls_1]), output_shape=(1,), name='centernet_loss')
    #([y1, y2, y3, hm_input, wh_input, reg_input, reg_mask_input, index_input])
    loss_ = cnt_loss([y1, y2, y3, hm_input, wh_input, reg_input, reg_mask_input, index_input])

    model = tf.keras.Model(inputs=[image_input, hm_input, wh_input, reg_input, reg_mask_input, index_input], outputs=[loss_])

    detections = tf.keras.layers.Lambda(lambda x: decode(*x, max_objects=max_objects, score_threshold=score_threshold, nms=nms, num_classes=num_classes),output_shape=(max_objects, 6))([y1, y2, y3])
    prediction_model = tf.keras.Model(inputs=image_input, outputs=detections)
    debug_model = tf.keras.Model(inputs=image_input, outputs=[y1, y2, y3])
    return model, prediction_model, debug_model

def evaluate_batch_item(batch_item_detections, num_classes, max_objects_per_class=140, max_objects=140,
                        iou_threshold=0.35, score_threshold=0.15):
    batch_item_detections = tf.boolean_mask(batch_item_detections,
                                            tf.greater(batch_item_detections[:, 4], score_threshold))
    detections_per_class = []
    for cls_id in range(num_classes):
        class_detections = tf.boolean_mask(batch_item_detections, tf.equal(batch_item_detections[:, 5], cls_id))
        nms_keep_indices = tf.image.non_max_suppression(class_detections[:, :4],
                                                        class_detections[:, 4],
                                                        max_objects_per_class,
                                                        iou_threshold=iou_threshold)
        class_detections = tf.gather(class_detections, nms_keep_indices)
        detections_per_class.append(class_detections)

    batch_item_detections = tf.concat(detections_per_class, axis=0)

    def filter():
        nonlocal batch_item_detections
        _, indices = tf.nn.top_k(batch_item_detections[:, 4], k=max_objects)
        batch_item_detections_ = tf.gather(batch_item_detections, indices)
        return batch_item_detections_

    def pad():
        nonlocal batch_item_detections
        batch_item_num_detections = tf.shape(batch_item_detections)[0]
        batch_item_num_pad = tf.maximum(max_objects - batch_item_num_detections, 0)
        batch_item_detections_ = tf.pad(tensor=batch_item_detections,
                                        paddings=[
                                            [0, batch_item_num_pad],
                                            [0, 0]],
                                        mode='CONSTANT',
                                        constant_values=0.0)
        return batch_item_detections_

    batch_item_detections = tf.cond(tf.shape(batch_item_detections)[0] >= 100,
                                    filter,
                                    pad)
    return batch_item_detections

def nms(heat, kernel=3):
    hmax = tf.nn.max_pool2d(heat, (kernel, kernel), strides=1, padding='SAME')
    heat = tf.where(tf.equal(hmax, heat), heat, tf.zeros_like(heat))
    return heat

def topk(hm, max_objects=140):
    hm = nms(hm)
    # (b, h * w * c)
    b, h, w, c = tf.shape(hm)[0], tf.shape(hm)[1], tf.shape(hm)[2], tf.shape(hm)[3]
    # hm2 = tf.transpose(hm, (0, 3, 1, 2))
    # hm2 = tf.reshape(hm2, (b, c, -1))
    hm = tf.reshape(hm, (b, -1))
    # (b, k), (b, k)
    scores, indices = tf.nn.top_k(hm, k=max_objects)
    # scores2, indices2 = tf.nn.top_k(hm2, k=max_objects)
    # scores2 = tf.reshape(scores2, (b, -1))
    # topk = tf.nn.top_k(scores2, k=max_objects)
    class_ids = indices % c
    class_ids = tf.where(tf.equal(class_ids, 2),class_ids - 1,class_ids)
    class_ids = tf.where(tf.equal(class_ids, 3),class_ids - 1,class_ids)
    class_ids = tf.where(tf.equal(class_ids, 4),class_ids - 2,class_ids)
    class_ids = tf.where(tf.equal(class_ids, 5),class_ids - 2,class_ids)
    class_ids = tf.where(tf.equal(class_ids, 6),class_ids - 2,class_ids)


    xs = indices // c % w
    ys = indices // c // w
    indices = ys * w + xs
    return scores, indices, class_ids, xs, ys

def decode(hm, wh, reg, max_objects=140, nms=True, num_classes=7, score_threshold=0.05):

    scores, indices, class_ids, xs, ys = topk(hm, max_objects=max_objects)
    b = tf.shape(hm)[0]
    # (b, h * w, 2)
    reg = tf.reshape(reg, (b, -1, tf.shape(reg)[-1]))
    # (b, h * w, 2)
    wh = tf.reshape(wh, (b, -1, tf.shape(wh)[-1]))
    # (b, k, 2)
    topk_reg = tf.gather(reg, indices, batch_dims=1)
    # (b, k, 2)
    topk_wh = tf.cast(tf.gather(wh, indices, batch_dims=1), tf.float32)
    topk_cx = tf.cast(tf.expand_dims(xs, axis=-1), tf.float32) + topk_reg[..., 0:1]
    topk_cy = tf.cast(tf.expand_dims(ys, axis=-1), tf.float32) + topk_reg[..., 1:2]
    scores = tf.expand_dims(scores, axis=-1)
    class_ids = tf.cast(tf.expand_dims(class_ids, axis=-1), tf.float32)
    topk_x1 = topk_cx - topk_wh[..., 0:1] / 2
    topk_x2 = topk_cx + topk_wh[..., 0:1] / 2
    topk_y1 = topk_cy - topk_wh[..., 1:2] / 2
    topk_y2 = topk_cy + topk_wh[..., 1:2] / 2
    # (b, k, 6)
    detections = tf.concat([topk_x1, topk_y1, topk_x2, topk_y2, scores, class_ids], axis=-1)
    if nms:
        detections = tf.map_fn(lambda x: evaluate_batch_item(x[0],
                                                             num_classes=num_classes,
                                                             score_threshold=score_threshold),
                               elems=[detections],
                               dtype=tf.float32)
    return detections

def create_model(h_, w_, num_classes):
    """create the training model"""
    K.clear_session()  # get a new session
    image_input = tf.keras.layers.Input(shape=(h_, w_, 3))

    model_body, prediction_model, debug_model = centernet(
        image_input,
        input_size=h_,
        max_objects=140,
        num_classes=num_classes,
        score_threshold=0.15,
        nms=True,
        Ls_1=Ls_1
    )

    return model_body, prediction_model, debug_model

###


train_model, prediction_model, debug_model = create_model(h_,w_, num_classes)
prediction_model.load_weights(weight_PATH)
print('load weights success')


tracker = Sort()
memory = {}



warnings.filterwarnings("ignore")

score_threshold = 0.3
nn = 0
frameIndex = 0

vs = cv2.VideoCapture(vid_path)
writer = None

fps = round(vs.get(cv2.CAP_PROP_FPS))
ph = fps * 1.8

det_list = []
passed = []

start = time.time()
# try to determine the total number of frames in the video file
try:
    prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
        else cv2.CAP_PROP_FRAME_COUNT
    total = int(vs.get(prop))
    print("{} total frames in video".format(total))
    print("{} fps".format(fps))
except:
    print("could not determine # of frames in video")
    print("no approx. completion time can be provided")
    total = -1

while True:
    (grabbed, frame) = vs.read()
    if not(grabbed):
        break

    #if frameIndex > 300:#fps*900:
    #    break

    if nn == 1:
        nn = 0
        continue
    else:

        image = frame
        o_h,o_w,o_c = np.shape(image)
        rate_w, rate_h = o_w / 512, o_h / 512
        image_ = np.copy(image)
        image = cv2.resize(image, (512, 512),interpolation=cv2.INTER_NEAREST)
        image_data = np.array(image, dtype='float32')

        detections = prediction_model.predict_on_batch(np.expand_dims(image_data / 255,axis=0))[0]

        scores = detections[:,4]
        indices = np.where(scores > score_threshold)[0]
        detections_ = detections[indices]
        detections_[:,0:4] = detections_[:,0:4] * 4

        out_boxes = detections_[:, 0:4]
        out_scores = np.expand_dims(detections_[:,4],axis=-1)
        out_classes = np.expand_dims(detections_[:,5],axis=-1)


        out_boxes[:,0] = out_boxes[:,0] * rate_w
        out_boxes[:,1] = out_boxes[:,1] * rate_h
        out_boxes[:,2] = out_boxes[:,2] * rate_w
        out_boxes[:,3] = out_boxes[:,3] * rate_h
        out_boxes = np.where(out_boxes < 0 , 0, out_boxes)


        dets = []
        if len(out_boxes) > 0:
            dets = np.concatenate((out_boxes,out_scores,out_classes),axis=-1)

        dets = np.asarray(dets)

        #print(dets)

        tracks = tracker.update(dets)


        boxes = []
        indexIDs = []
        c = []
        previous = memory.copy()
        memory = {}


        frame_d = {}


        for track in tracks:
            boxes.append([track[0], track[1], track[2], track[3], track[5]])
            indexIDs.append(int(track[4]))
            memory[indexIDs[-1]] = boxes[-1]


        total = 0
        dist_list = []

        if len(boxes) > 0:
            i = int(0)
            for box in boxes:
                # extract the bounding box coordinates
                (x, y) = (int(box[0]), int(box[1]))
                (w, h) = (int(box[2]), int(box[3]))
                cl = int(box[4])

                if cl == 0:
                    color = (255,0,0)
                elif cl == 1:
                    color = (0,255,255)
                elif cl == 2:
                    color = (0,0,255)
                elif cl == 3:
                    color = (255,255,255)
                elif cl == 4:
                    color = (0,0,0)


                cv2.rectangle(frame, (x, y), (w, h), color, 2)

                total += 1

                if indexIDs[i] in previous:
                    if indexIDs[i] in passed:
                        pass
                    else:
                        previous_box = previous[indexIDs[i]]
                        (x2, y2) = (int(previous_box[0]), int(previous_box[1]))
                        (w2, h2) = (int(previous_box[2]), int(previous_box[3]))
                        #if previous_box[4] == 1 or previous_box[4] == 2 or previous_box[4] == 3 or previous_box[4] == 4:
                        p0 = (int(x + (w-x)/2), int(y + (h-y)/4*3))
                        p1 = (int(x2 + (w2-x2)/2), int(y2 + (h2-y2)/4*3))

                        if speed_check == True:
                            pixel = MASK[p0[1], p0[0]]
                            if pixel > 128:

                                p0x = (f_[0][0]*p0[0] + f_[0][1]*p0[1] + f_[0][2]) / ((f_[2][0]*p0[0] + f_[2][1]*p0[1] + f_[2][2]))
                                p0y = (f_[1][0]*p0[0] + f_[1][1]*p0[1] + f_[1][2]) / ((f_[2][0]*p0[0] + f_[2][1]*p0[1] + f_[2][2]))
                                p0_a = (int(p0x), int(p0y)) # after transformation coordinate

                                p1x = (f_[0][0]*p1[0] + f_[0][1]*p1[1] + f_[0][2]) / ((f_[2][0]*p1[0] + f_[2][1]*p1[1] + f_[2][2]))
                                p1y = (f_[1][0]*p1[0] + f_[1][1]*p1[1] + f_[1][2]) / ((f_[2][0]*p1[0] + f_[2][1]*p1[1] + f_[2][2]))
                                p1_a = (int(p1x), int(p1y)) # after transformation coordinate

                                dist = math.hypot(p1_a[0] - p0_a[0], p1_a[1] - p0_a[1]) / 100

                                dist = int(dist * ph)
                                dist_list.append(dist)
                                cv2.putText(frame, str(dist), (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)

                                frame_d[indexIDs[i]] = (dist, cl)


                        cv2.line(frame, p0, p1, color, 6)

                        counter_ = 0
                        for ln in range(line_num):
                            if intersect(p0, p1, line_list[ln][0], line_list[ln][1]):
                                if cl == 0:
                                    counter_t[ln][0] += 1
                                if cl == 1:
                                    counter_t[ln][1] += 1
                                if cl == 2:
                                    counter_t[ln][2] += 1
                                if cl == 3:
                                    counter_t[ln][3] += 1
                                if cl == 4:
                                    counter_t[ln][4] += 1

                                counter_ += 1
                            if counter_ >= 1:
                                passed.append(indexIDs[i])

                            if len(passed) > 100:
                                passed.pop(0)
                text = "{}".format(indexIDs[i])
                cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                i += 1

    cv2.rectangle(frame, (5, 315), (30+45*class_num, 350+20*line_num), (255,255,255), -1)
    for lc in range(line_num):
        cv2.line(frame, line_list[lc][0], line_list[lc][1], line_color[lc], 2)
        for cc in range(class_num):
            cv2.putText(frame, str(counter_t[lc][cc]), (25 + 45*cc,350 + 20*lc), cv2.FONT_HERSHEY_DUPLEX, 0.5, line_color[lc], 1)


    det_list.append(frame_d)

    #save traffic
    os.path.join(tr_dir,'output',str(frameIndex))
    np.save(os.path.join(tr_dir,'output','traffic',str(frameIndex)) + '.npy',counter_t)

    #save speed
    if speed_check == True:
        speed_list = []
        for obj_id, (spd, cls_id) in frame_d.items():
            speed_list.append((obj_id, spd, cls_id))   # (ID, speed, class)
        speed_list = np.asarray(speed_list, dtype=np.float32)
        np.save(os.path.join(tr_dir,'output','speed',f'{frameIndex}.npy'), speed_list)


    if video_write == True:
        frame = cv2.resize(frame, dsize=(int(o_w/2), int(o_h/2)), interpolation=cv2.INTER_AREA)
        # check if the video writer is None
        if writer is None:
            # initialize our video writer
            fourcc = cv2.VideoWriter_fourcc(*"MP4V")
            writer = cv2.VideoWriter(os.path.join(tr_dir,'output','output_video' + '.mp4'), fourcc, round(fps/2),
                (frame.shape[1], frame.shape[0]), True)

            # write the output frame to disk
        writer.write(frame)

    # increase frame index2
    frameIndex += 2
    print(frameIndex)

    nn += 1


print("[INFO] cleaning up...")
if video_write == True:
    writer.release()

vs.release()

end = time.time()
print(end - start)
