from matplotlib import pyplot as plt
import matplotlib.colors
import numpy as np
import os
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
import math
import cv2
import sys


'''
Canny Edge Detector
developed by Cong Zou, 11/01/2019
'''

class HoughTransform(object):
    def __init__(self, edge, degree, f_threshold, img, k_size):
        super(HoughTransform, self).__init__()
        if len(edge.shape) != 2:
            raise Exception('The shape of image input of Hough Transfrom should be [h,w,1]')
        self.edge = edge
        self.degree = degree
        self.deg = int(self.degree*180)
        self.p = int((np.sqrt(len(self.edge)**2 + len(self.edge[0])**2)*2)*self.degree)
        self.count = 0
        if f_threshold > 1:
            raise ValueError('The value of f_threshold should be between 0 to 1')
        self.f_threshold = f_threshold
        self.origin = img
        self.f_kernel = k_size
        
    def line_fitting(self):
        #depending on different degree, we have different precision
        #and we should build different ptheta matrix
        #we have 180 degs in theta and p depending on the size of edge image
        ptheta = np.zeros([self.deg, self.p])
        
        pbar = tqdm(total = len(self.edge)*len(self.edge[0]), desc = 'Transforming from x-y to p-theta...')
        for i in range(len(self.edge)):
            for j in range(len(self.edge[0])):
                if self.edge[i][j] > 0:
                    self.__xy2ptheta(ptheta, i, j)
                    self.count += 1
                pbar.update()
        pbar.close()
        return ptheta

    def __xy2ptheta(self, ptheta, x, y):
        for i in range(self.deg):
            tempd = np.deg2rad((i-self.deg/2)/self.degree)
            tempp = int((x*np.cos(tempd) + y*np.sin(tempd))*self.degree + self.p/2)
            ptheta[i][tempp] += 1
        return

    def __Padding(self, img = [], height = 1, width = 1, val = 0):
        if len(img) == 0:
            img = self.img
        if len(img.shape) != 2:
            raise Exception('The format of image is supposed to be grayscale')
        img_h = len(img)
        img_w = len(img[0])
        
        #top and buttom padding
        htemp = np.zeros([height, img_w])
        htemp += val
        img = np.append(htemp, img, axis = 0)
        img = np.append(img, htemp, axis = 0)

        #left and right padding
        wtemp = np.zeros([img_h + 2*height, width])
        wtemp += val
        img = np.append(wtemp, img, axis = 1)
        img = np.append(img, wtemp, axis = 1)

        return img

    def Secondfilter(self, ptheta, k_size = [51,51]):
        height = int(k_size[0]/2)
        width = int(k_size[1]/2)
        res = np.zeros([len(ptheta),len(ptheta[0]),3])
        ptheta = self.__Padding(ptheta, height, width)
        pbar = tqdm(total = len(res), desc = 'Second Filtering...')
        
        for i in range(len(res)):
            for j in range(len(res[i])):
                if ptheta[i + height][j + width] > self.count*self.f_threshold:
                    maxnum = 0
                    for k in range(2*height+1):
                        for l in range(2*width+1):
                            maxnum = max(ptheta[i+k][j+l],maxnum)
                    if maxnum == ptheta[i + height][j + width]:
                        res[i][j] = maxnum
            pbar.update()
        pbar.close()
        count = 0
        record = []
        for i in range(len(res)):
            for j in range(len(res[0])):
                if res[i][j][0] > 0:
                    count += 1
                    record.append([i,j,res[i][j][0]])
        
        return res, record, count

    def Line_detector(self, point):
        origin = np.copy(self.origin)
        return self.__ptheta2xy(point, origin)

    def __ptheta2xy(self, points, origin):
        #points has multiple coordinate and value of point
        #point consists of [deg, p, value(255)]
        pbar = tqdm(total = len(points)*len(origin[0]), desc = 'Detecting Lines...')
        for point in points:
            point[0] = (point[0] - self.deg/2)/self.degree
            point[1] = (point[1] - self.p/2)/self.degree
            if point[0] >= -45 and point[0] <= 45:
                for y in range(len(origin[0])):
                    x = int((point[1] - (y)*np.sin(np.deg2rad(point[0])))/np.cos(np.deg2rad(point[0])))
                    if x < len(origin) and x >= 0:
                        if origin[x][y] < 255:
                            origin[x][y] = 255
                    pbar.update()

            elif (point[0] >= -90 and point[0] < -45) or (point[0] > 45 and point[0] < 90):
                for x in range(len(origin)):
                    y = int((point[1] - (x)*np.cos(np.deg2rad(point[0])))/np.sin(np.deg2rad(point[0])))
                    if y < len(origin[0]) and y >= 0:
                        if origin[x][y] < 255:
                            origin[x][y] = 255
                    pbar.update()
                    
        pbar.close()
        return origin
                            
    def Forward(self):
        ptheta = self.line_fitting()
        s_ptheta, point, count = self.Secondfilter(np.copy(ptheta),[self.f_kernel*self.deg,self.f_kernel*self.p])
        print(self.count*self.f_threshold)
        print(count)
        print(point)
        res = self.Line_detector(point)
        return ptheta, s_ptheta, res

        
'''
    def __call__(self):
        return forward()
'''




        










        





            
            
