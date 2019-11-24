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
Face Detector
developed by Cong Zou, 11/10/2019
'''

class FDfunc(object):
    def __init__(self, img, delta, mode):
        super(FDfunc, self).__init__()
        self.img = img
        print(img.shape)
        self.n = 20
        self.gray = cv2.cvtColor(np.copy(self.img), cv2.COLOR_BGR2GRAY)
        #position (x,y,w,h) -- [y:y+h, x:x+w]
        self.position = self.Initialize()
        print(self.position)
        self.delta = delta
        self.dirs = self.__get_ellipse()
        self.mode = mode
        self.sample = self.img[self.position[1]:self.position[1]+self.position[3], self.position[0]:self.position[0]+self.position[2],:]
        self.s_histogram = self.__trans2color(np.copy(self.sample))
        temp, edge_sample = self.__image_gradient(np.copy(self.img))
        self.edge_sample = edge_sample[self.position[1]:self.position[1]+self.position[3], self.position[0]:self.position[0]+self.position[2]]
        

    def Initialize(self):
        face_cascade = cv2.CascadeClassifier('D:\Files\opencv\opencv\sources\data\haarcascades\haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(self.gray, 1.1, 5)
        if len(faces) != 1:
            raise ValueError('Only one face should be detected in intialization!')
        '''
        for (x,y,w,h) in faces:
            print((x,y,w,h))
            roigray = self.gray[y:y+h, x:x+w]
            img = cv2.rectangle(np.copy(self.img), (x,y), (x+w,y+h),(255,0,0))
        '''
        return faces[0]
    
    def __get_ellipse(self):
        angle = int(360/self.n)
        a = int(self.position[2]/2)
        b = int(self.position[3]/2)
        dirs = []
        for i in range(self.n):
            temp = i*angle
            x = int(a*np.cos(np.deg2rad(temp)))
            y = int(b*np.sin(np.deg2rad(temp)))
            dirs.append([y, x])
        return dirs
            

    def __Gpadding(self, img = [], k_size = [3, 3]):
        img_h = len(img)
        img_w = len(img[0])
        pad_h = int(k_size[0]/2)
        pad_w = int(k_size[1]/2)

        #padding [x,y,z] to [x+2*pad_h,y,z]
        temp = img[1: pad_h+1,...]
        img = np.append(temp[::-1,...], img, axis = 0)
        temp = img[-pad_h-1:-1,...]
        img = np.append(img,temp[::-1,...], axis = 0)

        #padding [x+2*pad_h,y,z] to [x+2*pad_h,y+2*pad_w,z]
        temp = img[:,1:pad_w+1]
        img = np.append(temp[:,::-1], img, axis = 1)
        temp = img[:,-pad_w-1:-1]
        img = np.append(img, temp[:,::-1], axis = 1)

        return img

    def __image_gradient(self, img = [], mode = 'sobel'):
        if len(img) == 0:
            raise Exception('Image input of image gradient cannot be null')
        if mode == 'sobel':
            xkernel = np.array([[-1,0,1],
                               [-2,0,2],
                               [-1,0,1]])
            ykernel = np.array([[1,2,1],
                                [0,0,0],
                                [-1,-2,-1]])

        pad_h = int(len(xkernel)/2)
        pad_w = int(len(xkernel[0])/2)
        img_h = len(img)
        img_w = len(img[0])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mag = np.zeros(img.shape)
        direct = np.zeros(img.shape)
        least = 0.0000000000000001

        img = self.__Gpadding(img, xkernel.shape)

        for i in range(img_h):
            for j in range(img_w):
                ii = i + pad_h
                jj = j + pad_w
                Gx = np.sum(xkernel*img[ii-pad_h:ii+pad_h+1, jj-pad_w:jj+pad_w+1])
                Gy = np.sum(ykernel*img[ii-pad_h:ii+pad_h+1, jj-pad_w:jj+pad_w+1])

                mag[i][j] = int(np.sqrt(Gx**2 + Gy**2))
                if Gx >= 0:
                    direct[i][j] = np.degrees(np.arctan(Gy/(Gx + least)))
                else:
                    direct[i][j] = np.degrees(np.arctan(Gy/(Gx + least))) + 180
        return mag, direct

    def __get_edgescore(self, mag, direct, x, y, dir_mode = True, sample = []):
        score = 0
        if not dir_mode:
            for dire in self.dirs:
                score += mag[y+dire[0]][x+dire[1]]
        else:
            for dire in self.dirs:
                sigma = np.cos(np.deg2rad(sample[dire[0]][dire[1]] - direct[y+dire[0]][x+dire[1]]))
                score += sigma*mag[y+dire[0]][x+dire[1]]
        return score/self.n

    def __trans2color(self, img):
        #input BGR, weight 
        histogram = np.zeros((3,256*3)).astype('float')
        for i in range(len(img)):
            for j in range(len(img[0])):
                if (i - int(self.position[3]/2))**2 + (j - int(self.position[2]/2))**2 <= (int(self.position[3]/2))**2 + (int(self.position[2]/2))**2:
                    histogram[0][int(img[i][j][0])-int(img[i][j][1]) + 255] += 1
                    histogram[1][int(img[i][j][0])-int(img[i][j][2]) + 255] += 1
                    histogram[2][np.sum(img[i][j])] += 1
        return histogram

    def __get_colorscore(self, img, s_histogram):
        i_histogram = self.__trans2color(img)
        res = 0
        for i in range(len(i_histogram[0])):
            temp = 0
            temp += 2*min(i_histogram[0][i], s_histogram[0][i])
            temp += 2*min(i_histogram[1][i], s_histogram[1][i])
            temp += min(i_histogram[2][i], s_histogram[2][i])
            temp = float(temp)/(2*float(np.sum(i_histogram[...,i])) - float(i_histogram[2][i]) + 0.000000000000001)
            res += temp
        return res

    def Get_score(self, img = []):
        if len(img) == 0:
            raise ValueError('Image cannot be None')
        center = [0, 0]
        if self.mode == 1:
            #execute SSD mode
            minnum = 99999999999999999999999
            res = []
            for i in range(0, len(img)-self.position[3], self.delta):
                for j in range(0, len(img[0])-self.position[2], self.delta):
                    temp = np.sum(np.square(self.sample - img[i:i+self.position[3], j:j+self.position[2],:]))
                    if temp < minnum:
                        minnum = temp
                        center = (j, i)
        elif self.mode == 2:
            #execute CC mode
            maxnum = 0
            for i in range(0, len(img)-self.position[3], self.delta):
                for j in range(0, len(img[0])-self.position[2], self.delta):
                    temp = np.sum(self.sample*img[i:i+self.position[3], j:j+self.position[2],:])
                    if temp > maxnum:
                        maxnum = temp
                        center = (j, i)
        elif self.mode == 3:
            #execute NCC mode
            sample = np.copy(self.sample)
            bavgi = np.sum(sample[...,0])/(self.position[2]*self.position[3])
            gavgi = np.sum(sample[...,1])/(self.position[2]*self.position[3])
            ravgi = np.sum(sample[...,2])/(self.position[2]*self.position[3])
            sample = sample - [bavgi, gavgi, ravgi]
            timg = img
            maxnum = 0
            for i in range(0, len(img)-self.position[3], self.delta):
                for j in range(0, len(img[0])-self.position[2], self.delta):
                    bavgt = np.sum(timg[i:i+self.position[3], j:j+self.position[2],0])/(self.position[2]*self.position[3])
                    gavgt = np.sum(timg[i:i+self.position[3], j:j+self.position[2],1])/(self.position[2]*self.position[3])
                    ravgt = np.sum(timg[i:i+self.position[3], j:j+self.position[2],2])/(self.position[2]*self.position[3])
                    t = timg[i:i+self.position[3], j:j+self.position[2]] - [bavgt, gavgt, ravgt]
                    temp = np.sum(np.absolute(sample)*np.absolute(t))/np.sqrt(np.sum(np.square(sample))*np.sum(np.square(t)))
                    if temp > maxnum:
                        maxnum = temp
                        center = (j, i)
        elif self.mode == 4:
            #cvpr98 without unit vector
            minedge = 999999999999999
            maxedge = -1
            mincolor = 999999999999999
            maxcolor = -1
            #temp, edge_sample = self.__image_gradient(np.copy(self.img))
            #edge_sample = edge_sample[self.position[1]:self.position[1]+self.position[3], self.position[0]:self.position[0]+self.position[2],:].astype('float')
            matrix = np.zeros((int((len(img)-self.position[3])/self.delta), int((len(img[0])-self.position[2])/self.delta), 2))
            mag, direct = self.__image_gradient(np.copy(img))
            score = 0
            
            for i in range(int(self.position[3]/2), int(len(img)-self.position[3]/2), self.delta):
                for j in range(int(self.position[2]/2), int(len(img[0])-self.position[2]/2), self.delta):
                    edge = self.__get_edgescore(mag, direct, j, i, False, sample = [])
                    if edge > maxedge:
                        maxedge = edge
                    if edge < minedge:
                        minedge = edge
                    color = self.__get_colorscore(img[i-int(self.position[3]/2): i+int(self.position[3]/2)][j-int(self.position[2]/2):j+int(self.position[2]/2),:], self.s_histogram)
                    if color > maxcolor:
                        maxcolor = color
                    if color < mincolor:
                        mincolor = color
                    matrix[i-int(self.position[3]/2)][j-int(self.position[2]/2)] = [edge, color]

            for i in range(len(matrix)):
                for j in range(len(matrix[0])):
                    temp = (matrix[i][j][0]-minedge)/(maxedge-minedge) + (matrix[i][j][1]-mincolor)/(maxcolor-mincolor)
                    if temp > score:
                        score = temp
                        center = (j*self.delta, i*self.delta)
        elif self.mode == 5:
            #cvpr98 without unit vector
            minedge = 999999999999999
            maxedge = -1
            mincolor = 999999999999999
            maxcolor = -1
            #temp, edge_sample = self.__image_gradient(np.copy(self.img))
            #edge_sample = edge_sample[self.position[1]:self.position[1]+self.position[3], self.position[0]:self.position[0]+self.position[2],:].astype('float')
            matrix = np.zeros((int((len(img)-self.position[3])/self.delta), int((len(img[0])-self.position[2])/self.delta), 2))
            mag, direct = self.__image_gradient(np.copy(img))
            score = 0
            
            for i in range(int(self.position[3]/2), int(len(img)-self.position[3]/2), self.delta):
                for j in range(int(self.position[2]/2), int(len(img[0])-self.position[2]/2), self.delta):
                    edge = self.__get_edgescore(mag, direct, j, i, True, self.edge_sample)
                    if edge > maxedge:
                        maxedge = edge
                    if edge < minedge:
                        minedge = edge
                    color = self.__get_colorscore(img[i-int(self.position[3]/2): i+int(self.position[3]/2)][j-int(self.position[2]/2):j+int(self.position[2]/2),:], self.s_histogram)
                    if color > maxcolor:
                        maxcolor = color
                    if color < mincolor:
                        mincolor = color
                    matrix[i-int(self.position[3]/2)][j-int(self.position[2]/2)] = [edge, color]

            for i in range(len(matrix)):
                for j in range(len(matrix[0])):
                    temp = (matrix[i][j][0]-minedge)/(maxedge-minedge) + (matrix[i][j][1]-mincolor)/(maxcolor-mincolor)
                    if temp > score:
                        score = temp
                        center = (j*self.delta, i*self.delta)
                    
                        
        res = cv2.rectangle(np.copy(img), center, (int(center[0])+self.position[3], int(center[1])+self.position[2]), (255,0,0))
        return res
            
        
        


    def Forward(self, img):
        res = self.Get_score(img)
        return res

        
        


        
'''
    def __call__(self):
        return forward()
'''




        










        





            
            
