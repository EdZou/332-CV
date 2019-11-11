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
developed by Cong Zou, 10/24/2019
'''

class CEDfunc(object):
    def __init__(self, img, gmode, elmode, threshold):
        super(CEDfunc, self).__init__()
        self.img = img
        self.gmode = gmode
        self.elmode = elmode
        self.threshold = threshold
        sys.setrecursionlimit(1000000)

    def Gaussian_smooth(self, img = [], k_size = [3, 3], sigma = 3):
        if len(img) == 0:
            raise Exception('Image input of Gaussian_Smooth cannot be null')
        if k_size[0]%2 == 0 or k_size[1]%2 == 0:
            print('kernel size is supposed to be odd number')
        
        img_h = len(img)
        img_w = len(img[0])
        pad_h = int(k_size[0]/2)
        pad_w = int(k_size[1]/2)
        
        img = self.__Gpadding(img, k_size)
        temp = np.zeros(img.shape)
        kernel = self.__generate_Gkernel(k_size, sigma)

        pbar = tqdm(total = img_h*img_w, desc = 'Gaussian Smoothing...')

        for i in range(img_h):
            for j in range(img_w):
                ii = i + pad_h
                jj = j + pad_w
                temp[ii][jj][0] = int(sum(np.reshape(kernel*img[ii-pad_h:ii+pad_h+1, jj-pad_w:jj+pad_w+1, 0],(-1))))
                temp[ii][jj][1] = int(sum(np.reshape(kernel*img[ii-pad_h:ii+pad_h+1, jj-pad_w:jj+pad_w+1, 1],(-1))))
                temp[ii][jj][2] = int(sum(np.reshape(kernel*img[ii-pad_h:ii+pad_h+1, jj-pad_w:jj+pad_w+1, 2],(-1))))
                pbar.update()
        pbar.close()

        return temp[pad_h:-pad_h, pad_w:-pad_w, :]
        


    def __generate_Gkernel(self, k_size = [3, 3], sigma = 3):
        k_h = k_size[0]
        k_w = k_size[1]
        center = [int(k_h/2), int(k_w/2)]
        kernel = np.zeros(k_size)
        total = 0

        #first time get 2-D gaussian matrix
        for i in range(k_h):
            for j in range(k_w):
                distance = (i-center[0])**2 + (j-center[1])**2
                kernel[i][j] = (np.exp(-distance/(2*(sigma)**2))/(2*np.pi*(sigma)**2))
                total += kernel[i][j]
        #normalization
        kernel = kernel/total
        return kernel

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
        temp = img[:,1:pad_w+1,:]
        img = np.append(temp[:,::-1,:], img, axis = 1)
        temp = img[:,-pad_w-1:-1,:]
        img = np.append(img, temp[:,::-1,:], axis = 1)

        return img


    def __Rgb2gray(self, img = []):
        for i in range(len(img)):
            for j in range(len(img[0])):
                img[i][j] = sum(img[i][j])/3
        return img


    def Image_gradient(self, img = [], mode = 'sobel'):
        if len(img) == 0:
            raise Exception('Image input of Gaussian_Smooth cannot be null')
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
        img = self.__Rgb2gray(img)
        mag = np.zeros(img.shape)
        direct = np.zeros(img.shape)
        least = 0.0000000000000001

        img = self.__Gpadding(img, xkernel.shape)

        pbar = tqdm(total = img_h*img_w, desc = 'Calculating Image Gradient...')

        for i in range(img_h):
            for j in range(img_w):
                ii = i + pad_h
                jj = j + pad_w
                Gx = sum(np.reshape(xkernel*img[ii-pad_h:ii+pad_h+1, jj-pad_w:jj+pad_w+1, 0],(-1)))
                Gy = sum(np.reshape(ykernel*img[ii-pad_h:ii+pad_h+1, jj-pad_w:jj+pad_w+1, 0],(-1)))

                mag[i][j] = int(np.sqrt(Gx**2 + Gy**2))
                if Gx >= 0:
                    direct[i][j] = np.degrees(np.arctan(Gy/(Gx + least)))
                else:
                    direct[i][j] = np.degrees(np.arctan(Gy/(Gx + least))) + 180
                pbar.update()
        pbar.close()
        return mag, direct


    #Find threshold for Edge detecting, return High threshold and low, can be equaled
    def Find_threshold(self, mag = [], percentage = 0.8):
        if len(mag) == 0:
            raise Exception('Magnitude input of Find_threshold cannot be null')
        mag_x = len(mag)
        mag_y = len(mag[0])
        maxnum = 0
        h_threshold = 0

        for i in range(mag_x):
            for j in range(mag_y):
                if maxnum < mag[i][j][0]:
                    maxnum = mag[i][j][0]
        maxnum = int(maxnum)
        
        histogram = np.zeros(maxnum)
        total = mag_x*mag_y
        h_sum = 0

        #Build Histogram
        for i in range(mag_x):
            for j in range(mag_y):
                histogram[int(mag[i][j][0])-1] += 1/total
        #Find threhold
        for i in range(maxnum):
            if h_sum >= percentage:
                h_threshold = i
                break
            h_sum += histogram[i]

        return [h_threshold, int(h_threshold/2)]


    #suppress nonmaxima contains LUT mode and Interpolation mode
    def Suppress_Nonmaxima(self, mag = [], direct = [], mode = 'lut'):
        mag = self.__Gpadding(mag, [3, 3])
        res = np.zeros(direct.shape).astype('float')
        res_h = len(res)
        res_w = len(res[0])
        
        
        if mode == 'lut':
            pbar = tqdm(total = res_h*res_w, desc = 'Non-maxima Suppression(LUT)...')
            for i in range(res_h):
                for j in range(res_w):
                    if direct[i][j][0] >= 22.5 and direct[i][j][0] < 67.5:
                        p1 = [-1, 1]
                    elif direct[i][j][0] >= 67.5 and direct[i][j][0] < 112.5:
                        p1 = [-1, 0]
                    elif direct[i][j][0] >= 112.5 and direct[i][j][0] < 157.5:
                        p1 = [-1, -1]
                    elif direct[i][j][0] >= 157.5 and direct[i][j][0] < 202.5:
                        p1 = [0, -1]
                    elif direct[i][j][0] >= 202.5 and direct[i][j][0] < 247.5:
                        p1 = [1, -1]
                    elif direct[i][j][0] >= 247.5 and direct[i][j][0] <= 270:
                        p1 = [1, 0]
                    elif direct[i][j][0] >= -90.0 and direct[i][j][0] < -67.5:
                        p1 = [1, 0]
                    elif direct[i][j][0] >= -67.5 and direct[i][j][0] < -22.5:
                        p1 = [1, 1]
                    elif direct[i][j][0] >= -22.5 and direct[i][j][0] < 22.5:
                        p1 = [0, 1]
                    else:
                        print(direct[i][j])
                        raise Exception('Wrong direction value under LUT mode!')

                    if mag[i][j][0] >= mag[i + p1[0]][j + p1[1]][0] and mag[i][j][0] >= mag[i - p1[0]][j - p1[1]][0]:
                        res[i][j] = mag[i][j][0]
                    pbar.update()

        elif mode == 'interpolation':
            pbar = tqdm(total = res_h*res_w, desc = 'Non-maxima Suppression(LUT)...')
            for i in range(res_h):
                for j in range(res_w):
                    if direct[i][j][0] >= 0 and direct[i][j][0] < 45:
                        p1 = [0, 1]
                        p2 = [-1, 1]
                        alpha = 1 - abs(np.tan(np.deg2rad(direct[i][j][0])))
                    elif direct[i][j][0] >= 45 and direct[i][j][0] < 90:
                        p1 = [-1, 1]
                        p2 = [-1, 0]
                        alpha = 1/abs(np.tan(np.deg2rad(direct[i][j][0])))
                    elif direct[i][j][0] >= 90 and direct[i][j][0] < 135:
                        p1 = [-1, 0]
                        p2 = [-1, 1]
                        alpha = 1 - 1/abs(np.tan(np.deg2rad(direct[i][j][0])))
                    elif direct[i][j][0] >= 135 and direct[i][j][0] < 180:
                        p1 = [-1, 1]
                        p2 = [0, -1]
                        alpha = abs(np.tan(np.deg2rad(direct[i][j][0])))
                    elif direct[i][j][0] >= 180 and direct[i][j][0] < 225:
                        p1 = [0, -1]
                        p2 = [1, -1]
                        alpha = 1 - abs(np.tan(np.deg2rad(direct[i][j][0])))
                    elif direct[i][j][0] >= 225 and direct[i][j][0] <= 270:
                        p1 = [1, -1]
                        p2 = [1, 0]
                        alpha = 1/abs(np.tan(np.deg2rad(direct[i][j][0])))
                    elif direct[i][j][0] >= -90 and direct[i][j][0] < -45:
                        p1 = [1, 0]
                        p2 = [1, 1]
                        alpha = 1 - 1/abs(np.tan(np.deg2rad(direct[i][j][0])))
                    elif direct[i][j][0] >= -45 and direct[i][j][0] < 0:
                        p1 = [1, 1]
                        p2 = [0, 1]
                        alpha = abs(np.tan(np.deg2rad(direct[i][j][0])))
                    else:
                        raise Exception('Wrong direction value under Interpolation mode!')

                    if alpha*mag[i + p1[0]][j + p1[1]][0] + (1 - alpha)*mag[i + p2[0]][j + p2[1]][0] <= mag[i][j][0] \
                       and alpha*mag[i - p1[0]][j - p1[1]][0] + (1 - alpha)*mag[i - p2[0]][j - p2[1]][0] <= mag[i][j][0]:
                        res[i][j] = mag[i][j][0]

                    #res[i][j] = alpha*mag[i + p1[0]][j + p1[1]][0] + (1 - alpha)*mag[i + p2[0]][j + p2[1]][0]
                    pbar.update()

        else:
            raise Exception('Wrong Non-Maxima Suppression mode')
        pbar.close()

        return res


    #Filter the Edge, threshold is supposed to be [high, low]
    def Filter(self, threshold = [], mag = []):
        if len(mag) == 0:
            raise Exception('Magnitude input of Filter cannot be null')
        if len(threshold) == 0:
            raise Exception('Threshold input of Filter cannot be null')

        low_res = np.zeros(mag.shape)
        high_res = np.zeros(mag.shape)

        for i in range(len(mag)):
            for j in range(len(mag[0])):
                if mag[i][j][0] > threshold[1]:
                    low_res[i][j] = 255
                    if mag[i][j][0] > threshold[0]:
                        high_res[i][j] = 255

        return low_res, high_res


    #pad the image, can adjust the padding number in two directions and padding value
    def __Padding(self, img = [], height = 1, width = 1, val = 0):
        if len(img) == 0:
            img = self.img
        if len(img[0][0]) != 3:
            raise Exception('The format of image is supposed to be RGB')
        img_h = len(img)
        img_w = len(img[0])
        
        #top and buttom padding
        htemp = np.zeros([height, img_w, 3])
        htemp += val
        img = np.append(htemp, img, axis = 0)
        img = np.append(img, htemp, axis = 0)

        #left and right padding
        wtemp = np.zeros([img_h + 2*height, width, 3])
        wtemp += val
        img = np.append(wtemp, img, axis = 1)
        img = np.append(img, wtemp, axis = 1)

        return img


    def Edge_link(self, low_res = [], high_res = [], k_size = [3, 3], mode = 'hysteresis'):
        if len(low_res) == 0:
            raise Exception('Low threshold result input of Edge_link cannot be null')
        if len(high_res) == 0:
            raise Exception('High threshold result input of Edge_link cannot be null')
        res_h = len(high_res)
        res_w = len(high_res[0])
        kernel = np.ones(k_size)
        pad_h = int(k_size[0]/2)
        pad_w = int(k_size[1]/2)
        kernel[pad_h][pad_w] = 0

        if mode == 'recursive':
            final = self.__Padding(high_res, pad_w, pad_h, 0)
            for i in range(res_h):
                for j in range(res_w):
                    self.__Edge_helper(res_h, res_w, pad_h, pad_w, kernel, low_res, high_res, final, i, j)
                    
        elif mode == 'hysteresis':
            high_res = self.__Padding(high_res, pad_w, pad_h, 0)
            final = self.__Hys_helper(np.copy(low_res), high_res, kernel, pad_h, pad_w)
        return final[pad_h:-pad_h, pad_w:-pad_w, :]


    def __Edge_helper(self, res_h, res_w, pad_h, pad_w, kernel, low_res, high_res, final, i, j):
        if i < 0 or i >= res_h or j < 0 or j >= res_w:
            return
        if low_res[i][j][0] > 0 and final[i + pad_h][j + pad_w][0] == 0:
            if (kernel*final[i:i+2*pad_w+1, j:j+2*pad_h+1, 0]).any():
                final[i+pad_w][j+pad_h] = 255
                if i < res_h - 1:
                    self.__Edge_helper(res_h, res_w, pad_h, pad_w, kernel, low_res, high_res, final, i+1, j)
                if j < res_w - 1:
                    self.__Edge_helper(res_h, res_w, pad_h, pad_w, kernel, low_res, high_res, final, i, j+1)
                if i < res_h - 1 and j < res_w - 1:
                    self.__Edge_helper(res_h, res_w, pad_h, pad_w, kernel, low_res, high_res, final, i+1, j+1)
                if i > 0:
                    self.__Edge_helper(res_h, res_w, pad_h, pad_w, kernel, low_res, high_res, final, i-1, j)
                if j > 0:
                    self.__Edge_helper(res_h, res_w, pad_h, pad_w, kernel, low_res, high_res, final, i, j-1)
                if i > 0 and j > 0:
                    self.__Edge_helper(res_h, res_w, pad_h, pad_w, kernel, low_res, high_res, final, i-1, j-1)
                if i < res_h - 1 and j > 0:
                    self.__Edge_helper(res_h, res_w, pad_h, pad_w, kernel, low_res, high_res, final, i+1, j-1)
                if i > 0 and j < res_w - 1:
                    self.__Edge_helper(res_h, res_w, pad_h, pad_w, kernel, low_res, high_res, final, i+1, j-1)
        return


    def __Hys_helper(self, low_res, high_res, kernel, pad_h, pad_w):
        for i in range(len(low_res)):
            for j in range(len(low_res[0])):
                if low_res[i][j][0] > 0:
                    if (kernel*high_res[i:i+2*pad_w+1, j:j+2*pad_h+1, 0]).any() == False:
                        low_res[i][j]  = 0
        return low_res
        


    def Forward(self, img = [], kernel = [3, 3], sigma = 3):
        img = self.Gaussian_smooth(img, kernel, sigma)
        mag, direct = self.Image_gradient(np.copy(img))
        threshold = self.Find_threshold(mag, self.threshold)
        NMS_mag = self.Suppress_Nonmaxima(np.copy(mag), direct, self.gmode)
        low_res, high_res = self.Filter(threshold, NMS_mag)
        final = self.Edge_link(low_res, np.copy(high_res), [3,3], self.elmode)
        
        return img, mag, NMS_mag, low_res, high_res, final

        
        


        
'''
    def __call__(self):
        return forward()
'''




        










        





            
            
