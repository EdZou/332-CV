from matplotlib import pyplot as plt
import matplotlib.colors
import numpy as np
import os
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
import math
import cv2


'''
Color Segmentation
developed by Cong Zou, 10/16/2019
'''

class CSfunc(object):
    def __init__(self, pixels):
        self.pixels = pixels

        
    def Rgb2nrgb(self, pixels = []):
        if len(pixels) == 0:
            raise Exception('Rgb2nrgb pixel imreading failed')
        
        pdim = pixels.ndim
        if pdim != 2:
            height = len(pixels)
            width = len(pixels[0])
            pixels = np.reshape(pixels,(-1,3))
            
        if pixels.dtype != 'float':
            pixels = pixels.astype('float') / 255

        print('transforming data(RGB to NRGB)...')
        pbar = tqdm(total = len(pixels), desc = 'transforming pixels')
        
        for i in range(len(pixels)):
            total = sum(pixels[i])
            if total == 0:
                pixels[i] = total
            else:
                pixels[i] = [val/total for val in pixels[i]]
            pbar.update()
        pbar.close()
        
        if pdim != 2:
            pixels = np.reshape(pixels,(height, width, 3))
        
        return pixels
        

    def Rgb2hsi(self, pixels = []):
        #tranform [B,G,R] into [I,S,H]
        eqs = 0.0000000000000001
        if len(pixels) == 0:
            raise Exception('Rgb2nrgb pixel imreading failed')

        pdim = pixels.ndim
        if pdim != 2:
            height = len(pixels)
            width = len(pixels[0])
            pixels = np.reshape(pixels,(-1,3))
            
        if pixels.dtype != 'float':
            pixels = pixels.astype('float') / 255.0

        print('transforming data(RGB to HSI)...')
        pbar = tqdm(total = len(pixels), desc = 'transforming pixels')

        for i in range(len(pixels)):
            temp = np.copy(pixels[i])
            B = temp[0]
            G = temp[1]
            R = temp[2]
            if temp.any() == False:
                pixels[i] = 0
                pbar.update()
                continue
            comp = 0.5*(R - G + R - B) / (np.sqrt((R-G)**2 + (R-B)*(G-B)) + eqs)
            I = sum(temp)/3
            S = 1 - (3*min(temp)/sum(temp))
            H = np.arccos(comp)
            if B > G:
                H = 2*math.pi - H
            if np.sqrt((R-G)**2 + (R-B)*(G-B)) <= 0:
                H = 0
            pixels[i] = [I,S,H/(2*math.pi)]
            pbar.update()
        pbar.close()
        
        if pdim != 2:
            pixels = np.reshape(pixels,(height, width, 3))

        return pixels


    def Build_histogram(self, pixels = []):
        #build orginal R-G Histogram
        if len(pixels) == 0:
            raise Exception('The pixels imreading failed')
        F_table = {}
        total = len(pixels)

        print('training datasets...')
        pbar = tqdm(total = len(pixels), desc = 'building histogram')
        
        for i in range(total):
            #key is R-G, NR-NG or H-S pair
            key = str(int(pixels[i][2]*255)) + '_' + str(int(pixels[i][1]*255))
            if F_table.__contains__(key):
                F_table[key] += 1/total
            else:
                F_table[key] = 1/total
            pbar.update()
        pbar.close()
        return F_table


    def Hsi2rgb(self, pixels = []):
        #HSI should be stored in [I,S,H] format
        if len(pixels) == 0:
            raise Exception('Hsi2nrgb pixel imreading failed')
        
        pdim = pixels.ndim
        if pdim != 2:
            height = len(pixels)
            width = len(pixels[0])
            pixels = np.reshape(pixels,(-1,3))

        if pixels.dtype != 'float':
            raise Exception('Data type of HSI must be float')

        for i in range(len(pixels)):
            I = pixels[i][0]
            S = pixels[i][1]
            H = pixels[i][2]*(2*math.pi)
            angle = math.degrees(H)
            
            if angle < 120:
                if math.cos(math.pi/3 - H) != 0:
                    comp = I*(1 + S*math.cos(H)/math.cos(math.pi/3 - H))
                else:
                    comp = 0
                B = I*(1 - S)
                R = comp
                G = 3*I - R - B

            elif angle >= 120 and angle < 240:
                H -= math.pi*2/3
                if math.cos(math.pi/3 - H) != 0:
                    comp = I*(1 + S*math.cos(H)/math.cos(math.pi/3 - H))
                else:
                    comp = 0
                R = I*(1 - S)
                G = comp
                B = 3*I - R - G

            elif angle >= 240 and angle < 360:
                H -= math.pi*4/3
                if math.cos(math.pi/3 - H) != 0:
                    comp = I*(1 + S*math.cos(H)/math.cos(math.pi/3 - H))
                else:
                    comp = 0
                G = I*(1 - S)
                B = comp
                R = 3*I - B - G

            else:
                print(angle)
                raise Exception('Wrong angle in HSI')

            pixels[i] = [B,G,R]

        if pdim != 2:
            pixels = np.reshape(pixels,(height, width, 3))

        return pixels


    def Build_guassian(self, pixels = []):
        #pixels in [B,G,R] or [I,S,H], use R,G or H,S to calculate u,v
        if len(pixels) == 0:
            raise Exception('The Building Guassian pixels imreading failed')
        param = []
        ur = 0
        ug = 0
        vrr = 0
        vrg = 0
        vgr = 0
        vgg = 0
        total = len(pixels)
        print('start Gaussian Model training...')
        pbar = tqdm(total = len(pixels), desc = 'computing average...')

        for pixel in pixels:
            ur += pixel[2]/total
            ug += pixel[1]/total
            pbar.update()
        pbar.close()

        param.append([ur,ug])

        pbar = tqdm(total = len(pixels), desc = 'computing covariance...')
        for pixel in pixels:
            vrr += (pixel[2]-ur)**2/total
            vrg += (pixel[2]-ur)*(pixel[1]-ug)/total
            vgg += (pixel[1]-ug)**2/total
            pbar.update()
        pbar.close()
        vgr = vrg
        
        param.append([vrr,vrg])
        param.append([vgr,vgg])

        return np.array(param)


    #training function, mode 1 RGB 2 NRGB 3 HSI 
    def Train(self, pixels, mode = 1):
        print('start training...')
        if mode == 1:
            print('RGB color space')
            pixels = pixels.astype('float') / 255
            param = self.Build_histogram(pixels)

        elif mode == 2:
            print('NRGB color space')
            pixels = self.Rgb2nrgb(pixels) 
            param = self.Build_histogram(pixels)

        elif mode == 3:
            print('HSI color space')
            pixels = self.Rgb2hsi(pixels)
            param = self.Build_histogram(pixels)

        elif mode == 4:
            print('RGB color space')
            pixels = pixels.astype('float') / 255
            param = self.Build_guassian(pixels)

        elif mode == 5:
            print('NRGB color space')
            pixels = self.Rgb2nrgb(pixels) 
            param = self.Build_guassian(pixels)

        elif mode == 6:
            print('HSI color space')
            pixels = self.Rgb2hsi(pixels)
            param = self.Build_guassian(pixels)

        else:
            raise Exception('Wrong training mode')

        return param


    #test function for histogram
    def Histogram_test(self, img = [], F_table = [], threshold = 0.1, mode = 1):
        print('start Histogram testing...')
        if len(img) == 0:
            raise Exception('The image imreading of test failed')
        if len(F_table) == 0:
            raise Exception('The Histogram imreading of test failed')
        
        if mode == 1:
            timg = img.astype('float') / 255
        elif mode == 2:
            timg = self.Rgb2nrgb(np.copy(img))
        elif mode == 3:
            timg = self.Rgb2hsi(np.copy(img))
        else:
            raise Exception('Wrong mode in Histogram test function')

        for i in range(len(timg)):
            for j in range(len(timg[0])):
                temp = np.copy(timg[i][j])
                key = str(int(timg[i][j][2]*255)) + '_' + str(int(timg[i][j][1]*255))
                if F_table.__contains__(key) and F_table[key] > threshold:
                        continue 
                else:
                    timg[i][j] = 0
                    
        if mode == 2:
            for i in range(len(timg)):
                for j in range(len(timg[0])):
                    if timg[i][j].any() == False:
                        img[i][j] = 0
            timg = img.astype('float') / 255.0
        elif mode == 3:
            timg = self.Hsi2rgb(timg)

        return timg


    #test function for gaussian, 4 RGB, 5 NRGB, 6 HSI
    def Guassian_test(self, img = [], param = [], threshold = 3, mode = 4):
        print('start Gaussian Model testing...')
        if len(img) == 0:
            raise Exception('The image imreading of test failed')
        if len(param) == 0:
            raise Exception('The Guassian imreading of test failed')

        u = np.array(param[0])
        cov = np.array(param[1:3])
        s1s2 = np.sqrt(cov[0][0]*cov[1][1])#sigma1*sigma2
        p = cov[0][1] / s1s2#covariance param
        threshold = (1+p)*(threshold**2)*s1s2*((np.sqrt(cov[0][0])+np.sqrt(cov[1][1]))**2)

        if mode == 4:
            timg = img.astype('float') / 255
        elif mode == 5:
            timg = self.Rgb2nrgb(np.copy(img))
        elif mode == 6:
            timg = self.Rgb2hsi(np.copy(img))
        else:
            raise Exception('Wrong mode in Gaussian test function')

        for i in range(len(timg)):
            for j in range(len(timg[0])):
                temp = timg[i][j][1:3]
                judge = np.dot(np.dot(temp, cov), temp.T)
                if judge >= threshold:
                    timg[i][j] = 0

        if mode == 5:
            for i in range(len(timg)):
                for j in range(len(timg[0])):
                    if timg[i][j].any() == False:
                        img[i][j] = 0
            timg = img.astype('float') / 255.0
        elif mode == 6:
            timg = self.Hsi2rgb(timg)

        return timg
        

        
'''
    def __call__(self):
        return forward()
'''




        










        





            
            
