from matplotlib import pyplot as plt
import numpy as np
import os
from mpl_toolkits.mplot3d import Axes3D


'''
Histogram Equalization
developed by Cong Zou, 10/8/2019
'''

class HEfunc(object):
    def __init__(self, image):
        #super(CCLfunc, self).__init__()
        self._img = np.copy(image)
        self.height = len(self._img)
        self.width = len(self._img[0])

        self.S_table = np.zeros(256)
    #simply get the average of the pixels, in the moon.bmp, this function is not used
    def Rgb2gray(self, img = []):
        if len(img) == 0:
            img = np.copy(self._img)
            
        for i in range(self.height):
            for j in range(self.width):
                if (img[i][j] - img[i][j][0]).any():
                    img[i][j] = sum(img[i][j])/3
        return img

    #build a table contains the pixel values and their corresponding frequency
    def Buildfreq(self, img = [], S_table = []):
        if len(img) == 0:
            img = np.copy(self._img)
        if len(S_table) == 0:
            S_table = self.S_table
            
        for i in range(self.height):
            for j in range(self.width):
                S_table[img[i][j][0]] += 1
        return S_table
        
    #project the original image to the preprocessed image
    #record the pixel value frequency of the processed image
    #get ready for lighting correction
    def Project(self, img = [], S_table = []):
        if len(img) == 0:
            img = np.copy(self._img)
        if len(S_table) == 0:
            S_table = self.S_table

        count = 0
        C_table = np.zeros(256)
        
        #project frequency to pixel value    
        for i in range(256):
            count += S_table[i]
            C_table[i] = int((count*255)/(self.height*self.width))
            
        #assign new value to image
        for i in range(self.height):
            for j in range(self.width):
                if C_table[img[i][j][0]] != img[i][j][0]:
                    img[i][j] = C_table[img[i][j][0]]
                    
        #rebuild S_table
        S_table = np.zeros(256)
        for i in range(self.height):
            for j in range(self.width):
                S_table[img[i][j][0]] += 1
                
        return img, S_table

    #get the plane image using a matrix
    def Plane_fitting(self, plane = [], quadratic = False):
        if len(plane) == 0:
            plane = np.copy(self._img)

        a = self.Get_param(plane, quadratic)
        #print(a)

        for i in range(self.height):
            for j in range(self.width):
                if quadratic:
                    if a[0]*(i**2) + a[1]*i*j + a[2]*(j**2) + a[3]*i + a[4]*j + a[5] > 255:
                        plane[i][j] = 255
                    elif a[0]*(i**2) + a[1]*i*j + a[2]*(j**2) + a[3]*i + a[4]*j + a[5] < 0:
                        plane[i][j] = 0
                    else:
                        plane[i][j] = a[0]*(i**2) + a[1]*i*j + a[2]*(j**2) + a[3]*i + a[4]*j + a[5]

                else:
                    if a[0]*i + a[1]*j +a[2] > 255:
                        plane[i][j] = 255
                    elif a[0]*i + a[1]*j +a[2] < 0:
                        plane[i][j] = 0
                    else:
                        plane[i][j] = a[0]*i + a[1]*j +a[2]
        return plane

    #get the a matrix, preparing for building plane image
    def Get_param(self, plane = [], quadratic = False):
        if len(plane) == 0:
            plane = np.copy(self._img)

        uvmatrix = self.Builduvmatrix(plane, quadratic)
        ymatrix = self.Buildymatrix(plane)
        Apseudo = np.linalg.pinv(uvmatrix)

        return np.dot(Apseudo, ymatrix)
        

    def Builduvmatrix(self, plane = [], quadratic = False):
        if len(plane) == 0:
            plane = np.copy(self._img)
        uvmatrix = []

        if quadratic:
            for u in range(len(plane)):
                for v in range(len(plane[0])):
                    uvmatrix.append([u**2, u*v, v**2, u, v, 1])
        else:
            for u in range(len(plane)):
                for v in range(len(plane[0])):
                    uvmatrix.append([u, v, 1])
        return np.matrix(uvmatrix)
    

    def Buildymatrix(self, plane = []):
        if len(plane) == 0:
            plane = np.copy(self._img)
        ymatrix = []

        for u in range(len(plane)):
            for v in range(len(plane[0])):
                ymatrix.append([plane[u][v][0]])
        return np.matrix(ymatrix)

    #lighting correction using plane image, including truncate and scale process
    def LightingC(self, img, plane, mode):
        maxnum = 0
        minnum = 0
        scaled = np.copy(img)
        truncated = np.copy(img)
        for i in range(self.height):
            for j in range(self.width):
                minnum = min((int(img[i][j][0]) - int(plane[i][j][0])), minnum)
                maxnum = max((int(img[i][j][0]) - int(plane[i][j][0])), maxnum)

        #scaled mode
        if mode == 1 or mode == 3:
            for i in range(self.height):
                for j in range(self.width):
                    scaled[i][j] = (int(img[i][j][0]) - int(plane[i][j][0]) - minnum)*(255/(maxnum-minnum))
        #truncated mode
        if mode == 2 or mode == 3:
            for i in range(self.height):
                for j in range(self.width):
                    if int(img[i][j][0]) - int(plane[i][j][0]) + (maxnum + minnum)/2 + 128< 0:
                        truncated[i][j] = 0
                    elif int(img[i][j][0]) - int(plane[i][j][0]) + (maxnum + minnum)/2 + 128 > 255:
                        truncated[i][j] = 255
                    else:
                        truncated[i][j] = int(img[i][j][0]) - int(plane[i][j][0]) + (maxnum + minnum)/2 + 128
                    
        return scaled, truncated


    def Draw_histogram(self, table = {}, img = [], mode = 1, res_dir = None):
        #draw original Histogram graph
        if mode == 1:
            x = np.arange(256)
            y = table
            plt.bar(x, y)
            plt.title('Frequency Distribution Before HE')
            plt.xlabel('Pixel Value')
            plt.ylabel('Frequency')
            imgpath = '\Original_Histogram.jpg'
            
        elif mode == 2:
            x = np.arange(256)
            y = table
            plt.bar(x, y)
            plt.title('Frequency Distribution After HE')
            plt.xlabel('Pixel Value')
            plt.ylabel('Frequency')
            imgpath = '\AfterHE_Histogram.jpg'

        elif mode == 3:
            x = np.arange(self.height)
            y = np.arange(self.width)
            fig = plt.figure()
            ax = Axes3D(fig)
            x, y = np.meshgrid(x, y)
            ax.plot_surface(x, y, img[...,0])
            imgpath = '\Linear_Plane.jpg'

        elif mode == 4:
            x = np.arange(self.height)
            y = np.arange(self.width)
            fig = plt.figure()
            ax = Axes3D(fig)
            x, y = np.meshgrid(x, y)
            ax.plot_surface(x, y, img[...,0])
            imgpath = '\Quadratic_Plane.jpg'

        else:
            raise Exception('Wrong draw histogram mode')

        if res_dir != None:
            res_dir = os.path.expanduser(res_dir)
            if os.path.exists(res_dir) == False:
                os.makedirs(res_dir)
            imgpath = res_dir + imgpath
            plt.savefig(imgpath)
        plt.show()
            

    #quick use of all algorithm above
    def Forward(self, img = [], S_table = [], fit = False, plane_mode = False, process_mode = 3, res_dir = None):
        if len(img) == 0:
            img = np.copy(self._img)
        if len(S_table) == 0:
            S_table = self.S_table
        plane = []
        scaled = []
        truncated = []

        S_table = self.Buildfreq(img, S_table)
        self.Draw_histogram(table = S_table, mode = 1, res_dir = res_dir)
        img, S_table = self.Project(img, S_table)
        self.Draw_histogram(table = S_table, mode = 2, res_dir = res_dir)
        if fit:
            if plane_mode:
                plane = self.Plane_fitting(np.copy(img), plane_mode)
                self.Draw_histogram(img = plane, mode = 4, res_dir = res_dir)

            else:
                plane = self.Plane_fitting(np.copy(img), plane_mode)
                self.Draw_histogram(img = plane, mode = 3, res_dir = res_dir)
                
            scaled, truncated = self.LightingC(np.copy(img), np.copy(plane), process_mode)
        
        return img, plane, scaled, truncated
        
'''
    def __call__(self):
        return forward()
'''




        










        





            
            
