from matplotlib import pyplot as plt
import numpy as np

'''
Code for MP2
Including dilation, erosion, opening, closing, boundary, filter
developed by Cong Zou, 10/7/2019
'''

class MP2func(object):
    def __init__(self, image, k = np.ones([3,3])):
        #super(CCLfunc, self).__init__()
        self.img = image
        self.height = len(self.img)
        self.width = len(self.img[0])
        self.kernel = k
        self.k_h = len(k)
        self.k_w = len(k[0])

    #pad the image, can adjust the padding number in two directions and padding value
    def Padding(self, img = [], height = 1, width = 1, val = 0):
        if len(img) == 0:
            img = self.img
        if len(img[0][0]) != 3:
            raise Exception('The format of image is supposed to be RGB')
        
        #top and buttom padding
        htemp = np.zeros([height, self.width, 3])
        htemp += val
        img = np.append(htemp, img, axis = 0)
        img = np.append(img, htemp, axis = 0)

        #left and right padding
        wtemp = np.zeros([self.height + 2*height, width, 3])
        wtemp += val
        img = np.append(wtemp, img, axis = 1)
        img = np.append(img, wtemp, axis = 1)

        return img
        

    #dilation function
    def Dilation(self, img = [], k = []):
        if len(k) == 0:
            k = self.kernel
        if len(img) == 0:
            img = self.img
        k_h = int((len(k) - 1)/2)
        k_w = int((len(k[0]) - 1)/2)
        position = []
        img = self.Padding(img, k_h, k_w, 0)
        
        
        for i in range(k_h, self.height + k_h):
            for j in range(k_w, self.width + k_w):
                if img[i][j][0] == 0:
                    mark = self.Anycom(img[i-k_h:i+k_h+1, j-k_w:j+k_w+1], k)

                    if mark:
                        position.append([1, i, j])
        img = self.Visualize(img, position)
        return img[k_h: self.height + k_h, k_w: self.width + k_w, :]
            
    #check if there is any positive value in kernel area, core is [i,j]
    def Anycom(self, img, k):
        if len(img) != len(k) or len(img[0]) != len(k[0]):
            raise Exception('kernel size should be same as the input in anycom')
        for i in range(len(k)):
            for j in range(len(k[0])):
                if k[i][j]*img[i][j][0] > 0:
                    return True
        return False

    #erosion value
    def Erosion(self, img = [], k = []):
        if len(k) == 0:
            k = self.kernel
        if len(img) == 0:
            img = self.img
        k_h = int((len(k) - 1)/2)
        k_w = int((len(k[0]) - 1)/2)
        position = []
        img = self.Padding(img, k_h, k_w, 255)
        
        
        for i in range(k_h, self.height + k_h):
            for j in range(k_w, self.width + k_w):
                if img[i][j][0] > 0:
                    mark = self.Everycom(img[i-k_h:i+k_h+1, j-k_w:j+k_w+1], k)

                    if mark:
                        position.append([0, i, j])
        img = self.Visualize(img, position)
        return img[k_h: self.height + k_h, k_w: self.width + k_w, :]

    #check if there is any zero value in kernel area
    def Everycom(self, img, k):
        if len(img) != len(k) or len(img[0]) != len(k[0]):
            raise Exception('kernel size should be same as the input in anycom')
        for i in range(len(k)):
            for j in range(len(k[0])):
                if k[i][j]*img[i][j][0] == 0:
                    return True
        return False


    def Opening(self, img = [], k = []):
        if len(k) == 0:
            k = self.kernel
        if len(img) == 0:
            img = self.img

        img = self.Erosion(img, k)
        img = self.Dilation(img, k)
        return img


    def Closing(self, img = [], k = []):
        if len(k) == 0:
            k = self.kernel
        if len(img) == 0:
            img = self.img

        img = self.Dilation(img, k)
        img = self.Erosion(img, k)
        return img


    def Boundary(self, img = [], k = []):
        if len(k) == 0:
            k = self.kernel
        if len(img) == 0:
            img = self.img

        eimg = self.Erosion(img, k)
        return img - eimg

    def Filter(self, img = [], k = [], D = 2, E = 3):
        if len(k) == 0:
            k = self.kernel
        if len(img) == 0:
            img = self.img

        for i in range(D):
            img = self.Dilation(img, k)
        for j in range(E):
            img = self.Erosion(img, k)
        return img

        
    #visualize the image depending on the label value and position information
    def Visualize(self, img, position):
        if position[0][0] == 1:
            rgb = [255, 255, 255]
        elif position[0][0] == 0:
            rgb = [0, 0, 0]
        else:
            raise Exception('Wrong label was given in visualize')
        for item in position:
            img[item[1]][item[2]] = rgb
        return img
'''
    def __call__(self):
        return forward()
'''




        










        





            
            
