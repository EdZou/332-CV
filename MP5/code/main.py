import cv2
import numpy as np
from matplotlib import pyplot as plt
from CannyEdgeDetector import CEDfunc
import argparse
import os
import pickle

'''
main function of Canny Edge Detetor algorithm
developed by Cong Zou, 10/26/2019

To use it, open the terminal in Linux or cmd in windows
enter the directory of main.py and CannyEdgeDector.py
CAUTION: These two .py files should be in the same file folder
imread the img by changing the root
For instance, enter this order in terminal/cmd
python main.py --img.dir D:/files/image/moon.bmp
to find the image.

'''


parser = argparse.ArgumentParser()
parser.add_argument('--img_dir', type = str, default = 'lena.bmp')
parser.add_argument('--res_dir', type = str, default = None)
parser.add_argument('--kernel_height', type = int, default = 3)
parser.add_argument('--kernel_width', type = int, default = 3)
parser.add_argument('--sigma',type = float, default = 3.0)
parser.add_argument('--gradient_mode', type = str, choices = ['lut', 'interpolation'], default = 'interpolation')
parser.add_argument('--edge_mode', type = str, choices = ['recursive', 'hysteresis'], default = 'recursive')
parser.add_argument('--threshold', type = float, default = 0.8)

def main(args):
    #===========================Test=============================
    img_dir = args.img_dir
    args.img_dir = os.path.expanduser(args.img_dir)
    img = cv2.imread(args.img_dir)
    if img.any() == False:
        raise Exception('The image imreading failed')

    k_size = [args.kernel_height, args.kernel_width]

    plt.imshow(img[...,[2,1,0]])
    plt.title('Original Image')
    plt.show()

    csd = CEDfunc(img,args.gradient_mode, args.edge_mode, args.threshold)
    res, mag, NMS_mag, low_res, high_res, final = csd.Forward(img, k_size, args.sigma)
    
    plt.imshow(res[...,[2,1,0]].astype('uint8'))
    plt.title('Gaussian Processed Image')
    plt.show()

    plt.imshow(mag[...,[2,1,0]].astype('uint8'))
    plt.title('Image Gradient Image')
    plt.show()

    plt.imshow(NMS_mag[...,[2,1,0]].astype('uint8'))
    plt.title('NMS Magnitude Image')
    plt.show()

    plt.imshow(low_res[...,[2,1,0]].astype('uint8'))
    plt.title('Low Threshold Image')
    plt.show()

    plt.imshow(high_res[...,[2,1,0]].astype('uint8'))
    plt.title('High Threshold Image')
    plt.show()
    
    plt.imshow(final[...,[2,1,0]].astype('uint8'))
    plt.title('Final Image')
    plt.show()
    

    #=====================save images results=============================
    if args.res_dir != None:
        args.res_dir = os.path.expanduser(args.res_dir)
            
        if os.path.exists(args.res_dir) == False:
            os.makedirs(args.res_dir)

        mode = ['\\Gaussian_', '\\Gradient_', '\\NMS_', '\\Low_', '\\High_', '\\Egdelink_']
        '''
        imgpath = args.res_dir + mode[0] + args.gradient_mode + '_' + img_dir
        cv2.imwrite(imgpath, res.astype('uint8'))
        imgpath = args.res_dir + mode[1] +  args.gradient_mode + '_' + img_dir
        cv2.imwrite(imgpath, mag.astype('uint8'))
        imgpath = args.res_dir + mode[2] +  args.gradient_mode + '_' + img_dir
        cv2.imwrite(imgpath, NMS_mag.astype('uint8'))
        imgpath = args.res_dir + mode[3] +  args.gradient_mode + '_' + img_dir
        cv2.imwrite(imgpath, low_res.astype('uint8'))
        
        imgpath = args.res_dir + mode[4] +  args.gradient_mode + '_' + img_dir
        cv2.imwrite(imgpath, high_res.astype('uint8'))
        '''
        imgpath = args.res_dir + mode[5] +  str(args.kernel_height) + \
                  'X' + str(args.kernel_width) + '_' + str(args.threshold) + \
                  args.gradient_mode + '_' + args.edge_mode + '_' + img_dir
        cv2.imwrite(imgpath, final.astype('uint8'))


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)

    


