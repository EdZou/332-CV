import cv2
import numpy as np
from matplotlib import pyplot as plt
from CannyEdgeDetector import CEDfunc
import argparse
import os
import pickle
from Hough import HoughTransform

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
parser.add_argument('--img_dir', type = str, default = 'input.bmp')
parser.add_argument('--res_dir', type = str, default = None)
parser.add_argument('--kernel_height', type = int, default = 3)
parser.add_argument('--kernel_width', type = int, default = 3)
parser.add_argument('--sigma',type = float, default = 3.0)
parser.add_argument('--gradient_mode', type = str, choices = ['lut', 'interpolation'], default = 'interpolation')
parser.add_argument('--edge_mode', type = str, choices = ['recursive', 'hysteresis'], default = 'recursive')
parser.add_argument('--h_threshold', type = float, default = 0.95)
parser.add_argument('--degree', type = float, default = 1)
parser.add_argument('--f_threshold', type = float, default = 0.04)
parser.add_argument('--f_kernel', type = float, default = 0.1)
#parameters for input.bmp --degree 2 --f_threshold 0.026 or --degree 0.5 --f_threshold 0.045 --f_kernel 0.2
#parameters for test.bmp  --f_threshold 0.07
#for test2.bmp, same parameters

def main(args):
    #===========================Test=============================
    img_dir = args.img_dir
    args.img_dir = os.path.expanduser(args.img_dir)
    img = cv2.imread(args.img_dir)
    if args.f_kernel > 1:
        raise ValueError('f_kernel can never be bigger than 1')
    if img.any() == False:
        raise Exception('The image imreading failed')

    k_size = [args.kernel_height, args.kernel_width]

    plt.imshow(img[...,[2,1,0]])
    plt.title('Original Image')
    plt.show()

    csd = CEDfunc(np.copy(img), args.gradient_mode, args.edge_mode, args.h_threshold)
    res, mag, NMS_mag, low_res, high_res, final = csd.Forward(np.copy(img), k_size, args.sigma)
    '''
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
    '''
    plt.imshow(final[...,[2,1,0]].astype('uint8'))
    plt.title('Edge Image')
    plt.show()

    ht = HoughTransform(final[...,0], args.degree, args.f_threshold, np.copy(img[...,0]), args.f_kernel)
    ptheta, s_ptheta, res = ht.Forward()

    plt.imshow(ptheta, cmap = 'gray')
    plt.title('P-Theta Image')
    plt.show()
    
    plt.imshow(s_ptheta.astype('uint8'))
    plt.title('P-Theta Image After Second Filter')
    plt.show()

    plt.imshow(res.astype('uint8'), cmap = 'gray')
    plt.title('Final Image')
    plt.show()
    

    #=====================save images results=============================
    if args.res_dir != None:
        args.res_dir = os.path.expanduser(args.res_dir)
            
        if os.path.exists(args.res_dir) == False:
            os.makedirs(args.res_dir)

        imgpath = args.res_dir + '\\P-Theta' + '_' + str(args.degree) \
                  + '_' + str(args.f_threshold) + '_' + img_dir
        cv2.imwrite(imgpath, ptheta.astype('uint8'))

        imgpath = args.res_dir + '\\Filtered' + '_' + str(args.degree) \
                  + '_' + str(args.f_threshold) + '_' + img_dir
        cv2.imwrite(imgpath, s_ptheta.astype('uint8'))

        imgpath = args.res_dir + '\\' + str(args.degree) + '_' \
                  + str(args.f_threshold) + '_' + img_dir
        cv2.imwrite(imgpath, res.astype('uint8'))


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)

    


