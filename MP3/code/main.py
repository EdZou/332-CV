import cv2
import numpy as np
from matplotlib import pyplot as plt
from Histogram import HEfunc
import argparse
import os

'''
main function of Histogram Equalization algorithm
developed by Cong Zou, 10/10/2019

In Python3.7 (Libraries include Matplotlib, opencv(cv2), numpy, argparse, os)
unzip the file, open terminal/cmd and cd to MP2_CongZou\code. Type:
python main.py --img_dir moon.bmp --res_dir results --plane_fitting --quadratic_mode --process_mode 3
--img_dir is the directory of image, default to be 'moon.bmp'
--res_dir is the directory to save the output, default to be None. Once set, will automatically create file folder and save the results.
--plane_fitting is the switch of whether using lighting correction
--quadratic_mode is the switch of whether using quadratic mode
--process_mode is designed to choose the processing mode: 1 only scaled, 2 only truncated, 3 both
'''


parser = argparse.ArgumentParser()
parser.add_argument('--img_dir', type = str, default = 'moon.bmp')
parser.add_argument('--res_dir', type = str, default = None)
parser.add_argument('--plane_fitting', action = 'store_true', default = False)
parser.add_argument('--quadratic_mode', action = 'store_true', default = False)
parser.add_argument('--process_mode', type = int, default = 3)

def main(args):
    img_dir = args.img_dir
    args.img_dir = os.path.expanduser(args.img_dir)
    img = cv2.imread(args.img_dir)
    if img.any() == False:
        raise Exception('The image imreading failed')
    if args.process_mode != 1 and args.process_mode != 2 and args.process_mode != 3:
        raise Exception('Wrong process mode')

    plt.imshow(img)
    plt.title('Original Image')
    plt.show()

    he = HEfunc(img)
    res, plane, scaled, truncated = he.Forward(fit = args.plane_fitting, \
                                               plane_mode = args.quadratic_mode, \
                                               process_mode = args.process_mode, \
                                               res_dir = args.res_dir)

    plt.imshow(res)
    plt.title('Histogram Equalization')
    plt.show()
    

    if args.plane_fitting:
        
        plt.imshow(plane)
        plt.title('Plane Fitting')
        plt.show()

        if args.process_mode == 2 or args.process_mode == 3:
            plt.imshow(truncated)
            plt.title('Truncated Version')
            plt.show()

        if args.process_mode == 1 or args.process_mode == 3:
            plt.imshow(scaled)
            plt.title('Scaled Version')
            plt.show()
            


    #save images results
    if args.res_dir != None:
        args.res_dir = os.path.expanduser(args.res_dir)
        if os.path.exists(args.res_dir) == False:
            os.makedirs(args.res_dir)
        imgpath = args.res_dir + '\HE_' + img_dir
        cv2.imwrite(imgpath, res)
        
        if args.plane_fitting:
            #imwrite the image of plane
            if args.quadratic_mode:
                plane_mode = 'LC_Quadratic_'
            else:
                plane_mode = 'LC_Linear_'

            imgpath = args.res_dir + '\_' + plane_mode + img_dir
            cv2.imwrite(imgpath, plane)

            #imwrite the image of final version
            if args.process_mode == 1 or args.process_mode == 3:
                process_mode = 'Scale_'
                imgpath = args.res_dir + '\_' + plane_mode + process_mode + img_dir
                cv2.imwrite(imgpath, scaled)

            if args.process_mode == 2 or args.process_mode == 3:
                process_mode = 'Truncate_'
                imgpath = args.res_dir + '\_' + plane_mode + process_mode + img_dir
                cv2.imwrite(imgpath, truncated)

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)

    


