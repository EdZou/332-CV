import cv2
import numpy as np
from matplotlib import pyplot as plt
from Histogram import HEfunc
import argparse
import os

'''
main function of connected component labeling algorithm
developed by Cong Zou, 10/10/2019

To use it, open the terminal in Linux or cmd in windows
enter the directory of main.py and Histogram.py
CAUTION: These two .py files should be in the same file folder
imread the img by changing the root
For instance, enter this order in terminal/cmd
python main.py --img.dir D:/files/image/moon.bmp
to find the image.

You can use lighting correction func by --plane_fitting
and decide whether to use quadratic mode by --quadratic mode
To get the results of different version, use --process_mode
1 is only scaled version, 2 is only truncated version, 3 is both of them
e.g.
python main.py --img_dir moon.bmp --res_dir results --plane_fitting --quadratic_mode --process_mode 3
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

    


