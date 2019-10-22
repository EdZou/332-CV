import cv2
from matplotlib import pyplot as plt
from MP2 import MP2func
import argparse
import numpy as np
import os

'''
main function of MP2
developed by Cong Zou, 10/7/2019

To use it, open the terminal in Linux or cmd in windows
enter the directory of main.py and MP2.py
CAUTION: These two .py files should be in the same file folder
imread the img by changing the root
For instance, enter this order in terminal/cmd
python main.py --img.dir D:/files/image/palm.bmp
to find the image.

You can adjust the size of kernel size by --kernel_height and --kernel_width
e.g.
python main.py --action erosion --kernel_height 5 --kernel_width 5
then the function will execute erosion in the kernel size [5,5]

To use the image after filtered:
python main.py --img_filter
'''


parser = argparse.ArgumentParser()
parser.add_argument('--img_dir', type = str, default = 'palm.bmp')
parser.add_argument('--res_dir', type = str, default = None)
parser.add_argument('--action',  type = str, \
                    choices = ['dilation','erosion','open','close','boundary', \
                               'filter'],
                    default = 'dilation')
parser.add_argument('--kernel_height', type = int, default = 3)
parser.add_argument('--kernel_width', type = int, default = 3)
parser.add_argument('--img_filter', action = 'store_true', default = False)


def main(args):
    img_dir = args.img_dir
    args.img_dir = os.path.expanduser(args.img_dir)
    img = cv2.imread(args.img_dir)
    if img.any() == False:
        raise Exception('The image imreading failed')
    if args.kernel_height%2 != 1 or args.kernel_width%2 != 1:
        print('WARNING: Kernel size is supposed to have odd numbers.')
    if args.kernel_height < 1 or args.kernel_width < 1:
        raise Exception('The size of kernel should be greater than [1, 1]')

    k = np.ones([args.kernel_height, args.kernel_width])

    plt.imshow(img)
    plt.show()

    
    mp2 = MP2func(img, k)
    
    if args.img_filter:
        if 'gun.bmp' in args.img_dir:
            img = mp2.Filter(np.copy(img), k)
            img = mp2.Dilation(np.copy(img))
        elif 'palm.bmp' in args.img_dir:
            img = img = mp2.Filter(np.copy(img), k, 3, 0)
        else:
            raise Exception('Not defined cases cannot use img_filter')

        plt.imshow(img)
        plt.show()
        
    if args.action == 'dilation':
        res = mp2.Dilation(np.copy(img))
        '''
        test = cv2.dilate(img, k, iterations = 1)
        plt.imshow(test)
        plt.show()
        '''
        
    elif args.action == 'erosion':
        res = mp2.Erosion(np.copy(img))
        '''
        test = cv2.erode(img, k, iterations = 1)
        plt.imshow(test)
        plt.show()
        '''

    elif args.action == 'open':
        res = mp2.Opening(np.copy(img))
        '''
        test = cv2.morphologyEx(img, cv2.MORPH_OPEN, k)
        plt.imshow(test)
        plt.show()
        '''

    elif args.action == 'close':
        res = mp2.Closing(np.copy(img))
        '''
        test = cv2.morphologyEx(img, cv2.MORPH_CLOSE, k)
        plt.imshow(test)
        plt.show()
        '''

    elif args.action == 'boundary':
        res = mp2.Boundary(np.copy(img))

    elif args.action == 'filter':
        res = mp2.Filter(np.copy(img))
        res = mp2.Dilation(res)

    else:
        raise Exception('Wrong action executed')
        

    plt.imshow(res)
    plt.show()

    #save images results
    if args.res_dir != None:
        args.res_dir = os.path.expanduser(args.res_dir)
        if os.path.exists(args.res_dir) == False:
            os.makedirs(args.res_dir)
        if args.img_filter:
            imgpath = args.res_dir + '\MP2_' + 'clean_' + args.action + '_' +  \
                  str(args.kernel_height) + 'X' +str(args.kernel_width) + img_dir
        else:
            imgpath = args.res_dir + '\MP2_' + args.action + '_' +  \
                  str(args.kernel_height) + 'X' +str(args.kernel_width) + img_dir
        cv2.imwrite(imgpath, res)
        

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)

    


