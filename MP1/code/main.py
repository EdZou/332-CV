import cv2
from matplotlib import pyplot as plt
from CCL import CCLfunc
import argparse
import os

'''
main function of connected component labeling algorithm
developed by Cong Zou, 10/2/2019

To use it, open the terminal in Linux or cmd in windows
enter the directory of main.py and CCL.py
CAUTION: These two .py files should be in the same file folder
imread the img by changing the root
For instance, enter this order in terminal/cmd
python main.py --img.dir D:/files/image/face.bmp
to find the image.

You can adjust the size of threshold by --threshold
e.g.
python main.py --size_filter --threshold 250
then the size filter function is open and the threshold is set as 250
'''


parser = argparse.ArgumentParser()
parser.add_argument('--img_dir', type = str, default = 'gun.bmp')
parser.add_argument('--res_dir', type = str, default = None)
parser.add_argument('--size_filter', action = 'store_true', default = False)
parser.add_argument('--threshold', type = int, default = 300)

def main(args):
    img_dir = args.img_dir
    args.img_dir = os.path.expanduser(args.img_dir)
    img = cv2.imread(args.img_dir)
    if img.any() == False:
        raise Exception("the image imreading failed")

    plt.imshow(img)
    plt.show()

    ccl = CCLfunc(img)
    res = ccl.Forward()

    plt.imshow(res)
    plt.show()

    #if args.size_filter is true, execute filter function in CCL
    if args.size_filter:
        fccl = CCLfunc(img)
        fres = fccl.Filter_forward(args.threshold)

        plt.imshow(fres)
        plt.show()

    #save images results
    if args.res_dir != None:
        args.res_dir = os.path.expanduser(args.res_dir)
        if os.path.exists(args.res_dir) == False:
            os.makedirs(args.res_dir)
        imgpath = args.res_dir + '\CCL_' + img_dir
        cv2.imwrite(imgpath, res)
        
        if args.size_filter:
            imgpath = args.res_dir + '\Filter' + str(args.threshold) + '_' + img_dir
            cv2.imwrite(imgpath, fres)

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)

    


