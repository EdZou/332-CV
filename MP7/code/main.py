import cv2
import numpy as np
from matplotlib import pyplot as plt
from FaceDetector import FDfunc
import argparse
import os
from Dataloader import ImageDataset
import pickle
from PIL import Image
from images2gif import writeGif
import imageio
from tqdm import tqdm

'''
main function of Face Detector Function
developed by Cong Zou, 11/10/2019

To use it, open the terminal in Linux or cmd in windows
enter the directory of main.py, Dataloader and FaceDetector.py
CAUTION: These two .py files should be in the same file folder
imread the img by changing the root
For instance, enter this order in terminal/cmd
python main.py --img.dir D:/files/image/moon.bmp
to find the image.

'''


parser = argparse.ArgumentParser()
parser.add_argument('--train_dir', type = str, default = 'video\\image_girl')
parser.add_argument('--train_mode', action = 'store_true', default = False)
parser.add_argument('--img_dir', type = str, default = 'video\\image_girl\\0001.jpg')
parser.add_argument('--res_dir', type = str, default = None)
parser.add_argument('--delta', type = int, default = 1)
parser.add_argument('--comp_mode',type = int, default = 1)
#comp_mode is compare mode, 1 corresponding to SSD, 2 to CC, 3 to NCC
#4 corresponding to cvpr98 without unit vector

def main(args):
    img_dir = os.path.expanduser(args.img_dir)
    print(img_dir)
    testimg = cv2.imread(img_dir)
    dl = ImageDataset(args.train_dir)
    
    fd = FDfunc(testimg, args.delta, args.comp_mode)
    '''
    plt.imshow(ini[...,[2,1,0]])
    plt.title('Initialized Image')
    plt.show()

    plt.imshow(roi, cmap = 'gray')
    plt.title('Face Part')
    plt.show()
    '''
    
    
    #=====================save images results=============================
    if args.res_dir != None:
        args.res_dir = os.path.expanduser(args.res_dir)
        rpath = ['\\ssd.gif', '\\cc.gif','ncc.gif']
            
        if os.path.exists(args.res_dir) == False:
            os.makedirs(args.res_dir)
        '''
        frames = []

        for img_name in dl.datapaths:
            frames.append(imageio.imread(img_name))

        gifpath = args.res_dir + '\\girl.gif'
        imageio.mimsave(gifpath, frames, 'GIF', duration = 0.1)
        '''
        #save gif and images
        frames = []
        dirs = ['SSD', 'CC', 'NCC','cvpr98','ncvpr98']
        pbar = tqdm(total = len(dl.datapaths), desc = 'Detecting Face...')
        if os.path.exists(args.res_dir + '\\'+ dirs[args.comp_mode-1]) == False:
            os.makedirs(args.res_dir + '\\'+ dirs[args.comp_mode-1])
        for img_name in dl.datapaths:
            fimg = cv2.imread(img_name)
            resimg = fd.Forward(fimg)
            imgpath = args.res_dir + '\\'+ dirs[args.comp_mode-1] + '\\' + img_name[-8:]
            cv2.imwrite(imgpath, resimg)
            frames.append(resimg[...,[2,1,0]])
            pbar.update()
        pbar.close()

        gifpath = args.res_dir + '\\' + dirs[args.comp_mode-1] + '.gif'
        imageio.mimsave(gifpath, frames, 'GIF', duration = 0.1)
        
            


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)

    


