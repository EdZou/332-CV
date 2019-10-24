import cv2
import numpy as np
from matplotlib import pyplot as plt
from Segmentation import CSfunc
import argparse
import os
from Dataloader import ImageDataset
import pickle

'''
main function of Color Segmentation
developed by Cong Zou, 10/16/2019

In Python3.7 (Libraries include Matplotlib, opencv(cv2), numpy, argparse, os，tqdm,imghdr)
unzip the file, open terminal/cmd and cd to MP4_CongZou\code. Type:
python main.py --oneimage_dir hand.jpg --res_dir results --img_dir joy1.jpg --process_mode 6
--oneimage_dir is the switch and path to use 1 image as training data, default = None
--train_dir, --gt_dir is the path to use dataset training, default = 'datasets\FacePhoto' and 'datasets\GT_FacePhoto'
--config_dir is the path to save and use the model param, if = None then don't save, default = 'config'
--train_mode switch whether to use train_mode, default = False
--img_dir is the directory of image, default to be 'joy1.bmp'
--res_dir is the directory to save the output, default to be None. Once set, will automatically create file folder and save the results.
--process_mode is designed to choose the processing mode: 1 RGB Histogram, 2 NRGB Histogram, 3  HSI Histogram, 4 RGB Gaussian, 5 NRGB Gaussian, 6 HSI Gaussian
--h_threshold threshold for Histogram, 0.0001 for hsi(mode 3), 0.0002 for nrgb(mode 2), 0.00002 for rgb(mode 1). For one image version, all of them should be 1*e^-16
--g_threshold threshold for Gaussian, 1.5 for RGB(mode 4), 7 for NRGB(mode 5), 1.1 for HSI(mode 6). One image version, 6.8 for HSI, 35 for NRGB, 3 for RGB
'''


parser = argparse.ArgumentParser()
parser.add_argument('--oneimage_dir', type = str, default = None)
parser.add_argument('--train_dir', type = str, default = 'datasets\FacePhoto')
parser.add_argument('--gt_dir', type = str, default = 'datasets\GT_FacePhoto')
parser.add_argument('--config_dir', type = str, default = 'config')
parser.add_argument('--train_mode', action = 'store_true', default = False)
parser.add_argument('--img_dir', type = str, default = 'joy1.bmp')
parser.add_argument('--res_dir', type = str, default = None)
parser.add_argument('--process_mode', type = int, default = 3)
parser.add_argument('--h_threshold', type = float, default = 0.0001)
#Histogram: 0.0001 for hsi(mode 3), 0.0002 for nrgb(mode 2), 0.00002 for rgb(mode 1)
#For one image version, all of them should be 1*e^-16
parser.add_argument('--g_threshold', type = float, default = 1.02)
#Gaussian: 1.5 for RGB, 7 for NRGB, 1.1 for HSI
#For one image version, 6.8 for HSI, 35 for NRGB, 3 for RGB

def main(args):
    if args.process_mode < 1 or args.process_mode > 6:
        raise Exception('Wrong process mode')
    cpaths = ['\\rgb.p','\\nrgb.p','\\hsi.p','\\G_rgb.p','\\G_nrgb.p','\\G_hsi.p',]
    
    #=============Dataload and Train/Get_Histogram====================
    #dataload and pixels making
    #Here, dataset refers to Tan et al, IEEE T-II, 2012

    cs = CSfunc([])

    if args.train_mode:
        if args.oneimage_dir == None:
            idata = ImageDataset(args.train_dir, args.gt_dir)
            pixels = np.array(idata.Make_pixels())
        else:
            args.oneimage_dir = os.path.expanduser(args.oneimage_dir)
            idata = cv2.imread(args.oneimage_dir)
            pixels = np.reshape(idata,(-1,3))
            
        param = cs.Train(pixels, args.process_mode)
        #save the parameters
        if args.config_dir != 'None':
            args.config_dir = os.path.expanduser(args.config_dir)
            if args.oneimage_dir != None:
                args.config_dir = args.config_dir + '\\1img_config'
            else:
                args.config_dir = args.config_dir + '\\dataset_config'
            if os.path.exists(args.config_dir) == False:
                os.makedirs(args.config_dir)
            cpath = args.config_dir + cpaths[args.process_mode - 1]
            with open(cpath, 'wb') as file:
                pickle.dump(param, file)

    else:
        if args.config_dir == None:
            raise Exception('Trained Histogram cannot be loaded from None directory')
        args.config_dir = os.path.expanduser(args.config_dir)
        if args.oneimage_dir != None:
            args.config_dir = args.config_dir + '\\1img_config'
        else:
            args.config_dir = args.config_dir + '\\dataset_config'
        cpath = args.config_dir + cpaths[args.process_mode - 1]
        with open(cpath, mode = 'rb') as file:
            param = pickle.load(file)
        
            
    #===========================Test=============================
    img_dir = args.img_dir
    args.img_dir = os.path.expanduser(args.img_dir)
    img = cv2.imread(args.img_dir)
    if img.any() == False:
        raise Exception('The image imreading failed')
    if args.process_mode < 1 or args.process_mode > 6:
        raise Exception('Wrong process mode')

    plt.imshow(img[...,[2,1,0]])
    plt.title('Original Image')
    plt.show()

    if args.process_mode > 0 and args.process_mode < 4:
        res = cs.Histogram_test(img, param, args.h_threshold, args.process_mode)
    else:
        res = cs.Guassian_test(img, param, args.g_threshold, args.process_mode)
    plt.imshow(res[...,[2,1,0]])
    plt.title('Processed Image')
    plt.show()


    #=====================save images results=============================
    if args.res_dir != None:
        args.res_dir = os.path.expanduser(args.res_dir)
        if args.oneimage_dir != None:
            args.res_dir = args.res_dir + '\\1img_results'
        else:
            args.res_dir = args.res_dir + '\\dataset_results'
            
        if os.path.exists(args.res_dir) == False:
            os.makedirs(args.res_dir)

        mode = ['\\RGB_', '\\NRGB_', '\\HSI_', '\\Gaussian_RGB_', '\\Gaussian_NRGB_', '\\Gaussian_HSI_']
        if args.process_mode > 3:
            threshold = args.g_threshold
        else:
            threshold = args.h_threshold
        imgpath = args.res_dir + mode[args.process_mode - 1] + str(threshold) + img_dir
        cv2.imwrite(imgpath, res*255)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)

    


