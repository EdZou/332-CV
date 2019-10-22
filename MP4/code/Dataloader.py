from Segmentation import CSfunc
import os
import cv2
from tqdm import tqdm
import imghdr

class ImageDataset(object):
    def __init__(self, data_dir, gt_dir):
        #mode 1 is only rgb, mode 2 is only nrgb, mode 3 is only hsv, mode 4 is all of them
        super(ImageDataset, self).__init__()
        self.data_dir = os.path.expanduser(data_dir)
        self.gt_dir = os.path.expanduser(gt_dir)
        self.datapaths = self.__load_imgpaths_from_dir(self.data_dir)
        self.gtpaths = self.__load_imgpaths_from_dir(self.gt_dir)

    def __is_imgfile(self, filepath):
        filepath = os.path.expanduser(filepath)
        return os.path.isfile(filepath) and imghdr.what(filepath)

    def __load_imgpaths_from_dir(self, dirpath):
        imgpaths = []
        dirpath = os.path.expanduser(dirpath)

        for path in os.listdir(dirpath):
            path = os.path.join(dirpath, path)
            if self.__is_imgfile:
                imgpaths.append(path)
        return imgpaths

    def Make_pixels(self):
        if len(self.datapaths) != len(self.gtpaths):
            raise Exception('The number of label image and real image is not equal')
        pixels = []

        print('loading training datasets...')
        pbar = tqdm(total = len(self.datapaths), desc = 'loading valid pixels...')
        for i in range(len(self.datapaths)):
            datapath = os.path.expanduser(self.datapaths[i])
            gtpath = os.path.expanduser(self.gtpaths[i])

            data = cv2.imread(datapath)
            gt = cv2.imread(gtpath)

            for i in range(len(gt)):
                for j in range(len(gt[0])):
                    if gt[i][j][0] != 0:
                        pixels.append(data[i][j])
            pbar.update()
        pbar.close()
        return pixels



            
            
