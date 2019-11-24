import os
import cv2
from tqdm import tqdm
import imghdr

class ImageDataset(object):
    def __init__(self, data_dir):
        super(ImageDataset, self).__init__()
        self.data_dir = os.path.expanduser(data_dir)
        self.datapaths = self.__load_imgpaths_from_dir(self.data_dir)

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



            
            
