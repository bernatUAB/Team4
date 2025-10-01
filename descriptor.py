import numpy as np
import cv2
from pathlib import Path
import pickle


class ImageDescriptor:
    
    ''' Class to compute descriptors using 1D histograms'''

    def __init__(self, color_mapping=None, color_space='HSV', bins_per_channel = 32):
        self.color_mapping = color_mapping
        self.color_space = color_space.upper() # 'RGB', 'HSV', 'LAB', 'GRAY', 'YCrCb', 'Cielab'
        self.bins_per_channel = bins_per_channel

    def compute_descriptor(self, image: np.ndarray):
        
        match self.color_mapping:
            case 'MAX_ABS_SCALE':
                image = (image - image.min()) / (image.max() - image.min())
            case 'STD_SCALER':
                image = (image - image.mean()) / image.std()

        match self.color_space:
            case 'GRAY':
                converted_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                channels = [converted_img]
                ranges = [(0, 256)]
            case 'HSV':
                converted_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                channels = cv2.split(converted_img)
                ranges = [(0, 180), (0, 256), (0, 256)]
            case 'RGB':
                converted_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                channels = cv2.split(converted_img)
                ranges = [(0, 256), (0, 256), (0, 256)]
            case 'LAB':
                converted_img = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
                channels = cv2.split(converted_img)
                ranges = [(0, 256), (0, 256), (0, 256)]
            case 'YCRCB':
                converted_img = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
                channels = cv2.split(converted_img)
                ranges = [(0, 256), (0, 256), (0, 256)]
            case 'HLS':
                converted_img = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
                channels = cv2.split(converted_img)
                ranges = [(0, 180), (0, 256), (0, 256)]
            case _:
                raise ValueError(f"Sorry, we do not have this color space yet: {self.color_space}")
        
        histograms = []
        for i, channel in enumerate(channels):
            hist = np.histogram(channel, bins=self.bins_per_channel, range=ranges[i])[0]
            # print(hist.shape)
            histograms.append(hist)
            
        descriptor = np.concatenate(histograms)
        return descriptor
                
    #TODO COMPUTE HISTO FOR WHOLE DATASET
                
def create_method1_descriptor(bins=32):
    "descriptor for hsv color space"
    return ImageDescriptor(color_space='HSV', bins_per_channel=bins)

def create_method2_descriptor(bins=256):
    "descriptor for gray color space"
    return ImageDescriptor(color_space='GRAY', bins_per_channel=bins)


if __name__ == "__main__":
    
    #Paths for the dataset
    BBDD_DIR = "dataset/BBDD"
    QSD1_DIR = "dataset/QSD1"
    OUTPUT_DIR = "descriptors"
    
    # dataset/qsd1_w1/00003.jpg
    descr = ImageDescriptor(color_mapping='MAX_ABS_SCALE', color_space='RGB', bins_per_channel=64)
    img = cv2.imread("dataset/qsd1_w1/00003.jpg")
    hist = descr.compute_descriptor(img)
    print(hist.shape)
    print(hist.max())
    print(hist.min())
    print(hist.mean())
    print(hist.std())
    