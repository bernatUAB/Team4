import numpy as np
import cv2
from pathlib import Path
import pickle
import matplotlib.pyplot as plt


class ImageDescriptor:
    
    ''' Class to compute descriptors using 1D histograms'''

    def __init__(self, color_mapping=None, color_space='HSV', bins_per_channel = 32):
        self.color_mapping = color_mapping
        self.color_space = color_space.upper() # 'RGB', 'HSV', 'LAB', 'GRAY', 'YCrCb', 'Cielab'
        self.bins_per_channel = bins_per_channel

    def compute_descriptor(self, image: np.ndarray):
        
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
        
        # Print info before normalization
        print(f"\n=== Before normalization ===")
        for i, ch in enumerate(channels):
            print(f"Channel {i}: dtype={ch.dtype}, shape={ch.shape}, min={ch.min()}, max={ch.max()}, mean={ch.mean():.2f}, std={ch.std():.2f}")
        
        # Apply normalization after color conversion
        match self.color_mapping:
            case 'MAX_ABS_SCALE':
                channels = [(ch - ch.min()) / (ch.max() - ch.min()) for ch in channels]
                # Update also the ranges
                ranges = [(0, 1) for _ in ranges]
            case 'STD_SCALER':
                channels = [(ch - ch.mean()) / ch.std() for ch in channels]
                # Update ranges based on actual min/max after standardization
                ranges = [(ch.min(), ch.max()) for ch in channels]
        
        # Print info after normalization
        if self.color_mapping:
            print(f"\n=== After normalization ({self.color_mapping}) ===")
            for i, ch in enumerate(channels):
                print(f"Channel {i}: dtype={ch.dtype}, shape={ch.shape}, min={ch.min():.4f}, max={ch.max():.4f}, mean={ch.mean():.4f}, std={ch.std():.4f}")
        
        histograms = []
        for i, channel in enumerate(channels):
            hist = np.histogram(channel, bins=self.bins_per_channel, range=ranges[i])[0]
            hist = hist.astype(np.float32)
            hist /= hist.sum() + 1e-8  # Normalize histogram
            #Plot the histogram
            print(f"Histogram Channel {i}: shape={hist.shape}, min={hist.min()}, max={hist.max()}, mean={hist.mean():.2f}, std={hist.std():.2f}")
            plt.figure()
            plt.title(f"Histogram Channel {i}")
            plt.plot(hist)
            plt.xlabel('Bin')
            plt.ylabel('Frequency')
            plt.show()
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
    img = cv2.imread("qsd1_w1/00003.jpg")
    
    hist = descr.compute_descriptor(img)
    
    print(hist.shape)
    print(hist.max())
    print(hist.min())
    print(hist.mean())
    print(hist.std())
    print(hist.sum())
    