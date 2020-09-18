# Estrutura básica para projetos de Machine Learning e Deep Learning
# Por Adriano Santos.
# Modulo de Augumentacao de Dados para Modelos de Deteccao de Objetos 

# normalize ok
# letterbox ok
# resize_image ok
# crop ok
# center_crop ok
# random_crop ok
# rotate_cv ok
# random_crop_imagem_bbox ok
# draw_box ok

# RandomScale
# RandomShear
# RandomTranslate
# ToTensor
# transformsXY


import cv2
import numpy as np
from pathlib import Path
from PIL import Image, ImageOps
import matplotlib.pyplot as plt 
import random

class DAOD(object):

    def __init__(self):
        pass

    # Normalize function
    # mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225] values are based on ImageNet
    def normalize(self, image, mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]):
        return (image - mean)/std


    # Letterbox function
    def letterbox(self, image, size):
        (H, W) = image.shape[:2]
        h, w = (size, size)
        
        # Define scale
        scale = min(h / H, w / W)
        nh = int(H * scale)
        nw = int(W * scale)

        image = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_CUBIC)
        new_img = np.full((h, w, 3), 128, dtype='uint8') #180 color
        new_img[(h - nh) // 2:(h - nh) // 2 + nh,
                (w - nw) // 2:(w - nw) // 2 + nw,
                :] = image.copy()
        return new_img
    
    # Resize function
    def resize(self, image, size):
        (H, W) = image.shape[:2]

        f = H if H >= W else W
        r = size/H if H >= W else size/W

        dim = (size, int(f * r))
        
        return cv2.resize(image, dim, interpolation = cv2.INTER_AREA) 


    # Crop function
    def crop(self, image, x, w, y, h): 
        return image[y:h+y, x:x+w]
    

    # Center crop function
    def center_crop(self, image, r=20):
        (H, W) = image.shape[:2]
       
        x = round(r*W/H)
        w =  W-2*x
        y = r
        h = H-2*r

        return self.crop(image, x, w, y, h)

    # Random crop function
    def random_crop(self, image, r=10):
        start_x, w, start_y, h = self._get_random_positions(image.copy(), r)
        return self.crop(image, start_x, w, start_y, h)

    # Random crop image and bounding box together function
    def random_crop_image_bbox(self, image, bbox_image, r=8):
        start_x, w, start_y, h = self._get_random_positions(image.copy(), r)

        new_image = self.crop(image, start_x, w, start_y, h)
        new_bbox = self.crop(bbox_image, start_x, w, start_y, h)

        return new_image, new_bbox

    # Rotate function
    def rotate(self,image, deg, flags=False, mode=cv2.BORDER_REFLECT, interpolation=cv2.INTER_AREA):
        """ Rotates an image by deg degrees"""
        (H, W) = image.shape[:2]

        M = cv2.getRotationMatrix2D((W/2,H/2),deg,1)
        if flags:
            return cv2.warpAffine(image, M,(W,H), borderMode=cv2.BORDER_CONSTANT)
        return cv2.warpAffine(image,M,(W,H), borderMode=mode, flags=cv2.WARP_FILL_OUTLIERS+interpolation)

    # Flipping image function
    # 1 = hor
    # 0 = ver
    # -1 = hor + ver
    def flipping(self, image, op=1):
        return cv2.flip(image, op)


    # Create mask funcion (YOLO patters)
    # bbox = c, x, y, w, h
    def create_mask(self, image, bbox):
         # Start array
        (H, W) = image.shape[:2]
        bbox_image = np.zeros((H, W))
        
        # Get coorden.
        x, y, w, h = self._get_coor_x_y_w_h(image, bbox)

        # Filled retagle
        bbox_image[y:y+h, x:x+w] = 255.

        return bbox_image

    """Convert mask to a bounding box, assumes 0 as background nonzero object"""
    def convert_mask_to_bbox(self, imagem_box):
        (H, W) = imagem_box.shape[:2]

        rows, cols = np.nonzero(imagem_box)
        if len(cols)==0: 
            return np.zeros(4, dtype=np.float32)
        y1 = np.min(rows)
        x1 = np.min(cols)
        y2 = np.max(rows)
        x2 = np.max(cols)
        
        x = (x1 + x2) / 2
        y = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1
        
        box = [x/W,y/H,w/W,h/H]

        return np.array(box, dtype=np.float32)

    # Draw rect in image
    def draw_box(self, image, bbox, color = (255, 0, 0)):
        
        x, y, w, h = self._get_coor_x_y_w_h(image,bbox)
    
        x1, x2, y1, y2 = [x, x+w, y, y+h]

        cv2.rectangle(image,(x1, y1), (x2, y2), color ,2)

        return image

    # Auxiliar function    
    def _get_random_positions(self, image, r=10):
        (H, W) = image.shape[:2]
        x = round(r*W/H)
        w =  W-2*x
        h = H-2*r
        
        rand_y = random.uniform(0, 1)
        rand_x = random.uniform(0, 1)

        start_y = np.floor(2*rand_y*r).astype(int)
        start_x = np.floor(2*rand_x*x).astype(int)
        
        return (start_x, w, start_y, h) 

    def _get_coor_x_y_w_h(self, image, bbox):
        (H, W) = image.shape[:2]
        
        # Get x and y base
        w = float(bbox[2])
        h = float(bbox[3])
        x = float(bbox[0]) 
        y = float(bbox[1])

        # Get x and y base
        x = x - w / 2
        y = y - h / 2

        # Update values according to image dimensions
        x = int(x * W)
        w = int(w * W)
        y = int(y * H)
        h = int(h * H)

        return x, y, w, h

    def show(self, imagem):
        plt.imshow(imagem)
        plt.show()

