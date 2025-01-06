import cv2  # For image processing
import numpy as np  # For numerical operations
from typing import Union, Tuple  # For type annotations
from PIL import Image  # For resizing images with aspect ratio
import matplotlib.pyplot as plt  # For displaying images
import math  # For mathematical operations
import scipy.ndimage  # For binary fill holes
import tensorflow as tf  # For loading and using the segmentation model
import os
from tensorflow import keras
import segmentation_models as sm
from deskew import determine_skew  # For calculating skew angle of images


import easyocr
from glob import glob
import pandas as pd


#Function to resize image.
def prod_resize_input(img_link):
    '''
    Function takes an image and resizes it.
    '''
    img = cv2.imread(img_link)
    img = cv2.resize(img, (224, 224))
    return img.astype('uint8')

#Create function to crop images.
def crop_for_seg(img, bg, mask):
    '''
    Function extracts an image where it overlaps with its binary mask.
    img - Image to be cropped.
    bg - The background on which to cast the image.
    mask - The binary mask generated from the segmentation model.
    '''
    #mask = mask.astype('uint8')
    fg = cv2.bitwise_or(img, img, mask=mask) 
    fg_back_inv = cv2.bitwise_or(bg, bg, mask=cv2.bitwise_not(mask))
    New_image = cv2.bitwise_or(fg, fg_back_inv)
    return New_image

def extract_meter(image_to_be_cropped):
    '''
    Function further extracts image such that the meter reading takes up the majority of the image.
    The function finds the edges of the ROI and extracts the portion of the image that contains the entire ROI.
    '''
    where = np.array(np.where(image_to_be_cropped))
    if where.size == 0:
        return image_to_be_cropped
    x1, y1, z1 = np.amin(where, axis=1)
    x2, y2, z2 = np.amax(where, axis=1)
    sub_image = image_to_be_cropped.astype('uint8')[x1:x2, y1:y2]
    return sub_image

def rotate(image: np.ndarray, angle: float, background: Union[int, Tuple[int, int, int]]) -> np.ndarray:
    '''
    This function attempts to rotate meter reading images to make them horizontal.
    Its arguments are as follows:
    
    image - The image to be deskewed (in numpy array format).
    angle - The current angle of the image, found with the determine_skew function of the deskew library.
    background - The pixel values of the boarder, either int (default 0) or a tuple.
    
    The function returns a numpy array.
    '''
    old_width, old_height = image.shape[:2]
    angle_radian = math.radians(angle)
    width = abs(np.sin(angle_radian) * old_height) + abs(np.cos(angle_radian) * old_width)
    height = abs(np.sin(angle_radian) * old_width) + abs(np.cos(angle_radian) * old_height)
    
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    rot_mat[1, 2] += (width - old_width) / 2
    rot_mat[0, 2] += (height - old_height) / 2
    return cv2.warpAffine(image, rot_mat, (int(round(height)), int(round(width))), borderValue=background)

def resize_aspect_fit(img, final_size: int):
    '''
    Function resizes the image to specified size.
    
    path - The path to the directory with images.
    final_size - The size you want the final images to be. Should be in int (will be used for w and h).
    write_to - The file you wish to write the images to. 
    save - Whether to save the files (True) or return them.
    '''   
    im_pil = Image.fromarray(img)
    size = im_pil.size
    ratio = float(final_size) / max(size)
    new_image_size = tuple([int(x*ratio) for x in size])
    im_pil = im_pil.resize(new_image_size, Image.ANTIALIAS)
    new_im = Image.new("RGB", (final_size, final_size))
    new_im.paste(im_pil, ((final_size-new_image_size[0])//2, (final_size-new_image_size[1])//2))
    new_im = np.asarray(new_im)
    return np.array(new_im)

def prep_for_ocr(img):
    img = resize_aspect_fit(img, 224)
    output_name = 'test_img_for_ocr.jpg'
    cv2.imwrite(output_name, img)
    return output_name

def expand_mask_surrounding(mask):
    global new_mask  # Declare that we're using the global variable
    new_mask = mask.copy()  # Make a copy of the mask and assign it to the global variable
    positions = np.column_stack(np.where(mask == 1))
    if positions.size == 0:
        return new_mask
    min_x, min_y = positions.min(axis=0)
    max_x, max_y = positions.max(axis=0)
    new_mask[min_x:max_x + 1, min_y:max_y + 1] = 1
    return new_mask

#Segment input image.
def segment_input_img(img):
    os.environ["SM_FRAMEWORK"] = "tf.keras"    
    #Resize image.
    img_small = prod_resize_input(img)
    
    #Open image and get dimensions.
    input_img = cv2.imread(img, cv2.IMREAD_UNCHANGED)
    input_w = int(input_img.shape[1])
    input_h = int(input_img.shape[0])
    dim = (input_w, input_h)
    
    # #Load model, preprocess input, and obtain prediction.
    # BACKBONE = 'resnet34'
    # preprocess_input = sm.get_preprocessing(BACKBONE)
    # img_small = preprocess_input(img_small)
    img_small = img_small.reshape(-1, 224, 224, 3).astype('uint8')
    model = tf.keras.models.load_model('segment_model.hdf5', custom_objects={'binary_crossentropy_plus_jaccard_loss': sm.losses.bce_jaccard_loss, 'iou_score' : sm.metrics.iou_score})
    mask = model.predict(img_small)
    
    #Change type to uint8 and fill in holes.
    mask = mask.astype('uint8')
    mask = scipy.ndimage.morphology.binary_fill_holes(mask[0, :, :, 0]).astype('uint8')
    
    #Resize mask to equal input image size.
    mask = cv2.resize(mask, dsize=dim, interpolation=cv2.INTER_AREA)
   
    # Taking a matrix of size 5 as the kernel
    kernel = np.ones((10,10), np.uint8)
    
    mask = cv2.dilate(mask, kernel, iterations=3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    
    #Create background array.
    bg = np.zeros_like(input_img, 'uint8')
    expand_mask = expand_mask_surrounding(mask)
 
    #Get new cropped image and make RGB.
    New_image = crop_for_seg(input_img, bg, expand_mask)
    New_image = cv2.cvtColor(New_image, cv2.COLOR_BGR2RGB)
    plt.imshow(New_image)
    #Extract meter portion.
    extracted = extract_meter(New_image)
    
    grayscale = cv2.cvtColor(extracted, cv2.COLOR_BGR2GRAY)
    angle = determine_skew(grayscale)
    
    if angle == None:
        angle = 1
    
    rotated = rotate(extracted, angle, (0, 0, 0))
    
    return rotated

def read_number_eazyocr(img_path):
    reader = easyocr.Reader(['en'], gpu = True)
    results = reader.readtext(img_path, allowlist ='0123456789.')
    df = pd.DataFrame(results, columns=['bbox','text','conf'])
    return ''.join(df['text'].tolist())

def read_meter_reading(img_path):
    segment_img = segment_input_img(img_path)
    return read_number_eazyocr(segment_img)

def save_segment_img(seg_img, img_path):
    return cv2.imwrite(img_path, seg_img)