import os

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from patchify import patchify, unpatchify

import tensorflow as tf
import keras.backend as K
from keras.models import load_model

from skimage.morphology import skeletonize
from skan import summarize, Skeleton

from keras import backend as K

def f1(y_true, y_pred):
    """
    Calculate the F1 score metric.

    Parameters:
    - y_true (tensorflow.Tensor): Ground truth values.
    - y_pred (tensorflow.Tensor): Predicted values.

    Returns:
    - f1_score (tensorflow.Tensor): The F1 score.
    """
    def recall_m(y_true, y_pred):
        """
        Calculate the recall metric.

        Parameters:
        - y_true (tensorflow.Tensor): Ground truth values.
        - y_pred (tensorflow.Tensor): Predicted values.

        Returns:
        - recall (tensorflow.Tensor): The recall.
        """
        TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        Positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = TP / (Positives + K.epsilon())
        return recall
    
    def precision_m(y_true, y_pred):
        """
        Calculate the precision metric.

        Parameters:
        - y_true (tensorflow.Tensor): Ground truth values.
        - y_pred (tensorflow.Tensor): Predicted values.

        Returns:
        - precision (tensorflow.Tensor): The precision.
        """
        TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        Pred_Positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = TP / (Pred_Positives + K.epsilon())
        return precision
    
    # Calculate precision and recall using the helper functions
    precision, recall = precision_m(y_true, y_pred), recall_m(y_true, y_pred)
    
    # Calculate the F1 score using precision and recall
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))

def iou(y_true, y_pred):
    """
    Calculate the Intersection over Union (IoU) metric.

    Parameters:
    - y_true (tensorflow.Tensor): Ground truth values.
    - y_pred (tensorflow.Tensor): Predicted values.

    Returns:
    - iou_score (tensorflow.Tensor): The IoU score.
    """
    def f(y_true, y_pred):
        """
        Calculate the IoU for each sample.

        Parameters:
        - y_true (tensorflow.Tensor): Ground truth values.
        - y_pred (tensorflow.Tensor): Predicted values.

        Returns:
        - iou_per_sample (tensorflow.Tensor): IoU for each sample.
        """
        threshold = 0.5 
        y_pred_binary = K.round(y_pred + 0.5 - threshold)
       
        intersection = K.sum(K.abs(y_true * y_pred_binary), axis=[1,2,3])
        total = K.sum(K.square(y_true), [1,2,3]) + K.sum(K.square(y_pred_binary), [1,2,3])
        union = total - intersection
        return (intersection + K.epsilon()) / (union + K.epsilon())
    
    # Calculate the mean IoU across all samples
    return K.mean(f(y_true, y_pred), axis=-1)


custom_objects = {"f1": f1, "iou": iou}


model = load_model(f"root_4.h5", custom_objects=custom_objects)

def padder(image, patch_size = 256):
    """
    Pad an image to make its dimensions divisible by a specified patch size so it can be processed by the root prediction model.

    Parameters:
    - image (numpy.ndarray): The input image to be padded.
    - patch_size (int): The desired patch size for the dimensions of the padded image, has a default of 256 to work with the model.

    Returns:
    - padded_image (numpy.ndarray): The padded image.
    """

    # Get the height and width of the input image
    h, w = image.shape[0], image.shape[1]

    # Calculate the amount of padding needed to make the height divisible by patch_size
    height_padding = ((h // patch_size) + 1) * patch_size - h

    # Calculate the amount of padding needed to make the width divisible by patch_size
    width_padding = ((w // patch_size) + 1) * patch_size - w

    # Calculate the top and bottom padding based on the height padding
    top_padding = int(height_padding / 2)
    bottom_padding = height_padding - top_padding

    # Calculate the left and right padding based on the width padding
    left_padding = int(width_padding / 2)
    right_padding = width_padding - left_padding

    # Use cv2.copyMakeBorder to pad the image with constant values of [0, 0, 0]
    padded_image = cv2.copyMakeBorder(image, top_padding, bottom_padding, left_padding, right_padding, cv2.BORDER_CONSTANT, value=[0, 0, 0])

    print(f'\nTop padding is {top_padding}')
    print(f'Bottom padding is {bottom_padding}')
    print(f'Left padding is {left_padding}')
    print(f'Right padding is {right_padding}\n')

    # Return the padded image
    return padded_image


def model_predictions(image_path, patch_size = 256):

    # Read the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    x, y, w, h = 740, 50, 2804, 2804

    image = image[y:y+h, x:x+w]
    

    # Pad the image
    image = padder(image, patch_size)

    # Patchify the image
    patches = patchify(image, (patch_size, patch_size), step=patch_size)

    i, j = patches.shape[0], patches.shape[1]

    # Reshape patches for model prediction
    patches = patches.reshape(-1, patch_size, patch_size, 1)

    # Predict using the model
    preds = model.predict(patches / 255)

    # Reshape predictions
    preds = preds.reshape(i, j, 256, 256)

    # Unpatchify to get the full predicted mask
    predicted_mask = unpatchify(preds, (image.shape[0], image.shape[1]))

    # Apply threshold
    predicted_mask = predicted_mask > 0.5

    # Convert to uint8
    predicted_mask = predicted_mask.astype(np.uint8)

    return predicted_mask


def landmarks(image_path):
    mask = model_predictions(image_path, 256)
    
    # Read mask image as grayscale and apply dilation
    kernel = np.ones((5, 5), dtype="uint8")
    im_dilation = cv2.dilate(mask.astype(np.uint8), kernel, iterations=2)

    # Apply connected components
    _, labels, stats, _ = cv2.connectedComponentsWithStats(im_dilation, connectivity=8)
    areas = stats[:, cv2.CC_STAT_AREA]
    sorted_indices = np.argsort(areas)[::-1]
    filtered_mask = np.zeros_like(im_dilation)

    for i in range(min(6, len(sorted_indices))):
        index = sorted_indices[i]
        filtered_mask[labels == index] = im_dilation[labels == index]

    # Skeletonize the filtered image
    skeleton = skeletonize(filtered_mask)

    # Obtain a dataframe of the skeleton data
    skeleton_branch_data = summarize(Skeleton(skeleton))

    # empty list for coordinates
    coordinates = []

    # Apply landmark detection logic to both images
    for skeleton_id in skeleton_branch_data['skeleton-id'].unique():
        # Root tips
        root_tips = skeleton_branch_data[skeleton_branch_data['skeleton-id'] == skeleton_id]

        max_row = root_tips.loc[root_tips['coord-dst-0'].idxmax()]
        coord_src_0_max = int(max_row['coord-dst-0'])
        coord_src_1_max = int(max_row['coord-dst-1'])


        coordinates.append({"primary_root_tip": (coord_src_1_max, coord_src_0_max)})
        coordinates = sorted(coordinates, key=lambda k: k["primary_root_tip"][0])

    return coordinates