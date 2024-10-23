import os
import cv2
import numpy as np
from patchify import patchify, unpatchify





def cropping_images_in_folder(images_folder, reference_image_path):
    """
    Processes all images in the specified folder by applying the cropping reference obtained 
    from the reference image.

    Args:
        images_folder (str): The path to the folder containing images.
        reference_image_path (str): The path to the reference image for consistent cropping.

    Returns:
        tuple: Contains:
            - dict: A dictionary with filenames as keys and cropped images as values
            - tuple: (x, y, w, h) coordinates for cropping
    """
    # Read the reference image in color mode
    reference_im = cv2.imread(reference_image_path, cv2.IMREAD_COLOR)
    
    # Convert the reference image to grayscale for thresholding
    reference_im_gray = cv2.cvtColor(reference_im, cv2.COLOR_BGR2GRAY)
    
    # Apply Otsu's method to make it binary
    _, reference_im_binary = cv2.threshold(reference_im_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Find the contours in the binary reference image
    contours, _ = cv2.findContours(reference_im_binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Find the largest contour (petri dish)
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Find the bounding rectangle
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    # Dictionary to store processed images
    cropped_images = {}
    
    # Iterate over the images in the specified folder
    for image_filename in os.listdir(images_folder):
        image_path = os.path.join(images_folder, image_filename)
        im = cv2.imread(image_path, cv2.IMREAD_COLOR)
        # Crop the image using the reference coordinates
        cropped_images[image_filename] = im[y:y + h, x:x + w]
    
    return cropped_images, (x, y, w, h)

def padder(image, patch_size=256):
    """
    Pad an image to make its dimensions divisible by a specified patch size.

    Parameters:
        image (numpy.ndarray): The input image to be padded.
        patch_size (int): The desired patch size for the dimensions.

    Returns:
        tuple: Contains:
            - numpy.ndarray: The padded image
            - tuple: (left_padding, top_padding) values
    """
    h, w = image.shape[0], image.shape[1]
    
    # Calculate padding needed
    height_padding = ((h // patch_size) + 1) * patch_size - h
    width_padding = ((w // patch_size) + 1) * patch_size - w
    
    # Calculate padding for each side
    top_padding = int(height_padding / 2)
    bottom_padding = height_padding - top_padding
    left_padding = int(width_padding / 2)
    right_padding = width_padding - left_padding
    
    # Pad the image
    padded_image = cv2.copyMakeBorder(
        image, 
        top_padding, bottom_padding, 
        left_padding, right_padding, 
        cv2.BORDER_CONSTANT, 
        value=[0, 0, 0]
    )
    
    return padded_image, (left_padding, top_padding)

def create_prediction(model, image, patch_size=256):
    """
    Process a single image using a pre-trained model to create a prediction mask.

    Parameters:
        model: The pre-trained model
        image (numpy.ndarray): Input image
        patch_size (int): The patch size used for processing

    Returns:
        numpy.ndarray: The predicted mask
    """
    # Convert to grayscale if it's a color image
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Pad the image and get padding values
    padded_image, _ = padder(image, patch_size)
    
    # Patchify the image
    patches = patchify(padded_image, (patch_size, patch_size), step=patch_size)
    
    # Get dimensions for reshaping
    i, j = patches.shape[0], patches.shape[1]
    
    # Reshape patches for model prediction
    patches = patches.reshape(-1, patch_size, patch_size, 1)
    
    # Predict using the model
    preds = model.predict(patches / 255, verbose=0)
    
    # Reshape predictions
    preds = preds.reshape(i, j, patch_size, patch_size)
    
    # Unpatchify to get the full predicted mask
    predicted_mask = unpatchify(preds, (padded_image.shape[0], padded_image.shape[1]))
    
    # Apply threshold and convert to uint8
    predicted_mask = (predicted_mask > 0.5).astype(np.uint8)
    
    return predicted_mask*255

