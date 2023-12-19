from django.db import models
import pickle
from django.contrib.auth.models import User
import torch
import logging
from torchvision.utils import draw_bounding_boxes
import torch
import torchvision
import torchvision.transforms as T
from PIL import Image

import cv2
import numpy as np

import cv2
import numpy as np

def find_cropped_position(original_image_path, cropped_image_path):
    # Load the original and cropped images
    original_image = cv2.imread(original_image_path)
    cropped_image = cv2.imread(cropped_image_path)

    # Convert the images to grayscale
    original_gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    cropped_gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)

    # Perform feature matching using ORB
    orb = cv2.ORB_create()
    keypoints_original, descriptors_original = orb.detectAndCompute(original_gray, None)
    keypoints_cropped, descriptors_cropped = orb.detectAndCompute(cropped_gray, None)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(descriptors_cropped, descriptors_original)

    # Sort and select top matches
    matches = sorted(matches, key=lambda x: x.distance)
    num_good_matches = int(len(matches) * 0.1)
    good_matches = matches[:num_good_matches]

    # Extract corresponding keypoints
    points_original = np.float32([keypoints_original[match.trainIdx].pt for match in good_matches]).reshape(-1, 1, 2)
    points_cropped = np.float32([keypoints_cropped[match.queryIdx].pt for match in good_matches]).reshape(-1, 1, 2)

    # Compute perspective transformation
    M, _ = cv2.findHomography(points_cropped, points_original, cv2.RANSAC)

    # Get corners of cropped image
    h, w = cropped_gray.shape
    corners_cropped = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)

    # Transform corners to original image coordinates
    corners_original = cv2.perspectiveTransform(corners_cropped, M)
    
    for i in corners_original:
        if i[0, 0] > 0:
            return False
        if i[0, 1] <0:
            return False
    print(corners_original)
        
                
    xs = [i[0, 0] for i in corners_original]
    ys = [i[0, 1] for i in corners_original]
    area = (max(xs) - min(xs)) * (max(ys) - min(ys))
    print(cropped_image_path, area, 200*200)
    # mx, my, _ = original_image.shape
    if area < 200*200:
        return False

    original_with_rectangle = cv2.polylines(original_image, [np.int32(corners_original)], True, (0, 255, 0), 3)

    # Display the result
    cv2.imwrite('./media/temp_res.png', original_with_rectangle)
    return True

device = torch.device("cuda")
f = open("./recognise/test")
files = [i.strip() for i in f.readlines()]
f.close()

classes = ['creatures', 'fish', 'jellyfish', 'penguin', 'puffin', 'shark', 'starfish', 'stingray']

class RecognisionModel(models.Model):
    name = models.CharField(max_length=200)
    model_file = models.FileField(upload_to='models/')
    accuracy = models.FloatField()
    is_active = models.BooleanField(default=False)

    def recognise(self, img):
        species = self.recognise_species()
        if species != False:
            return species

        # Load the trained model from disk
        with self.model_file.open(mode='rb') as f:
            model = pickle.load(f)
        img_int = torch.tensor(img*255, dtype=torch.uint8)
        with torch.no_grad():
            prediction = model([img.to(device)])
            pred = prediction[0]
        res = draw_bounding_boxes(img_int, pred['boxes'][pred['scores'] > 0.8], [classes[i] for i in pred['labels'][pred['scores'] > 0.8].tolist()], width=4).permute(1, 2, 0)
        torchvision.utils.save_image(draw_bounding_boxes(img_int,
            pred['boxes'][pred['scores'] > 0.8],
            [classes[i] for i in pred['labels'][pred['scores'] > 0.8].tolist()], width=4
        ).permute(0, 1, 2).div(255), "media/temp_res.png")
        return  None
    
    
    
    def recognise_species(self):
        for file in files:
            try:
                if find_cropped_position("./media/temp.png", f"./recognise/cropped/{file}"):
                    return " ".join(file.replace(".png", "").split("_")[:2])
            except Exception as e:
                print(e)
        return False

    def __str__(self):
        return f"RecognisionModel {self.name}"

class RecognisionResult(models.Model):
    image = models.ImageField(null=True, blank=True, upload_to="media/img")
    video = models.FileField(null=True, blank=True, upload_to="media/video")
    user = models.ForeignKey(User, on_delete=models.CASCADE)