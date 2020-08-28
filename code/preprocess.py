import os
import yaml
import cv2
import numpy as np
from argparse import ArgumentParser
from glob import glob
import math
import shutil
import random
from utils import *

import matplotlib
matplotlib.use("WebAgg")
import matplotlib.pyplot as plt

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default='../../plants_dataset/Bonn 2016/', help="Dataset path")
    parser.add_argument("--annotation_path", type=str, default='../../sugar_beet_annotation/', help="Annotation path")
    parser.add_argument("--output_path", type=str, default='../../plants_dataset/SugarBeets_256/', help="Output path")
    parser.add_argument("--dimension", type=int, default=256, help="Image dimension")
    parser.add_argument("--background", type=str2bool, default=False, help="Keep (true) or remove (false) background")
    parser.add_argument("--blur", type=str2bool, default=True, help="Remove background with blur")

    return parser.parse_args()

# Flip image
def flip_image(img, mode = 1):
    # Horizontal = 1
    # Vertical = 0
    # Both = -1
    return cv2.flip(img, mode)

# Rotate image
def rotate_image(img, angle, dim):
    M = cv2.getRotationMatrix2D((dim/2,dim/2), angle, 1)
    return cv2.warpAffine(img,M,(dim,dim), borderMode=1)

# Return list of augmented images given one single image
def augment_image(img, dim):
    flip_list = [-1, 0, 1]
    rotation_list = list(range(0, 360, 90))
    images = []

    for flip in flip_list:
        flip_img = flip_image(img, flip)

        for rotation in rotation_list:
            rotation_img = rotate_image(flip_img, rotation, dim)

            images.append(rotation_img)

    return images

# Calculate maximum radius in pixel given contour and center stem
def find_max_radius(contours, stem_x, stem_y):
    dist = 0
    for c in contours:
        for point in c:
            point = point[0]
            x = point[0] - stem_x
            y = point[1] - stem_y
            new_dist = math.ceil(math.sqrt(math.pow(x, 2) + math.pow(y, 2)))
            if new_dist > dist:
                dist = new_dist

    return dist

def get_alignment_parameters(img2, img1):

    sift = cv2.xfeatures2d.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    # Apply ratio test
    good = []
    for m in matches:
        if m[0].distance < 0.5 * m[1].distance:
            good.append(m)
    matches = np.asarray(good)

    if len(matches[:, 0]) >= 4:
        src = np.float32([kp1[m.queryIdx].pt for m in matches[:, 0]]).reshape(-1, 1, 2)
        dst = np.float32([kp2[m.trainIdx].pt for m in matches[:, 0]]).reshape(-1, 1, 2)
        H, masked = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        # print H
    else:
        print("\t\tCan't find enough keypoints.")
        H = None

    return H

def generate_dataset(path, output_path, annotation_path, dim = 256, background = True, blur = True):

    annotationsPath = os.path.join(annotation_path, 'yamls/')
    nirImagesPath = 'images/nir/'
    rgbImagesPath = 'images/rgb/'
    maskNirPath = os.path.join(annotation_path, 'masks/iMap/')
    maskRgbPath = os.path.join(annotation_path, 'masks/color/')

    imageNumber = 0

    # Get folders
    folders = glob(path + '/*/')
    # folders = ['../../plants-dataset/Bonn 2016/CKA_160523/']
    print('Number of folders:', len(folders))
    radius_list = []

    for i, folder in enumerate(folders):
        # Get files
        files = os.listdir(folder + rgbImagesPath)
        # files = ['bonirob_2016-05-23-10-57-33_4_frame157.yaml']
        number_files = len(files)
        print('\nFolder %d/%d: %s' %(i+1,len(folders),folder))
        print('\tNumber of files:', number_files)

        for j, file in enumerate(files):

            yaml_file = annotationsPath + file.split('.')[0] + '.yaml'

            print('\tFile %d/%d: %s' % (j + 1, len(files), yaml_file))

            if not os.path.isfile(yaml_file):
                print('\t\tError: YAML does not exist')
                continue

            with open(yaml_file, 'r') as stream:
                # Image name
                imageName = os.path.splitext(file)[0]

                # Open images
                rgbimg = cv2.imread(folder + rgbImagesPath + imageName + '.png', cv2.IMREAD_COLOR)
                nirimg = cv2.imread(folder + nirImagesPath + imageName + '.png', cv2.IMREAD_GRAYSCALE)
                maskNir = cv2.imread(maskNirPath + imageName + '.png', cv2.IMREAD_GRAYSCALE)
                maskRgb = cv2.imread(maskRgbPath + imageName + '.png', cv2.IMREAD_COLOR)

                if rgbimg is None or nirimg is None or maskNir is None or maskRgb is None:
                    print('\t\tError: Image does not exist')
                    continue
                maskRgb = maskRgb[:, :,1]  # Get only green channel

                # Image shape
                shape = rgbimg.shape

                black_background = np.zeros(shape=(shape[0],shape[1],4), dtype="uint8")

                # Get content from yaml file
                content = yaml.safe_load(stream)

                # For each
                try:
                    field = content["annotation"]
                except:
                    print('\t\tError: Empty Yaml')
                    continue

                # Undistort images
                flag, nirimg, maskNir = align_images(rgbimg, nirimg, maskNir)

                if flag:
                    continue

                for ann in field:
                    if ann['type'] == 'SugarBeets':

                        # Contours
                        x = ann['contours'][0]['x']
                        y = ann['contours'][0]['y']

                        # Get stem
                        stem_x = ann['stem']['x']
                        stem_y = ann['stem']['y']

                        # Plant id
                        id = ann['plant_id']

                        # Only consider if image is inside picture
                        if (stem_y > 0 and stem_x > 0):

                            # Contour mask (roughy position of the plant)
                            mask = np.zeros(shape=(rgbimg.shape[0], rgbimg.shape[1]), dtype="uint8")
                            cv2.drawContours(mask, [np.array(list(zip(x, y)), dtype=np.int32)], -1, (255, 255, 255), -1)

                            # Bitwise with RGB mask and most extreme points along the contour
                            bitRgb = cv2.bitwise_and(maskRgb, maskRgb, mask=mask)
                            _, contoursRgb, _ = cv2.findContours(bitRgb, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
                            if not contoursRgb:
                                continue

                            # Bitwise with NIR mask and most extreme points along the contour
                            bitNir = cv2.bitwise_and(maskNir, maskNir, mask=mask)
                            _, contoursNir, _ = cv2.findContours(bitNir, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

                            # Final mask
                            finalMask = np.zeros(shape=(shape[0], shape[1]), dtype="uint8")
                            cv2.drawContours(finalMask, contoursRgb, -1, (255, 255, 255), -1)
                            cv2.drawContours(finalMask, contoursNir, -1, (255, 255, 255), -1)

                            # Final mask with blur
                            finalBlur = cv2.blur(finalMask, (5, 5))

                            # Find maximum radius of the plant
                            ret, thresh = cv2.threshold(finalMask, 127, 255, 0)
                            im2, contours, hierarchy = cv2.findContours(thresh, 1, 2)
                            radius = find_max_radius(contours, stem_x, stem_y)
                            radius_list.append(radius)
                            radius = int(radius * 1.1)

                            # Final image
                            if not background:      # Remove background
                                if blur:
                                    # finalRgb = cv2.bitwise_and(rgbimg, rgbimg, mask=finalBlur)
                                    # finalNir = cv2.bitwise_and(nirimg, nirimg, mask=finalBlur)
                                    final = np.concatenate((rgbimg, np.expand_dims(nirimg, axis=2)), axis=2)
                                    final = blend_with_mask_matrix(final, black_background, finalBlur)
                                    finalRgb = final[:,:,0:3]
                                    finalNir = final[:,:,3]
                                else:
                                    finalRgb = cv2.bitwise_and(rgbimg, rgbimg, mask=finalMask)
                                    finalNir = cv2.bitwise_and(nirimg, nirimg, mask=finalMask)
                            else:
                                finalRgb = rgbimg
                                finalNir = nirimg

                            right = stem_x + radius
                            left = stem_x - radius
                            top = stem_y + radius
                            bot = stem_y - radius

                            if bot > 0 and top < shape[0] and left > 0 and right < shape[1] and radius >= dim/2:# and radius < dim:
                                # Crop images
                                cropRgb = finalRgb[bot:top, left:right, :]
                                cropNir = finalNir[bot:top, left:right]
                                cropMask = finalMask[bot:top, left:right]
                                cropBlur = finalBlur[bot:top, left:right]

                                # Resize image
                                cropRgb = cv2.resize(cropRgb, (dim, dim), interpolation=cv2.INTER_AREA)
                                cropNir = cv2.resize(cropNir, (dim, dim), interpolation=cv2.INTER_AREA)
                                cropMask = cv2.resize(cropMask, (dim, dim), interpolation=cv2.INTER_NEAREST)
                                cropBlur = cv2.resize(cropBlur, (dim, dim), interpolation=cv2.INTER_NEAREST)

                                # Augment images
                                cropRgb_ = augment_image(cropRgb, dim)
                                cropNir_ = augment_image(cropNir, dim)
                                cropMask_ = augment_image(cropMask, dim)
                                cropBlur_ = augment_image(cropBlur, dim)

                                # Write image
                                for k in range(len(cropMask_)):
                                    cv2.imwrite(output_path + 'train/rgb/' + str(imageNumber) + '_' + str(k) + '.png', cropRgb_[k])
                                    cv2.imwrite(output_path + 'train/nir/' + str(imageNumber) + '_' + str(k) + '.png', cropNir_[k])
                                    cv2.imwrite(output_path + 'train/mask/' + str(imageNumber) + '_' + str(k) + '.png', cropMask_[k])
                                    cv2.imwrite(output_path + 'train/blur/' + str(imageNumber) + '_' + str(k) + '.png', cropBlur_[k])
                                imageNumber += 1

if __name__ == '__main__':
    args = parse_args()

    folders = ['train/', 'test/']
    subfolers = ['mask/', 'rgb/', 'nir/', 'blur/']

    output_path = args.output_path
    # Create folders if do not exist
    if os.path.exists(output_path):
        print('\nFolder', output_path, 'already exist, delete it before continue!\n')
    else:
        print('\nCreating folders!\n')

        os.makedirs(output_path)
        for f in folders:
            for s in subfolers:
                os.makedirs(output_path + f + s)

        # Generate data
        generate_dataset(path=args.dataset_path, output_path=output_path, annotation_path=args.annotation_path, dim=args.dimension, background=args.background, blur=args.blur)

    # Split train and test files
    files = os.listdir(output_path + folders[0] + 'mask/')
    cut_files = random.sample(files, 200)

    for c in cut_files:
                for s in subfolers:
                    shutil.move(output_path + folders[0] + s + c, output_path + folders[1] + s + c)
