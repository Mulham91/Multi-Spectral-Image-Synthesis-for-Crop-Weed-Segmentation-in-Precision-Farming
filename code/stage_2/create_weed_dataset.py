import os
import yaml
import cv2
import numpy as np
from argparse import ArgumentParser
from glob import glob
import math
import shutil
import random
from main import *

import sys
sys.path.append('../')
from utils import *
def m_args():
    desc = "Tensorflow implementation of SPADE"
    parser =  ArgumentParser(description=desc)
    parser.add_argument('--phase', type=str, default='train', choices=('train', 'guide', 'random'), help='phase name')
    parser.add_argument('--dataset_name', type=str, default='SugarBeets_256', help='Dataset name')
    parser.add_argument('--dataset_path', type=str, default='/', help='Dataset path')

    parser.add_argument('--epoch', type=int, default=300, help='The number of epochs to run')
    parser.add_argument('--iteration', type=int, default=1, help='The number of training iterations')
    # The total number of iterations is [epoch * iteration]

    parser.add_argument('--batch_size', type=int, default=1, help='The size of batch size')
    parser.add_argument('--print_freq', type=int, default=1, help='The number of image_print_freq')
    parser.add_argument('--save_freq', type=int, default=1, help='The number of ckpt_save_freq')
    parser.add_argument('--decay_flag', type=str2bool, default=True, help='The decay_flag')
    parser.add_argument('--decay_epoch', type=int, default=200, help='decay epoch')

    parser.add_argument('--lr', type=float, default=0.0002, help='The learning rate')
    parser.add_argument('--TTUR', type=str2bool, default=True, help='Use TTUR training scheme')

    parser.add_argument('--num_style', type=int, default=3, help='number of styles to sample')
    parser.add_argument('--guide_img', type=str, default='resources/guide', help='Style guided image translation')

    parser.add_argument('--ld', type=float, default=10.0, help='The gradient penalty lambda')
    parser.add_argument('--adv_weight', type=int, default=1, help='Weight about GAN')
    parser.add_argument('--vgg_weight', type=int, default=10, help='Weight about perceptual loss')
    parser.add_argument('--feature_weight', type=int, default=10, help='Weight about discriminator feature matching loss')
    parser.add_argument('--kl_weight', type=float, default=0.05, help='Weight about kl-divergence')

    parser.add_argument('--gan_type', type=str, default='hinge', help='gan / lsgan / hinge / wgan-gp / wgan-lp / dragan')
    parser.add_argument('--ch', type=int, default=64, help='base channel number per layer')
    # parser.add_argument('--ch', type=int, default=32, help='base channel number per layer')


    parser.add_argument('--n_dis', type=int, default=4, help='The number of discriminator layer')
    parser.add_argument('--n_scale', type=int, default=2, help='number of scales')
    parser.add_argument('--n_critic', type=int, default=1, help='The number of critic')
    parser.add_argument('--sn', type=str2bool, default=True, help='using spectral norm')

    parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
    parser.add_argument('--beta2', type=float, default=0.999, help='momentum term of adam')

    parser.add_argument('--num_upsampling_layers', type=str, default='more',
                        choices=('normal', 'more', 'most'),
                        help="If 'more', adds upsampling layer between the two middle resnet blocks. "
                             "If 'most', also add one more upsampling + resnet layer at the end of the generator")

    parser.add_argument('--img_height', type=int, default=256, help='The height size of image')
    parser.add_argument('--img_width', type=int, default=256, help='The width size of image ')
    parser.add_argument('--img_ch', type=int, default=4, help='The size of image channel')
    parser.add_argument('--segmap_ch', type=int, default=1, help='The size of segmap channel')
    parser.add_argument('--augment_flag', type=str2bool, default=False, help='Image augmentation use or not')

    parser.add_argument('--checkpoint_dir', type=str, default='resources/model',
                        help='Directory name to save the checkpoints')
    parser.add_argument('--samples_dir', type=str, default='resources/samples',
                        help='Directory name to save the generated images')
    parser.add_argument('--log_dir', type=str, default='resources/logs',
                        help='Directory name to save training logs')
    parser.add_argument('--gif_dir', type=str, default='resources/gif',
                        help='Directory name to save the samples on training')
    parser.add_argument('--seed_dir', type=str, default='resources/seed',
                        help='Directory name of the seed files')
    parser.add_argument('--result_dir', type=str, default='resources/results',
                        help='Directory name to save the generated images')

    return  parser.parse_args() 
#def parseArgs():
#    parser = ArgumentParser()
 #   parser.add_argument("--dataset_path", type=str, default='../../../plants_dataset/Bonn 2016/', help="Dataset path")
 #   parser.add_argument("--annotation_path", type=str, default='../../../sugar_beet_annotation/', help="Annotation path")
 #  parser.add_argument("--output_path", type=str, default='../../../plants_dataset/Segmentation/', help="Output path")
  #  parser.add_argument("--background", type=str2bool, default=False, help="Keep (true) or remove (false) background")
   # parser.add_argument("--blur", type=str2bool, default=True, help="Remove background with blur")

  #  return parser.parse_args()

# Flip image
def flip_image(img, mode = 1):
    # Horizontal = 1
    # Vertical = 0
    # Both = -1
    return cv2.flip(img, mode)

# Rotate image
def rotate_image(img, angle, shape):
    height = shape[0]
    width = shape[1]
    channels = shape[2]
    M = cv2.getRotationMatrix2D((width/2,height/2), angle, 1)
    return cv2.warpAffine(img,M,(width,height), borderMode=1)

# Return list of augmented images given one single image
def augment_image(img, shape):
#    flip_list = [-1, 0, 1]
    flip_list = [ 1]

#    images = [img]
    images = []
    for flip in flip_list:
        flip_img = flip_image(img, flip)

        images.append(flip_img)

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

# Calculate stem if crop is outside image and max and min values for coordinates
def calculateStem(contours, stem_x, stem_y):

    m_x = [10e3, -10e3]
    m_y = [10e3, -10e3]

    for c in contours:
        for point in c:
            point = point[0]

            m_x = [min(m_x[0], point[0]), max(m_x[1], point[0])]
            m_y = [min(m_y[0], point[1]), max(m_y[1], point[1])]

    # if stem_x < 0:
    #     stem_x = int(m_x[0] + (m_x[1] - m_x[0])/2)
    #
    # if stem_y < 0:
    #     stem_y = int(m_y[0] + (m_y[1] - m_y[0]) / 2)

    return stem_x, stem_y, m_x, m_y

def generate_dataset(path, output_path, annotation_path, background, blur, type="SugarBeets"):

    annotationsPath = os.path.join(annotation_path, 'yamls/')
    nirImagesPath = 'images/nir/'
    rgbImagesPath = 'images/rgb/'
    maskNirPath = os.path.join(annotation_path, 'masks/iMap/')
    maskRgbPath = os.path.join(annotation_path, 'masks/color/')
    print(annotationsPath)
    print(nirImagesPath)
    print(rgbImagesPath)
    print(maskNirPath)
    print(maskRgbPath)
    imageNumber = 0

    dim = 256

    # Get folders
    folders = glob(path + '/*/')
    print('Number of folders:', len(folders))
    complete_radius_list = []
    cutted_images = 0


    args = m_args()
    if args is None:
      exit()

    # open session
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        gan = spade(sess, args)

        # build graph
        gan.build_model()

        # Load model
        gan.load_model()
        print('*******************annotationsPath**********************')
        print(annotationsPath)
        print(nirImagesPath)
        print(rgbImagesPath)
        print(maskNirPath)
        print(maskRgbPath)
        for i, folder in enumerate(folders):
           	
 	
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
                    maskRgb = cv2.imread(maskRgbPath + imageName + '.png', cv2.IMREAD_COLOR)
                    maskNir = cv2.imread(maskNirPath + imageName + '.png', cv2.IMREAD_GRAYSCALE)

                    if rgbimg is None or nirimg is None or maskNir is None or maskRgb is None:
                        print('\t\tError: Image does not exist')
                        continue
                    maskRed = maskRgb[:, :, 2]  # Get only red channel
                    maskGreen = maskRgb[:, :, 1]  # Get only green channel

                    shape = rgbimg.shape

                    # Get content from yaml file
 #                   content = yaml.safe_load(stream)

                    # For each
#                    try:
#                       field = content["annotation"]
 #                   except:
#                        print('\t\tError: Empty Yaml')
 #                       continue

                    # Undistort images
                  #  flag, nirimg, maskNir = align_images(rgbimg, nirimg, maskNir)

                  #  if flag:
                  #      continue

                    # Blank mask
                    maskCrop = np.zeros(shape=(rgbimg.shape[0], rgbimg.shape[1]), dtype="uint8")

                    # Radius list
                    radius_list = []
                    i=0
                    for ann in field:
                        if ann['type'] == type:

                            # Contours
                            x = ann['contours'][0]['x']
                            y = ann['contours'][0]['y']

                            # Get stem
                            stem_x = ann['stem']['x']
                            stem_y = ann['stem']['y']

                            # Draw plant on mask
                            cv2.drawContours(maskCrop, [np.array(list(zip(x, y)), dtype=np.int32)], -1, (255, 255, 255), -1)
                            cv2.imwrite(output_path + 'train/original/image_'  +'Draw plant on mask'+ '_' + str(i) + '.png',maskCrop )
                            # # Only consider if image is inside picture
                            if (stem_y > 0 and stem_x > 0):

                                # Contour mask (roughy position of the plant)
                                mask = np.zeros(shape=(rgbimg.shape[0], rgbimg.shape[1]), dtype="uint8")
                                cv2.drawContours(mask, [np.array(list(zip(x, y)), dtype=np.int32)], -1, (255, 255, 255), -1)

                                # Bitwise with RGB mask and most extreme points along the contour
                                bitRgb = cv2.bitwise_and(maskGreen, maskGreen, mask=mask)
                                contoursRgb, _ = cv2.findContours(bitRgb, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
                                if not contoursRgb:
                                    continue

                                # Bitwise with NIR mask and most extreme points along the contour
                                bitNir = cv2.bitwise_and(maskNir, maskNir, mask=mask)
                                contoursNir, _ = cv2.findContours(bitNir, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

                                # Final mask
                                finalMask = np.zeros(shape=(shape[0], shape[1]), dtype="uint8")
                                cv2.drawContours(finalMask, contoursRgb, -1, (255, 255, 255), -1)
                                cv2.drawContours(finalMask, contoursNir, -1, (255, 255, 255), -1)

                                # Find maximum radius of the plant
                                ret, thresh = cv2.threshold(finalMask, 127, 255, 0)
  #                                im2, contours, hierarchy = cv2.findContours(thresh, 1, 2)
                                contours, hierarchy = cv2.findContours(thresh, 1, 2)
                                # Calculate stem if not given
                                stem_x, stem_y, m_x, m_y = calculateStem(contours, stem_x, stem_y)

                                radius = find_max_radius(contours, stem_x, stem_y)
                                radius = int(radius * 1.1)

                                right = m_x[1]
                                left = m_x[0]
                                top = m_y[1]
                                bot = m_y[0]

                                complete_radius_list.append(radius)

                                # Crop images
                                cropMask = np.zeros(shape=(2*radius, 2*radius), dtype="uint8")
                                cropMask[radius-(stem_y-bot):radius+(top-stem_y), radius-(stem_x-left):radius+(right-stem_x)]=finalMask[bot:top, left:right]
                                i=i+1
                                # Resize mask
                                cropMaskResized = cv2.resize(cropMask, (dim, dim), interpolation=cv2.INTER_NEAREST)
                                cv2.imwrite(output_path + 'train/original/image_' + str(imageNumber)+'_cropMaskResized_' + '_' + str(i) + '.png',cropMaskResized)
                                cv2.imwrite(output_path + 'train/original/image_' + str(imageNumber) +'_maskcrop_'+ '_' + str(i) + '.png',maskCrop)

                                radius_list.append([m_x, m_y, stem_x, stem_y, radius, cropMask, cropMaskResized])
                            else:
                                cutted_images += 1


                    # Bitwise with RGB mask and most extreme points along the contour
                    maskWeed = cv2.bitwise_not(maskCrop) # not crop

                    crop = cv2.bitwise_and(maskGreen, maskGreen, mask=maskCrop)
                    weed = cv2.bitwise_and(maskRed, maskRed, mask=maskWeed)

                    maskRgb = (crop/127.5 + weed/255).astype(np.uint8)
                    cv2.imwrite(output_path + 'train/original/image_' + str(imageNumber) +'_maskRgb_'+ '_' + str(i) + '.png',maskRgb)
                    cv2.imwrite(output_path + 'train/original/image_' + str(imageNumber) +'_maskWeed_'+ '_' + str(i) + '.png',maskWeed)
                    cv2.imwrite(output_path + 'train/original/image_' + str(imageNumber) +'_crop_'+ '_' + str(i) + '.png',crop)
                    cv2.imwrite(output_path + 'train/original/image_' + str(imageNumber) +'_weed_'+ '_' + str(i) + '.png',weed)

                    # Augment images
                    rgbimg_ = augment_image(rgbimg, shape)
                    nirimg_ = augment_image(nirimg, shape)
                    mask_ = augment_image(maskRgb, shape)

                    # Original images
                    for k in range(len(mask_)):

                        cv2.imwrite(output_path + 'train/original/rgb/image_' + str(imageNumber) + '_' + str(k) + '.png', rgbimg_[k])
                        cv2.imwrite(output_path + 'train/original/nir/image_' + str(imageNumber) + '_' + str(k) + '.png', nirimg_[k])
                        cv2.imwrite(output_path + 'train/original/mask/image_' + str(imageNumber) + '_' + str(k) + '.png', mask_[k])

                    # Number of folds for the original dataset compared to the original one
                    if len(radius_list) > 0:
                        for fold in range(1):
                            rgbimgCopy = rgbimg.copy()
                            nirimgCopy = nirimg.copy()
                            for [m_x, m_y, stem_x, stem_y, radius, cropMask, cropMaskResized] in radius_list:

                                right = m_x[1]
                                left = m_x[0]
                                top = m_y[1]
                                bot = m_y[0]

                                # # Generate image
                                synthetic_rgb, synthetic_nir = gan.generate_sample(cropMaskResized)
                                synthetic_rgb = cv2.cvtColor(synthetic_rgb, cv2.COLOR_BGR2RGB)
                                # Used for test only
                                # synthetic_nir = cropMaskResized
                                # synthetic_rgb = np.expand_dims(synthetic_nir, axis=2)
                                # synthetic_rgb = np.repeat(synthetic_rgb, 3, axis=2)

                                synthetic_rgb = cv2.resize(synthetic_rgb, (radius*2, radius*2), interpolation=cv2.INTER_AREA)
                                synthetic_rgb = synthetic_rgb[radius - (stem_y - bot): radius + (top - stem_y),
                                            radius - (stem_x - left): radius + (right - stem_x), :]

                                synthetic_nir = cv2.resize(synthetic_nir, (radius*2, radius*2), interpolation=cv2.INTER_AREA)
                                synthetic_nir = synthetic_nir[radius - (stem_y - bot): radius + (top - stem_y),
                                            radius - (stem_x - left): radius + (right - stem_x)]

                                if blur:
                                    cropMask = cv2.blur(cropMask, (5, 5))

                                if not background:

                                    original_rgb = rgbimgCopy[bot:top, left:right, :]
                                    original_nir = nirimgCopy[bot:top, left:right]

                                    original = np.concatenate((original_rgb, np.expand_dims(original_nir, axis=2)), axis=2)
                                    synthetic = np.concatenate((synthetic_rgb, np.expand_dims(synthetic_nir, axis=2)), axis=2)

                                    mask = cropMask[radius - (stem_y - bot): radius + (top - stem_y),
                                                    radius - (stem_x - left): radius + (right - stem_x)]

                                    blended = blend_with_mask_matrix(synthetic, original, mask)
                                    rgbimgCopy[bot:top,left:right,:] = blended[:,:,0:3]
                                    nirimgCopy[bot:top,left:right] = blended[:,:,3]

                                else:
                                    rgbimgCopy[bot:top, left:right, :] = synthetic_rgb
                                    nirimgCopy[bot:top, left:right] = synthetic_nir

                                # cv2.imshow('BEFORE', rgbimgCopy)
                                # cv2.waitKey(0)
                                # cv2.destroyAllWindows()

                            # Augment generated image
                            rgbimg_ = augment_image(rgbimgCopy, shape)
                            nirimg_ = augment_image(nirimgCopy, shape)

                            # Write image
                            for k in range(len(mask_)):
                                cv2.imwrite(output_path + 'train/synthetic/rgb/image_' + str(imageNumber) + '_' + str(fold) + '_' + str(k) + '.png', rgbimg_[k])
                                cv2.imwrite(output_path + 'train/synthetic/nir/image_' + str(imageNumber) + '_' + str(fold) + '_' + str(k) + '.png', nirimg_[k])
                                cv2.imwrite(output_path + 'train/synthetic/mask/image_' + str(imageNumber) + '_' + str(fold) + '_' + str(k) + '.png', mask_[k])

                    imageNumber += 1

    print(complete_radius_list)

    print("Cut images: ", str(cutted_images))

    above = 0
    below = 0

    for i in complete_radius_list:
        if i >= 128:
            above += 1

        else:
            below += 1

    print("Above: ", str(above))
    print("Below: ", str(below))

if __name__ == '__main__':
#    args = parseArgs()

    folders = ['train/', 'test/']
    subfolers = ['original/', 'synthetic/']
    subsubfolers = ['rgb/', 'nir/', 'mask/']

    output_path ='/' # #'../../dataset/Segmentation/'
    # output_path = '/Volumes/MAXTOR/Segmentation/'
    # Create folders if do not exist
    if False:
        print('\nFolder', output_path, 'already exist, delete it before continue!\n')
    else:
        print('\nCreating folders!\n')

        os.makedirs(output_path)

        for f in folders:
            for s in subfolers:
   
             for w in subsubfolers:
                    os.makedirs(output_path + f + s + w)

        # Generate data
        generate_dataset(path='', output_path=output_path, annotation_path='/', background=False, blur=True)

        # Split original train and test files
        # for s in subfolers:
        s = 'original/'
        files = os.listdir(output_path + folders[0] + s + subsubfolers[0])
        cut_files = random.sample(files, int(len(files)*0.2))

        for c in cut_files:
            for ss in subsubfolers:
                shutil.move(output_path + folders[0] + s + ss + c, output_path + folders[1] + s + ss + c)
