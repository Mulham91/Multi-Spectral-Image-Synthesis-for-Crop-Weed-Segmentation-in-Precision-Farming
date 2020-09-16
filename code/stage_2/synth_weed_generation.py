#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 12:29:09 2020

@author: mulham
"""


import os
import cv2
import glob
import numpy as np

from argparse import ArgumentParser
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


kernel = np.ones((5,5), np.uint8)
color = (255, 255, 255)
dim=128

finlenum=[ 'CKA_160523'] 
path_to_outPutFile="" 
def check_countor(contour_name):
#    print(contour_name)
    with open(path_to_outPutFile) as f:
        content = f.readlines()
#print(content)
    for line in  content  :
        name=line.split('/')[-1].split(".")[0]
        if(name==contour_name):
            print( line)
            return True
    return False
 
def get_contours(mask,mask_name):
    imgray= cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY)   
    img_dilation = cv2.dilate(imgray, kernel, iterations=1)       
    ret,thresh_class1 = cv2.threshold(img_dilation,0.5,255,0)
    #ret,thresh_class1 = cv2.threshold(imgray,0.5,255,0)       
    blur_class1 = cv2.blur(thresh_class1,(9,9))
    contours_class1, hierarchy = cv2.findContours(blur_class1,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    return contours_class1, img_dilation
 
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
    for num in finlenum:
        print ("in the loop")
        nir_path="/"
        rgb_path="/"
        output_path="/"
        color_mask_path="/"
        mask_path="/"
    
        
        
      
        rgb_name_list = glob( rgb_path + "*.jpg"  ) + glob( rgb_path + "*.png"  ) +  glob( rgb_path + "*.jpeg"  )
        rgb_name_list.sort()
        nir_name_list = glob( nir_path + "*.jpg"  ) + glob( nir_path + "*.png"  ) +  glob( nir_path + "*.jpeg"  )
        nir_name_list.sort()
        mask_name_list =glob( mask_path + "*.jpg"  ) +glob( mask_path + "*.png"  ) +  glob( mask_path + "*.jpeg"  )
        mask_name_list.sort()
        rgb_mask_name_list = glob( color_mask_path + "*.jpg"  ) + glob( color_mask_path + "*.png"  ) +  glob( color_mask_path + "*.jpeg"  )
        rgb_mask_name_list.sort()
        count=0
        print (count)
        print ("rgb %d" %len(rgb_name_list))
        print (len(nir_name_list))
        print (len(mask_name_list))
     
        for mask_image , rgb_image, nir_image, color_mask_image in zip(mask_name_list,rgb_name_list,nir_name_list,rgb_mask_name_list):
            count+=1
            assert(  mask_image.split('/')[-1].split(".")[0] ==  rgb_image.split('/')[-1].split(".")[0]==nir_image.split('/')[-1].split(".")[0])
            rgb = cv2.imread(rgb_image)
            mask = cv2.imread(mask_image)      
            nir = cv2.imread(nir_image)
            maskRgb = cv2.imread(color_mask_image) 

            orininal_h = rgb.shape[0]
            orininal_w = rgb.shape[1]
            outName  = rgb_image.replace( rgb_path , output_path )
    #        outName_not_gan_rgb  = rgb_image.replace( rgb_path , output_rgb_real_path )

            rgb_copy =rgb.copy()
            mask_name=mask_image.split('/')[-1].split(".")[0]
            # imgray= cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY)   
            # img_dilation = cv2.dilate(imgray, kernel, iterations=1)       
            # ret,thresh_class1 = cv2.threshold(img_dilation,0.5,255,0)
            # blur_class1 = cv2.blur(thresh_class1,(9,9))
            # contours_class1, hierarchy = cv2.findContours(blur_class1,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    
            contours_class1,dilated_img=get_contours(mask,mask_name)
            
            


            print(dilated_img.shape)
            maskRed = maskRgb[:, :, 2]  # Get only red channel
            maskGreen = maskRgb[:, :, 1]  # Get only green channel
            Mask = np.zeros(shape=(rgb.shape[0],rgb.shape[1]), dtype="uint8")
            new_mask=cv2.merge((Mask,maskGreen,dilated_img))
              
            new_rgb=rgb.copy()
            new_nir=nir.copy()
            
            gan_contor=[]
            not_gan_contor=[]
            c=1
            for i in range(len(contours_class1)):
                name=mask_image.split('/')[-1].split(".")[0]
                name=name[:-2]
                contour_name=name+str(i)
                if(check_countor(contour_name)):
                    gan_contor.append(contours_class1[i])
                    c=+1
                else:
                    not_gan_contor.append(contours_class1[i])
                    
            if (len(gan_contor)!=0):
                pose_size_gan=[None]*len(gan_contor)
                contours_poly = [None]*len(gan_contor)
                boundRect = [None]*len(gan_contor)
                for x in range(4):
                    outName_new_rgb=output_path+"Synthetic_rgb/"+mask_image.split('/')[-1].split(".")[0]+"_" + '_'+str(x)+".png"
                    outName_new_nir=output_path+"Synthetic_nir/"+mask_image.split('/')[-1].split(".")[0]+"_" +'_'+str(x)+".png"
                    outName_new_mask=output_path+"color_mask/"+mask_image.split('/')[-1].split(".")[0]+"_" +'_'+str(x)+".png"

                                             
                    for i in range(len(gan_contor)):
                        
                        contours_poly[i] = cv2.approxPolyDP(gan_contor[i], 3, True)
                        boundRect[i] = cv2.boundingRect(gan_contor[i])
                        x=boundRect[i][0]
                        y=boundRect[i][1]
                        w=boundRect[i][2]
                        h=boundRect[i][3]
                        pose_size_gan[i]=(x,y,w,h)
                        cropped_mask = mask[boundRect[i][1]:boundRect[i][1]+boundRect[i][3]+10, boundRect[i][0]:boundRect[i][0]+boundRect[i][2]+10]
                        cropped_rgb = rgb[boundRect[i][1]:boundRect[i][1]+boundRect[i][3]+10, boundRect[i][0]:boundRect[i][0]+boundRect[i][2]+10]
                        cropped_nir = nir[boundRect[i][1]:boundRect[i][1]+boundRect[i][3]+10, boundRect[i][0]:boundRect[i][0]+boundRect[i][2]+10]
        
        # resize mask (128)
                        cropMaskResized = cv2.resize(cropped_mask, (dim, dim), interpolation=cv2.INTER_NEAREST)
        #generate synth weed & rgb
    
                        b1,g1,r1 = cv2.split(cropMaskResized)
    
                        cropped_mask_gray= cv2.cvtColor(cropMaskResized,cv2.COLOR_BGR2GRAY)
                 #    cropped_mask_gray_dilation               
                        c_m_d = cv2.dilate(cropped_mask_gray, kernel, iterations=1)
                        c_m_d = cv2.blur(c_m_d,(9,9))

                        c_m_d_3c=cv2.merge((c_m_d,c_m_d,c_m_d))
                        c_m_d_3c=cv2.resize(c_m_d_3c, (w,h),interpolation=cv2.INTER_NEAREST)
 
                        synthetic_rgb, synthetic_nir = gan.generate_sample(c_m_d)
                       
                        # outName_synthetic_rgb=output_path+"synthetic_rgb"+str(i)+'_'+str(x)+".png"

                        synthetic_rgb = cv2.cvtColor(synthetic_rgb, cv2.COLOR_BGR2RGB)
                        synthetic_nir = cv2.cvtColor(synthetic_nir, cv2.COLOR_BGR2RGB)
                      
                        synthetic_rgb_resized=cv2.resize(synthetic_rgb, (w,h),interpolation=cv2.INTER_NEAREST)
                        synthetic_nir_resized=cv2.resize(synthetic_nir, (w,h),interpolation=cv2.INTER_NEAREST)

                        # cv2.imwrite(outName_synthetic_rgb, synthetic_rgb_resized)

                        copy_cropped_rgb=cropped_rgb.copy()
                        copy_cropped_nir=cropped_nir.copy()
                        
                        copy_cropped_rgb[np.where((c_m_d_3c==[255,255,255]).all(axis=2))] = synthetic_rgb_resized[np.where((c_m_d_3c==[255,255,255]).all(axis=2))]
                        copy_cropped_nir[np.where((c_m_d_3c==[255,255,255]).all(axis=2))] = synthetic_nir_resized[np.where((c_m_d_3c==[255,255,255]).all(axis=2))]
                        
                        
                        # outName1=output_path+"copy_cropped_rgb"+str(i)+'_'+str(x)+".png"
                        # outName2=output_path+"cropped_nir"+str(i)+'_'+str(x)+".png"
                        # outName3=output_path+"cropped_mask"+str(i)+'_'+str(x)+".png"
                        # outName4=output_path+"cropped_rgb"+str(i)+'_'+str(x)+".png"
                        new_rgb[boundRect[i][1]:boundRect[i][1]+boundRect[i][3]+10, boundRect[i][0]:boundRect[i][0]+boundRect[i][2]+10]=copy_cropped_rgb[:,:]
                        new_nir[boundRect[i][1]:boundRect[i][1]+boundRect[i][3]+10, boundRect[i][0]:boundRect[i][0]+boundRect[i][2]+10]=copy_cropped_nir[:,:]
    
                        # cv2.imwrite(outName1, copy_cropped_rgb)
                        # cv2.imwrite(outName2, cropped_nir)
                        # cv2.imwrite(outName3, cropped_mask)
                        # cv2.imwrite(outName4, cropped_rgb)
                    cv2.imwrite(outName_new_rgb, new_rgb)
                    cv2.imwrite(outName_new_nir, new_nir) 
                    cv2.imwrite(outName_new_mask, new_mask)    


    #                cv2.rectangle(rgb, (int(boundRect[i][0]), int(boundRect[i][1])), (int(boundRect[i][0]+boundRect[i][2]), int(boundRect[i][1]+boundRect[i][3])), color, 2)

    
            elif (len(not_gan_contor)!=0):
                pose_size_not_gan=[None]*len(not_gan_contor)
            # cv2.imwrite(outName, new_rgb)
                
                #cv2.imwrite(outName_not_gan_rgb, rgb)
                
    #    print(count)
     
