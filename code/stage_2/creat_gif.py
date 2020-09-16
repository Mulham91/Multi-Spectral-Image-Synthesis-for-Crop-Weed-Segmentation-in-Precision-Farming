from ops import *
from help import *
import time
from tensorflow.contrib.data import prefetch_to_device, shuffle_and_repeat, map_and_batch
import numpy as np
from vgg19_keras import VGGLoss
from glob import glob
import sys
sys.path.append('../')
from pytorchMetrics import *
from utils import *
import tqdm
model_name = 'SPADE_load_test'
os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
dataset_path='/'
img_ch=4
img_height=256
img_width=256
ssegmap_ch=1
augment_flag=False
dataset_path
img_class = Image_data(img_height, img_width, img_ch, ssegmap_ch, dataset_path, augment_flag)
img_class.preprocess()
gif_dir=''
n_scale=2
n_dis=4
TTUR=True
sn=True
gan_type='hinge'
adv_weight=1
vgg_weight=10
feature_weight=10
kl_weight=0.05
dataset_name='SugarBeets_256'
metrics_rgb = []
metrics_nir = []
checkpoint_dir='resources/model'
n_critic=1
num_upsampling_layers='more'



def model_dir():

        n_dis_ = str(n_scale) + 'multi_' + str(n_dis) + 'dis'


        if sn:
            sn_ = '_sn'
        else:
            sn_ = ''

        if TTUR :
            TTUR_ = '_TTUR'
        else :
            TTUR_ = ''


        return "{}_{}_{}_{}_{}_{}_{}_{}_{}{}{}_{}/".format(model_name, dataset_name,
                                                                   gan_type, n_dis_, n_critic,
                                                                   adv_weight, vgg_weight, feature_weight,
                                                                   kl_weight,
                                                                   sn_, TTUR_, num_upsampling_layers)


def load_dataset_list(directory, type):
    # Load the dataset
    files = glob(directory + '*.png')
    # number_files = len(files)
    # print('\nNumber of files: ', number_files)

    if type == 'mask' or type == 'nir':
        image = cv2.imread(files[0], 1)
        print("image.shape")
        print(image.shape)
        image = np.expand_dims(image, axis=3)
    else:
        image = cv2.imread(files[0], -1)

    shape = image.shape

    return files, shape

checkpoint_dir = os.path.join(checkpoint_dir, model_dir())
[metrics_rgb, metrics_nir] = load(os.path.join(checkpoint_dir, 'metrics.pkl'))

rgb_dataset, _ = load_dataset_list(img_class.img_test_dataset_path, type='rgb')
nir_dataset, _ = load_dataset_list(img_class.nir_test_dataset_path, type='nir')
create_gif(gif_dir, metrics_rgb, rgb_dataset, type='rgb')
#create_gif(gif_dir, metrics_nir, nir_dataset, type='nir')





