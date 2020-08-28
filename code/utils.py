from scipy import misc
import pickle
from glob import glob
import imageio
import os
import cv2
import numpy as np
from pytorchMetrics import *
import matplotlib
matplotlib.use("WebAgg")
matplotlib.rcParams['savefig.pad_inches'] = 0
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


GIF_MATRIX = 5

RGB_CAMERA_MATRIX = np.matrix([[2150.03686, 0.000000, 649.2291], [0.000000, 2150.03686, 480.680634], [0.000000, 0.000000, 1.000000]])
RGB_PROJECTION_MATRIX = np.matrix([[2150.03686, 0.000000, 649.2291, 0.000000], [0.000000, 2150.03686, 480.680634, 0.000000], [0.000000, 0.000000, 1.000000, 0.000000]])
RGB_DISTORT_COEFFICIENTS = np.matrix([0.0, 0.0, 0.0, 0.0, 0.0])

NIR_CAMERA_MATRIX = np.matrix([[2162.2948, 0.000000, 650.22019], [0.000000, 2162.2948, 481.20451], [0.000000, 0.000000, 1.000000]])
NIR_PROJECTION_MATRIX = np.matrix([[2162.2948, 0.000000, 650.22019, 0.000000], [0.000000, 2162.2948, 481.20451, 0.000000], [0.000000, 0.000000, 1.000000, 0.000000]])
NIR_DISTORT_COEFFICIENTS = np.matrix([0.0, 0.0, 0.0, 0.0, 0.0])

TRANSLATION = RGB_CAMERA_MATRIX[0:2,2] - NIR_CAMERA_MATRIX[0:2,2]

def align_images(rgbimg, nirimg, maskNir):

    shape = rgbimg.shape

    # Undistort NIR
    nirimg = cv2.undistort(nirimg, NIR_CAMERA_MATRIX, NIR_DISTORT_COEFFICIENTS, None, RGB_CAMERA_MATRIX)
    maskNir = cv2.undistort(maskNir, NIR_CAMERA_MATRIX, NIR_DISTORT_COEFFICIENTS, None, RGB_CAMERA_MATRIX)

    # Get parameter to superpose images
    try:
        H = get_alignment_parameters(cv2.cvtColor(rgbimg, cv2.COLOR_BGR2GRAY), nirimg)
    except:
        H = None

    if H is None:
        print('\t\tError: Empty H matrix')
        return 1,1,1

    # print(np.sum(np.abs(H)))
    if np.sum(np.abs(H)) > 10:
        print('\t\tError: Error in alignment')
        return 1,1,1

    nirimg = cv2.warpPerspective(nirimg, H, (shape[1], shape[0]))
    maskNir = cv2.warpPerspective(maskNir, H, (shape[1], shape[0]))

    return 0, nirimg, maskNir

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

def blend_with_mask_matrix(src1, src2, mask):
    mask = np.repeat(np.expand_dims(mask, axis=2), src1.shape[2], axis=2)
    res_channels = []
    for c in range(0, src1.shape[2]):
        a = src1[:, :, c]
        b = src2[:, :, c]
        m = mask[:, :, c]
        res = cv2.add(
            cv2.multiply(b, cv2.divide(np.full_like(m, 255) - m, 255.0, dtype=cv2.CV_32F), dtype=cv2.CV_32F),
            cv2.multiply(a, cv2.divide(m, 255.0, dtype=cv2.CV_32F), dtype=cv2.CV_32F),
           dtype=cv2.CV_8U)
        res_channels += [res]
    res = cv2.merge(res_channels)
    return res

def str2bool(x):
    return x.lower() in ("yes", "true", "t", "1")

def check_folder(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir

def imsave(image, path):
#    return misc.imsave(path, image)
    return imageio.imwrite(path, image)

# Save dictionary to file
def save(obj, name):
    with open(name, 'wb') as f:
        pickle.dump(obj, f)

# Load dictionary from file
def load(name):
    with open(name, 'rb') as f:
        return pickle.load(f)

# Convert to the range [-1, 1]
def preprocessing(x):
    x = x/127.5 - 1
    return x

# Convert to the range [0, 255]
def postprocessing(x):
    x = ((x + 1) / 2) * 255.0
    return x

def plot_gif(images, epoch, gif_dir, type):

    plt.figure(figsize=(GIF_MATRIX*3, GIF_MATRIX*3))
    gs1 = gridspec.GridSpec(GIF_MATRIX, GIF_MATRIX)
    gs1.update(wspace=0.025, hspace=0.025)
    for i in range(images.shape[0]):
        ax = plt.subplot(gs1[i])
        if type == 'mask':
            ax.imshow(images[i,:,:,0], cmap='gray', vmin=0, vmax=255, interpolation='nearest')
        elif  type == 'nir':
            ax.imshow(images[i,:,:], cmap='gray', vmin=0, vmax=255, interpolation='nearest')
        else:
            ax.imshow(images[i,:,:,:], vmin=0, vmax=255, interpolation='nearest')
        ax.axis('off')
        ax.set_aspect('equal')
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        ax.set_frame_on(False)

    name = type + "_%d.png" % epoch
    plt.savefig(os.path.join(gif_dir, name), bbox_inches='tight', pad_inches=0.025)
    plt.close()

# Create the gif given the dictionary and its size
def create_gif(images_directory, metrics, test_dataset, type, duration=20):
    files = glob(os.path.join(images_directory, type + '_*.png'))
    print("images_directory")
    print(images_directory)
    files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
    frames = []
    images = []

    size = (700, 700)
    graphs = generate_graphs(metrics, test_dataset, size, type)
    # Get gif images
    for f in files:
        img = cv2.imread(f, 1)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, size)
        images.append(img)

    # Construct graph
#    graphs = generate_graphs(metrics, test_dataset, size, type)

    for i, image in enumerate(images):
        graph = graphs[i]
        graph = cv2.cvtColor(graph, cv2.COLOR_RGB2BGR)
        # graph = graph[3:3 + size[0], 5:5 + size[1]]
        new_im = np.hstack((image, graph))
        frames.append(new_im)
    print("image.shape")
#    print(ln(frames))

    height, width, layers = frames[0].shape
    size = (width, height)

    # Repeat last frames
    # for i in range(int(len(files)*.5)):
    #     frames.append(frames[-1])

    # Calculate time between frames
    fps = len(frames)/duration

    out = cv2.VideoWriter(images_directory + 'training_' + type + '.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, size)

    for i in range(len(frames)):
        out.write(frames[i])
    out.release()

    # Create gif
    # imageio.mimsave(images_directory + 'training_' + type + '.mp4', frames, format='TIFF')#, duration=time)

def generate_graphs(metrics, test_dataset, size, type):

    # List of metrics
    # names = ['emd', 'fid', 'inception', 'knn', 'mmd', 'mode']
    # names = ['emd', 'fid', 'inception', 'mmd', 'mode']
    names = ['Inception', 'Mode', 'MMD', 'EMD', 'FID', 'KNN']

    emd = []
    fid = []
    inception = []
    knn = []
    mmd = []
    mode = []
    for m in metrics:
        emd.append(m.emd)
        fid.append(m.fid)
        inception.append(m.inception)
        knn.append(m.knn)
        mmd.append(m.mmd)
        mode.append(m.mode)

    metrics = {'emd': emd, 'fid': fid, 'inception': inception, 'knn': knn, 'mmd': mmd, 'mode': mode}
    l_emd=0 

    l_fid = 0
    l_inception =0
    l_knn =0
    l_mmd =0
    l_mode =0
    for i in range(5):
        c= 199-i
        print("C",c)
        l_emd =l_emd+emd[c]
        l_fid=l_fid+fid[c]
        l_inception=l_inception+inception[c]
        l_knn=l_knn+knn[c]
        l_mmd=l_mmd+mmd[c]
        l_mode=l_mode+mode[c]

    print("emd",emd[-1])
    print("fid",fid[-1])
    print("inception",inception[-1])
    print("knn",knn[-1])
    print("mmd",mmd[-1])
    print("mode",mode[-1])

    print("Lastemd",emd[199])
    print("Lastfid",fid[199])
    print("Lastinception",inception[199])
    print("Lastknn",knn[199])
    print("Lastmmd",mmd[199])
    print("Lastmode",mode[199])

    print("m_emd",l_emd/5)
    print("m_fid",l_fid/5)
    print("m_inception",l_inception/5)
    print("m_knn",l_knn/5)
    print("m_mmd",l_mmd/5)
    print("m_mode",l_mode/5)


    num_metrics = len(names)
    epochs = len(emd)

    frames = []

    # Calculate gold metrics
    gold_metrics = calculate_gold_metrics(test_dataset, type)

    # Graph size
    width = int(size[0] * 1.1)
    height = int(size[1] * 1.05)
    dpi = 100

    for epoch in range(epochs):

        fig, ax = plt.subplots(num_metrics, figsize=(width/dpi, height/dpi), dpi=dpi)
        fig.suptitle('Epoch: ' + str(epoch+1), x=0.11, y=.96, horizontalalignment='left', verticalalignment='top', fontsize=14)
        # fig.patch.set_visible(False)
        # fig.axes([0,0,1,1], frameon=False)
        # fig = plt.figure(figsize=(width/dpi, height/dpi), dpi=dpi, frameon=False, tight_layout=True)

        for i in range(num_metrics):
            horizontal = getattr(gold_metrics, names[i].lower())
            max_ = max(max(metrics[names[i].lower()]), horizontal)
            min_ = min(min(metrics[names[i].lower()]), horizontal)
            offset = (max_ - min_) * 0.1

            # ax = fig.add_subplot(num_metrics, 1, i + 1)
            ax[i].axhline(y=horizontal, color='r', linestyle=':')
            ax[i].set_xlim([0, epochs])
            ax[i].set_ylim([min_ - offset, max_ + offset])
            ax[i].set_ylabel(names[i])
            ax[i].yaxis.set_label_position("right")
            ax[i].plot(metrics[names[i].lower()][:epoch])

            if i != num_metrics-1:
                ax[i].axes.get_xaxis().set_visible(False)

        fig.canvas.draw()
        image = np.fromstring(fig.canvas.tostring_rgb(), dtype='uint8').reshape(height, width, 3)
        image = image[0:size[1], int(width*0.05):int(width*0.05) + size[0], :]
        image = cv2.resize(image, size, interpolation=cv2.INTER_NEAREST)
        frames.append(image)

        plt.close(fig)
    return frames

# Calculate metrics when comparing one set of real images with another
# These values are the desirable values to achieve with GAN
def calculate_gold_metrics(test_dataset, type):
    # files, _ = load_dataset_list(test_directory + 'mask/')
    if type == 'mask' or type == 'nir':
        data = load_data(test_dataset, type, repeat=True)
    else:
        data = load_data(test_dataset, type, repeat=False)

    metrics_list = []

    metrics = pytorchMetrics()

    for i in range(10):
        samples = data[np.random.choice(data.shape[0], 200)]
        real_1 = samples[:100]
        real_2 = samples[100:]

        metrics_list.append(metrics.compute_score(real_1, real_2))

    emd, mmd, knn, inception, mode, fid = np.array([(t.emd, t.mmd, t.knn, t.inception, t.mode, t.fid) for t in metrics_list]).T

    score = Score()
    score.emd = np.mean(emd)
    score.mmd = np.mean(mmd)
    score.knn = np.mean(knn)
    score.inception = np.mean(inception)
    score.mode = np.mean(mode)
    score.fid = np.mean(fid)
    print("gold Metrisc")
    print("score emd",score.emd)
    print("score fid",score.fid)
    print("score inception",score.inception)
    print("score knn",score.knn)
    print("score mmd",score.mmd)
    print("score mode",score.mode)

    return score

# Load data given a file list
# Input:
#   - files: list of files
#   - repeat: repeat third chanel
def load_data(files, type, repeat=False, scale=False):

    data = []
    for file in files:
        if type == 'mask' or type == 'nir':
            img = cv2.imread(file, 0)
        else:
            img = cv2.imread(file, -1)
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        data.append(img)

    data = np.asarray(data, dtype='uint8')

    # Rescale
    if scale:
        data = data / 127.5 - 1.

    if type == 'mask' or type == 'nir':
        data = np.expand_dims(data, axis=3)

    if repeat:
        data = np.repeat(data, 3, 3)

    return data

# Load list of files of a dictionary with image shape
def load_dataset_list(directory, type):
    # Load the dataset
    files = glob(directory + '*.png')
    # number_files = len(files)
    # print('\nNumber of files: ', number_files)

    if type == 'mask' or type == 'nir':
        image = cv2.imread(files[0], 0)
        image = np.expand_dims(image, axis=3)
    else:
        image = cv2.imread(files[0], -1)

    shape = image.shape

    return files, shape
