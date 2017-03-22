'''
Created on 15 Feb 2017

@author: Dave
'''
import matplotlib.pyplot as plt
import numpy as np
from scipy.misc import toimage, imresize, imsave
from scipy.stats import ttest_ind

from matplotlib.pyplot import figure, show, axes, sci
import matplotlib.image as mpimg
import matplotlib as mpl
from matplotlib import colors


# make a color map of fixed colors
cm = colors.ListedColormap(['grey','black', 'white'])
bounds=[-10,-0.5,0.5, 10]
norm = colors.BoundaryNorm(bounds, cm.N)

        
def add_sub_plot(fig, img, nimages, nsamples, sub_plot_idx, i, title, first_col=False, first_col_label=None):
    a = fig.add_subplot(nimages, nsamples + 2, sub_plot_idx)
    a.axis('off')
    if i==0:
        a.set_title(title)
    if first_col:    
#         a.text(0.99, 0.5, str(i))
        a.text(0.99, 0.5, first_col_label)
    else:
        plt.imshow(img, cmap=cm, norm=norm)
    return sub_plot_idx + 1

def get_fraction_of_ip(ip, nsteps):

    if nsteps == 300 or nsteps == 28: 
        w = h = 28
        ip_frac = np.reshape(ip, (w,h))
        indices = range(17)
        ip_frac = np.delete(ip_frac, indices, axis=0)
        for i in range(8):
            ip_frac[0,i] = -1
            
        if nsteps == 28: 
            indices = range(2,11)
            ip_frac = np.delete(ip_frac, indices, axis=0)
            for i in range(8,28):
                ip_frac[1,i] = -1
    elif nsteps==10:    
        #ip is 28x28 start from idx 484
        ip_frac = np.zeros((1, nsteps))
        for i in range(nsteps):
            ip_frac[0, i] = ip[i + 484]
            
    return ip_frac  


LUT_32 = {'s10': 1, 'f10': 4, 'v10': 7,
          's28': 7, 'f28': 23, 'v28': 13,
          's300': 8, 'f300': 4, 'v300': 1}

LUT_64 = {'s10': 44, 'f10': 36, 'v10': 40,
          's28': 10, 'f28': 40, 'v28': 15,
          's300': 9, 'f300': 5, 'v300': 6}

LUT_128 = {'s10': 10, 'f10': 16, 'v10': 1,
          's28': 26, 'f28': 36, 'v28': 1,
          's300': 0, 'f300': 7, 'v300': 4}

LUT_3x32 = {'s10': 3, 'f10': 13, 'v10': 7,
           's28': 3, 'f28': 6, 'v28': 0,
           's300': 4, 'f300': 7, 'v300': 5}

def show_report_in_paintings(samples, images, file_name, lut):       

    nimages = 9 # actually num rows
    nsamples = 5  
        
    fig = plt.figure()
    fig.patch.set_facecolor('grey')
    sub_plot_idx = 1
    i=0

    for key in ['s10', 'f10', 'v10', 's28', 'f28', 'v28', 's300', 'f300', 'v300']:
        image_idx = lut[key]
        nsteps = int(key[1:])
        
        rs = get_fraction_of_ip(images[image_idx], nsteps)
        sub_plot_idx = add_sub_plot(fig, None, 9, nsamples, sub_plot_idx, i, '', True, key)
        sub_plot_idx = add_sub_plot(fig, rs, nimages, nsamples, sub_plot_idx, i, 'gt')
        
        for sample_idx in range(nsamples):
            new_image = images[image_idx]
            new_image[-300:] = samples[sample_idx, image_idx, :]
            rs = get_fraction_of_ip(new_image, nsteps)
            sub_plot_idx = add_sub_plot(fig, rs, nimages, nsamples, sub_plot_idx, i, 's'+str((sample_idx+1)))
        i+=1
            
    plt.tight_layout()        
#     plt.subplots_adjust(left=None, bottom=0.02, right=0.36, top=0.97, wspace=0.5, hspace=0.12)
    plt.savefig(file_name)
    plt.show()

def show_in_paintings(samples, images, file_name, repeat, nsteps):       

    nimages = 10
#     image_idxs = np.random.randint(0, len(images), (nimages))
    nsamples = 5  
        
    fig = plt.figure()
    fig.patch.set_facecolor('grey')
    fig.suptitle(file_name+str(repeat), fontsize=11)
    sub_plot_idx = 1

    for i in range(nimages):
        image_idx = repeat*nimages + i
        rs = get_fraction_of_ip(images[image_idx], nsteps)
        sub_plot_idx = add_sub_plot(fig, None, nimages, nsamples, sub_plot_idx, i, '', True, str(i))
        sub_plot_idx = add_sub_plot(fig, rs, nimages, nsamples, sub_plot_idx, i, 'gt')
        
        for sample_idx in range(nsamples):
            new_image = images[image_idx]
            new_image[-300:] = samples[sample_idx, image_idx, :]
            rs = get_fraction_of_ip(new_image, nsteps)
            sub_plot_idx = add_sub_plot(fig, rs, nimages, nsamples, sub_plot_idx, i, 's'+str((sample_idx+1)))
    
#     plt.subplots_adjust(left=None, bottom=0.02, right=0.36, top=0.97, wspace=0.3, hspace=0.03)
    plt.tight_layout()
    plt.savefig(file_name)
    plt.show()
        
def get_cross_entropy(samples, images, model_probs, gt, file_name):
    
    nimages = model_probs.shape[0] #100
    npixels = model_probs.shape[1] #300
    nsamples = 10
    
    xent_gt = np.zeros((nimages, npixels), dtype='float32')
    xent_ip = np.zeros((nimages, nsamples, npixels), dtype='float32')
    
    images = np.squeeze(images, axis=2)
    
#     show_report_in_paintings(samples, images, file_name, LUT_3x32)
#     
#     nsteps = 300
#     for i in range(10):
#         show_in_paintings(samples, images, file_name, i, nsteps)
    
    for i in range(nimages):
        for j in range(npixels):
            xent_gt[i,j] -= gt[i,j] * np.log(model_probs[i,j]) + \
                                    (1-gt[i,j]) * np.log(1-model_probs[i,j])
            for k in range(nsamples):
                xent_ip[i,k,j] -= samples[k,i,j] * np.log(model_probs[i,j]) + \
                                        (1-samples[k,i,j]) * np.log(1-model_probs[i,j])
                               
    i=1
    print(i, '-step ground truth Xent', np.mean(xent_gt[:,:i]))
    print(i, '-step in-painting Xent', np.mean(xent_ip[:,0,:i])) #don't average across samples for some reason
    _, p_value = ttest_ind(xent_gt[:,:i], xent_ip[:,0,:i], axis=None)
    print(i, '-step t-test: p=', p_value)
    
    for i in [10,28,300]:
        print(i, '-step ground truth Xent', np.mean(xent_gt[:,:i]))
        print(i, '-step in-painting Xent', np.mean(xent_ip[:,:,:i]))
        
        _, p_value = ttest_ind(xent_gt[:,:i], xent_ip[:,:,:i], axis=None)
        print(i, '-step t-test: p=', p_value)
        
