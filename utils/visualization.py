import matplotlib.pyplot as plt

def plot_batch(item, imgch=0, sm_ch=0):
    img, (target, pdm, sm, voxel_size, item_idx) = item
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(12,12))
    ax[0,0].imshow(img[imgch])
    ax[0,0].set_title('Image ch {}'.format(imgch))
    ax[0,1].imshow(target[0])
    ax[0,1].set_title('Target')
    ax[1,0].imshow(pdm[0])
    ax[1,0].set_title('PDM')
    ax[1,1].imshow(sm[sm_ch])
    ax[1,1].set_title('SM ch {}'.format(sm_ch))
    plt.show()   
