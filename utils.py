import numpy as np
import matplotlib.pyplot as plt

def show(img, ax=None):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)), interpolation='nearest')
    plt.axis('off')
    plt.show()