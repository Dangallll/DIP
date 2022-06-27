import numpy as np
import tifffile as tiff
from utils.DIP import Image40



def DEMIX(image, filename, dark, P):
    frames = image.shape[0]

    a = np.zeros((frames, 120, 120), dtype='int16')

    for i in range(0, frames):

        k = Image40(image[i, :, :], P, dark)

        a[i, :, :] = k.processed()

        if i % 10000 == 0:
            print("processing", i, "th frame...")

    tiff.imsave(filename, a)
    print(a.dtype)

    print("=" * 50)
    print("done!")