
import argparse

from utils.DIP import Image40
from utils.demix import DEMIX
import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt


#argument parsing

parser = argparse.ArgumentParser(description='Original image, Dark image(for backgroound), Original ratios')
parser.add_argument('--src_path', required=True, type = str, help='path of image to be processed')
parser.add_argument('--dark_path', required=False, help='path of dark image to be processed')
parser.add_argument('--new_name', required=True, help='new file name ****.tiff')
parser.add_argument('--ratio_path', required=True, type = str, help='path of ratio')
parser.add_argument('--tune', required=False, action="store_true")


args = parser.parse_args()


IMG = tiff.imread(args.src_path) #image
Frames = IMG.shape[0]
print("\nloaded image!")

P = np.genfromtxt(args.ratio_path, dtype='float', delimiter=",") # P
print("\nloaded ratio!")


if args.dark_path == None:
    bgn = np.min(IMG)
    Dark = np.ones((120,120), dtype = 'uint8') * 355
else:
    Dark = tiff.imread(args.dark_path) # dark

print("\nDefined Dark")

if args.tune == True:
    img = Image40(IMG[int(Frames/2), :, :], P, Dark)
    print("\ninitial loss:", img.loss())
    img.Sfinetune()
    print("\nadjusted loss:", img.loss())

    adjP = img.P

else:
    adjP = P

print("\n Defined ratio!")
print("\n DEMIXING")

DEMIX(IMG, args.new_name, Dark, adjP)


