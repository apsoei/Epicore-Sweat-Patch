# lower_bound = (40,70,70)
# upper_bound = (180,255,255)

import matplotlib.pyplot as plt
import numpy as np
import cv2
from matplotlib.colors import hsv_to_rgb, rgb_to_hsv
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import colors
import argparse
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import colors

parser = argparse.ArgumentParser()
parser.add_argument('data', help='path to png file')
args = parser.parse_args()
dataPath = args.data

patch = cv2.imread(dataPath)
patch_RGB = cv2.cvtColor(patch.copy(), cv2.COLOR_BGR2RGB)

# pink1 = np.uint8([[[255,192,203]]])
# pink2 = np.uint8([[[199,21,133]]])
# pink1_h = cv2.cvtColor(pink1,cv2.COLOR_RGB2HSV)
# pink2_h = cv2.cvtColor(pink2,cv2.COLOR_RGB2HSV)

# purple1 = np.uint8([[[75,0,130]]])
# purple2 = np.uint8([[[230,230,250]]])
# purple1_h = cv2.cvtColor(purple1,cv2.COLOR_RGB2HSV)
# purple2_h = cv2.cvtColor(purple2,cv2.COLOR_RGB2HSV)

""" Color plot >>> """
# r, g, b = cv2.split(patch_RGB)
# fig = plt.figure()
# axis = fig.add_subplot(1, 1, 1, projection="3d")
# pixel_colors = patch_RGB.reshape((np.shape(patch_RGB)[0]*np.shape(patch_RGB)[1], 3))
# norm = colors.Normalize(vmin=-1.,vmax=1.)
# norm.autoscale(pixel_colors)
# pixel_colors = norm(pixel_colors).tolist()
# axis.scatter(r.flatten(), g.flatten(), b.flatten(), facecolors=pixel_colors, marker=".")
# axis.set_xlabel("Red")
# axis.set_ylabel("Green")
# axis.set_zlabel("Blue")
# plt.show()
"""  <<< Color plot """

# print(pink1_h)
# print(pink2_h)
# print(purple1_h)
# print(purple2_h)


# purple1_h = (240, 8, 98)
# purple3_h = (274, 100, 50)

# pink1_h = (322, 89, 78)
# pink2_h = (349, 24, 100)

patch_HSV = cv2.cvtColor(patch_RGB, cv2.COLOR_RGB2HSV)

"""
# hsv range
"""
# purple1 = np.uint8([[[204,153,255]]])
# purple2 = np.uint8([[[153,51,255]]])

# lower_bound =rgb_to_hsv(purple1)
# upper_bound =rgb_to_hsv(purple2)
# print("low = ", lower_bound)
# print("upp = ", upper_bound)

# lower_bound = (150,70,70)
# upper_bound = (200,255,255)

lower_bound = (40,70,70)
upper_bound = (180,255,255)

"""
# hsv range
"""


mask = cv2.inRange(patch_HSV, lower_bound, upper_bound)
# result = cv2.bitwise_and(patch_RGB,patch_RGB, mask=mask)


cnts, hierarchy = cv2.findContours(mask, 
        cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# print(np.shape(cnts))


area = 0
# newCnts = list()
# for c in cnts:
#     # if(cv2.contourArea(c) > 10000):
#     area += cv2.contourArea(c)
#     # print(np.shape(c))
#     newCnts.append(c)
#     cv2.drawContours(patch,[c],0,(0,255,0),cv2.FILLED)
#     cv2.drawContours(patch_HSV,[c],0,(0,255,0),cv2.FILLED)
#     cv2.drawContours(mask,[c],0,(128),cv2.FILLED)
# newCnts = np.asarray(newCnts)

# c = max(cnts, key = cv2.contourArea)
# cv2.drawContours(patch,[c],0,(0, 255, 0),3)
cv2.drawContours(patch,cnts,-1,(0, 255, 0),3)

cv2.imshow("origianl",patch)
cv2.waitKey()



# cv2.drawContours(mask,[c],0,255,-1)
pixelpoints = np.transpose(np.nonzero(mask))
print(np.shape(pixelpoints))
print("len pix points = ",len(pixelpoints))

(H,S,V) = (0,0,0)
overFlow = 0
for row in pixelpoints:
# print(overFlow)
    H += patch_HSV[row[0],row[1],0]
    S += patch_HSV[row[0],row[1],1]
    V += patch_HSV[row[0],row[1],2]

H /= len(pixelpoints)
S /= len(pixelpoints)
V /= len(pixelpoints)
print("H,S,V = ",H,",",S,",",V)


lo_square = np.full((10, 10, 3), (H,S,V), dtype=np.uint8) / 255.0
# plt.subplot(1, 2, 2)
# plt.imshow(hsv_to_rgb(lo_square))
# plt.show()

cv2.imshow("patch in hsv",patch_HSV)
cv2.waitKey()
# print("contour points = ", cnts[1])






# print("---------------")
# print(np.shape(newCnts))
# print("---------------")


# Initialize empty list
lst_intensities = []

# For each list of contour points...
# for i in range(len(newCnts)):
#     # Create a mask image that contains the contour filled in
#     cimg = np.zeros_like(patch_HSV)
#     cv2.drawContours(cimg, newCnts, i, color=255, thickness=-1)

#     # Access the image pixels and create a 1D numpy array then add to list
#     pts = np.where(cimg == 255)
#     print(pts[:][:6])
#     lst_intensities.append(patch_HSV[pts[0], pts[1]])
