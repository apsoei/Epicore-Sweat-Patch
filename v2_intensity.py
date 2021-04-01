import matplotlib.pyplot as plt
import numpy as np
import cv2
from matplotlib.colors import hsv_to_rgb
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import colors


patch = cv2.imread('../Pngs/patch2.png')
patch_RGB = cv2.cvtColor(patch.copy(), cv2.COLOR_BGR2RGB)

# pink1 = np.uint8([[[255,192,203]]])
# pink2 = np.uint8([[[199,21,133]]])
# pink1_h = cv2.cvtColor(pink1,cv2.COLOR_RGB2HSV)
# pink2_h = cv2.cvtColor(pink2,cv2.COLOR_RGB2HSV)

# purple1 = np.uint8([[[75,0,130]]])
# purple2 = np.uint8([[[230,230,250]]])
# purple1_h = cv2.cvtColor(purple1,cv2.COLOR_RGB2HSV)
# purple2_h = cv2.cvtColor(purple2,cv2.COLOR_RGB2HSV)





# print(pink1_h)
# print(pink2_h)
# print(purple1_h)
# print(purple2_h)


purple1_h = (240, 8, 98)
purple3_h = (274, 100, 50)

pink1_h = (322, 89, 78)
pink2_h = (349, 24, 100)

patch_HSV = cv2.cvtColor(patch_RGB, cv2.COLOR_RGB2HSV)


# hsv range
lower_bound = (40,70,70)
upper_bound = (180,255,255)

# mask = cv2.inRange(patch_HSV,pink1_h,pink2_h)
# mask = cv2.inRange(patch_HSV,purple1_h,purple2_h)
mask = cv2.inRange(patch_HSV, lower_bound, upper_bound)
result = cv2.bitwise_and(patch_RGB,patch_RGB, mask=mask)
# cv2.imshow("mask",mask)
# cv2.waitKey()

cnts, hierarchy = cv2.findContours(mask, 
        cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# print(np.shape(cnts))

print("contour len = ",len(cnts))
area = 0
newCnts = list()
for c in cnts:
    # if(cv2.contourArea(c) > 10000):
    area += cv2.contourArea(c)
    # print(np.shape(c))
    newCnts.append(c)
    cv2.drawContours(patch_RGB,[c],0,(0,255,0),cv2.FILLED)
    cv2.drawContours(patch_HSV,[c],0,(0,255,0),cv2.FILLED)
    cv2.drawContours(mask,[c],0,(128),cv2.FILLED)
newCnts = np.asarray(newCnts)
# print("---------------")
# print(np.shape(newCnts))
# print("---------------")


# Initialize empty list
lst_intensities = []

# For each list of contour points...
for i in range(len(newCnts)):
    # Create a mask image that contains the contour filled in
    cimg = np.zeros_like(patch_HSV)
    cv2.drawContours(cimg, newCnts, i, color=255, thickness=-1)

    # Access the image pixels and create a 1D numpy array then add to list
    pts = np.where(cimg == 255)
    print(pts[:][:6])
    lst_intensities.append(patch_HSV[pts[0], pts[1]])




cv2.imshow("cimg",cimg)
cv2.waitKey()
lst_intensities = np.asarray(lst_intensities)

total = len(lst_intensities[0][:][0])

mean_h = np.sum(lst_intensities[0][:][0])/total
mean_s = np.sum(lst_intensities[0][:][1])/total
mean_v = np.sum(lst_intensities[0][:][2])/total

print('h,s,v = ',mean_h,",",mean_s,",",mean_v)

# mean = cv2.mean(patch_HSV,mask=pts)
# print(mean)
print(np.shape(lst_intensities))
# print("1")
# cv2.drawContours(patch_HSV,[cnts[-1]],0,(0,0,255),cv2.FILLED)
# print("2")
# print(len(cnts))
# cv2.drawContours(mask,cnts,200,(128,128,128),cv2.FILLED)
# print("3")
    # cv2.drawContours(patch_HSV,[c], 0, (0,0,0), 3)

# result = cv2.bitwise_and(patch, patch, mask=mask)    
# plt.imshow(result)
# plt.show()
# print("area in pixel= ", area)



cv2.imshow("origianl",patch_RGB)
cv2.waitKey()

# plt.imshow(patch_HSV)
# plt.show()
# cv2.imshow('patch_HSV', patch_HSV)
# cv2.waitKey()
# cv2.imshow('mask', mask)
# cv2.waitKey()