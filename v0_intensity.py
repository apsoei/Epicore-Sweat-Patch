# 'purple': [[158, 255, 255], [129, 50, 70]],



import matplotlib.pyplot as plt
import numpy as np
import cv2
from matplotlib.colors import hsv_to_rgb
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import colors



patch = cv2.imread('../Pngs/patch2.png')
patch_RGB = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
'''
in HSV >>>
'''
# purple_1 = (158, 255, 255)
# purple_2 = (129, 50, 70)
# '''
# <<< in HSV 
# '''
# patch_HSV = cv2.cvtColor(patch_RGB, cv2.COLOR_RGB2HSV)
# mask = cv2.inRange(patch_HSV,purple_2,purple_1)
# result = cv2.bitwise_and(patch_RGB,patch_RGB, mask=mask)

# cv2.imshow('patch_HSV', patch_HSV)
# cv2.waitKey()
# cv2.imshow('mask', mask)
# cv2.waitKey()

# gray = cv2.cvtColor(patch.copy(), cv2.COLOR_BGR2GRAY)

# # Using the Canny filter to get contours
# edges = cv2.Canny(gray, 20, 30)
# # Using the Canny filter with different parameters
# edges_high_thresh = cv2.Canny(gray, 100, 200)
# # Stacking the images to print them together for comparison
# images = np.hstack((gray, edges, edges_high_thresh))
# cv2.imshow('gray, edge, edge high', images)
# cv2.waitKey()
plt.imshow(patch)
plt.show()
b,g,r = (patch_RGB[3000, 2000])
print (r)
print (g)
print (b)


########################################################################################################################
im = patch.copy()
imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(imgray, 127, 255, 0)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

cv2.drawContours(imgray, contours, -1, (0,255,0), 3)
cv2.imshow("gray",imgray)