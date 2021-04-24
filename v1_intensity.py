import matplotlib.pyplot as plt
import numpy as np
import cv2
from matplotlib.colors import hsv_to_rgb
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import colors


image = cv2.imread('../Pngs/patch2.png',0)




'''
in HSV >>>
'''



RGB_dark_purple = np.uint8([[[75,0,130 ]]])
HSV_dark_purple  = cv2.cvtColor(RGB_dark_purple,cv2.COLOR_RGB2HSV)
print(HSV_dark_purple)
RGB_light_purple = np.uint8([[[230,230,250 ]]])
HSV_light_purple = cv2.cvtColor(RGB_light_purple,cv2.COLOR_RGB2HSV)
print(HSV_light_purple)

'''
<<< in HSV 
'''

# patch_HSV = cv2.cvtColor(patch_RGB.copy(), cv2.COLOR_RGB2HSV)

# mask = cv2.inRange(patch_HSV,HSV_light_purple,HSV_dark_purple)

# cnts, hierarchy = cv2.findContours(mask, 
#         cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# area = 0

# for c in cnts:
#     if(cv2.contourArea(c) > 10000):
#         area += cv2.contourArea(c)
        
#     cv2.drawContours(patch_HSV,[c],0,(0,0,255),cv2.FILLED)
#     cv2.drawContours(mask,[c],0,(128),cv2.FILLED)

# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# _, binary = cv2.threshold(gray, 225, 255, cv2.THRESH_BINARY_INV)
# contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# image = cv2.drawContours(image, contours, -1, (0, 255, 0), 2)
# plt.imshow(image)
# plt.show()



edges = cv2.Canny(image,100,200)
plt.subplot(121),plt.imshow(image,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
plt.show()
