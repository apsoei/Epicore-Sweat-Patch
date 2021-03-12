import matplotlib.pyplot as plt
import numpy as np
import cv2
from matplotlib.colors import hsv_to_rgb
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import colors

patch = cv2.imread('./Pngs/patch2.png')
patch_RGB = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
# plt.imshow(patch_RGB)
# plt.show()
# flags = [i for i in dir(cv2) if i.startswith('COLOR_')]

# light_orange = (1, 190, 200)
'''
in HSV >>>
'''
light_orange = (0, 170, 190)
# dark_orange = (0, 89, 253)
dark_orange = (18, 255, 255)
'''
<<< in HSV 
'''

# print("1")
patch_HSV = cv2.cvtColor(patch_RGB, cv2.COLOR_RGB2HSV)
# ## >>
# # r, g, b = cv2.split(patch_RGB)
# # fig = plt.figure()
# # axis = fig.add_subplot(1, 1, 1, projection="3d")

# pixel_colors = patch_RGB.reshape((np.shape(patch_RGB)[0]*np.shape(patch_RGB)[1], 3))
# print("2")
# norm = colors.Normalize(vmin=-1.,vmax=1.)
# print("3")
# norm.autoscale(pixel_colors)
# print("4")
# pixel_colors = norm(pixel_colors).tolist()
# print("5")


# ## <<

# ## >>
# h, s, v = cv2.split(patch_HSV)
# print("6")
# fig = plt.figure()
# print("7")
# axis = fig.add_subplot(1, 1, 1, projection="3d")
# print("8")

# axis.scatter(h.flatten(), s.flatten(), v.flatten(), facecolors=pixel_colors, marker=".")
# axis.set_xlabel("Hue")
# axis.set_ylabel("Saturation")
# axis.set_zlabel("Value")
# print("9")
# plt.show()
# print("10")

## <<




mask = cv2.inRange(patch_HSV,light_orange,dark_orange)
result = cv2.bitwise_and(patch_RGB,patch_RGB, mask=mask)

# cnts = cv2.findContours(mask, 
#         cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
cnts, hierarchy = cv2.findContours(mask, 
        cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
# print(hierarchy)
cnts = cnts[0] if len(cnts)==2 else cnts[1]
print(len(cnts))


area = 0
for c in cnts:
    area += cv2.contourArea(c)
    cv2.drawContours(patch_HSV,[c], 0, (0,0,0), 3)
    # cv2.drawContours(patch_HSV,[c],0,(255,255,255),3)

print("area in pixel= ", area)


cv2.imshow('patch_HSV', patch_HSV)
cv2.waitKey()
cv2.imshow('mask', mask)
cv2.waitKey()


# plt.subplot(1, 2, 1)
# plt.imshow(mask, cmap="gray")
# plt.subplot(1, 2, 2)
# plt.imshow(result)
# plt.show()

# Converting the image to grayscale.
gray = cv2.cvtColor(patch.copy(), cv2.COLOR_BGR2GRAY)

# Using the Canny filter to get contours
edges = cv2.Canny(gray, 20, 30)
# Using the Canny filter with different parameters
edges_high_thresh = cv2.Canny(gray, 60, 120)
# Stacking the images to print them together for comparison
images = np.hstack((gray, edges, edges_high_thresh))

# Display the results
cv2.imshow('gray, edge, edge high', images)
cv2.waitKey()