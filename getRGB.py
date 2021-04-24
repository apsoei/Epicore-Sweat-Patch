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

#This will display all the available mouse click events  
# events = [i for i in dir(cv2) if 'EVENT' in i]
# print(events)

#This variable we use to store the pixel location
refPt = []

#click event function
# def click_event(event, x, y, flags, param):
#     if event == cv2.EVENT_LBUTTONDOWN:
#         print(x,",",y)
#         refPt.append([x,y])
#         font = cv2.FONT_HERSHEY_SIMPLEX
#         strXY = str(x)+", "+str(y)
#         cv2.putText(img, strXY, (x,y), font, 0.5, (255,255,0), 2)
#         cv2.imshow("image", img)

#     if event == cv2.EVENT_RBUTTONDOWN:
#         blue = img[y, x, 0]
#         green = img[y, x, 1]
#         red = img[y, x, 2]
#         font = cv2.FONT_HERSHEY_SIMPLEX
#         strBGR = str(blue)+", "+str(green)+","+str(red)
#         cv2.putText(img, strBGR, (x,y), font, 0.5, (0,255,255), 2)
#         cv2.imshow("image", img)

# reading the image
img = patch 

# displaying the image
print("should show image now !")
cv2.imshow('image', img)

# pix coords fixed
refPt = [[1130, 182], [1166, 202], 
          [1205, 223], [1247, 245],
          [1290, 270], [1334, 292],
          [1384, 322], [1433, 347],
          [1469, 368]]

print("before loop")
for coord in refPt:
    print("in loop")
    font = cv2.FONT_HERSHEY_SIMPLEX
    strXY = str(coord[0])+", "+str(coord[1])
    cv2.putText(img, strXY, (coord[0],coord[1]), font, 0.5, (255,255,0), 2)
    cv2.imshow("image", img)


# setting mouse hadler for the image
# and calling the click_event() function
# cv2.setMouseCallback('image', click_event)

# wait for a key to be pressed to exit

cv2.waitKey(0)
# print(refPt)
# print(np.shape(refPt))
R = 0
G = 0
B = 0

for pt in refPt:
    B += patch[pt[1],pt[0],0]
    G += patch[pt[1],pt[0],1]
    R += patch[pt[1],pt[0],2]

R /= len(refPt)
G /= len(refPt)
B /= len(refPt)

print("R,G,B = ",R,",",G,",",B)


# close the window
cv2.destroyAllWindows()






  
