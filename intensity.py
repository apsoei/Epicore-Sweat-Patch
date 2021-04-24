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
from scipy.optimize import curve_fit

parser = argparse.ArgumentParser()
parser.add_argument('data0', help='path to png file')
parser.add_argument('data1', help='path to png file')
parser.add_argument('data2', help='path to png file')
parser.add_argument('data3', help='path to png file')

args = parser.parse_args()
data0 = args.data0
data1 = args.data1
data2 = args.data2
data3 = args.data3

data = [data0,data1,data2,data3]

def objective(x,a,b):
    return a*x**b


#This will display all the available mouse click events  
events = [i for i in dir(cv2) if 'EVENT' in i]
print(events)

#This variable we use to store the pixel location
refPt = []
swatchPt = []
RGB = []
sRGB = []



#click event function
def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x,",",y)
        refPt.append([x,y])
        font = cv2.FONT_HERSHEY_SIMPLEX
        strXY = str(x)+", "+str(y)
        cv2.putText(img, strXY, (x,y), font, 0.5, (255,255,0), 2)
        cv2.imshow("image", img)

    if event == cv2.EVENT_RBUTTONDOWN:
        swatchPt.append([x,y])
        blue = img[y, x, 0]
        green = img[y, x, 1]
        red = img[y, x, 2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        strBGR = str(blue)+", "+str(green)+","+str(red)
        cv2.putText(img, strBGR, (x,y), font, 0.5, (0,255,255), 2)
        cv2.imshow("image", img)

# reading the image
refLab = []
swatchLab = []
for d in data:
    refPt.clear()
    swatchPt.clear()
    img = cv2.imread(d)
    cv2.imshow('image', img)
    cv2.setMouseCallback('image', click_event)
    cv2.waitKey(0)
    print(refPt)
    print(np.shape(refPt))
    R = 0
    G = 0
    B = 0
    sR=0
    sG=0
    sB=0

    for pt in refPt:
        B += img[pt[1],pt[0],0]
        G += img[pt[1],pt[0],1]
        R += img[pt[1],pt[0],2]
    for pt in swatchPt:
        sB += img[pt[1],pt[0],0]
        sG += img[pt[1],pt[0],1]
        sR += img[pt[1],pt[0],2]

    R /= len(refPt)
    G /= len(refPt)
    B /= len(refPt)
    sR /= len(swatchPt)
    sG /= len(swatchPt)
    sB /= len(swatchPt)
    RGB.append([R,G,B])
    sRGB.append([sR,sG,sB])

lab = []
slab = []

for rgb in RGB:
    lab.append(cv2.cvtColor( np.uint8([[rgb]] ), cv2.COLOR_RGB2LAB)[0][0])
for srgb in sRGB:
    slab.append(cv2.cvtColor( np.uint8([[srgb]] ), cv2.COLOR_RGB2LAB)[0][0])

lab = np.asarray(lab)
slab = np.asarray(slab)


diff =np.sqrt( ( (slab-lab)**2 ).sum(axis=1) )

x = np.array([0.75,1.25,1.75,2.25])

plt.figure(1)
plt.scatter(x,diff)
popt, _ = curve_fit(objective, x, diff)
a,b = popt
print("curve fit eqn: \n")
print('y = %.5f * x ^%.5f' %(a, b) )
# define a sequence of inputs between the smallest and largest known inputs
x_line = np.arange(0, max(x)+0.1, 0.1)
# calculate the output for the range
y_line = objective(x_line, a, b)
# create a line plot for the mapping function
plt.plot(x_line, y_line, '--', color='red')
strText = 'y = ' + str(round(a,2)) +"x^" +str(round(b,2))
plt.text(min(x),max(diff),strText)
plt.xlabel("Chloride concentration [g / L]")
plt.ylabel("CIELAB Color value difference (vector length)")
plt.grid()
plt.savefig("./plots/curve.png")
plt.show()

print("channel lab values: \n",lab)
print("swatch lab values: \n",slab)
print("diff (norm of vectors) lab values: \n",diff)



'''
# displaying the image
cv2.imshow('image', img)
print("?????")
# setting mouse hadler for the image
# and calling the click_event() function
cv2.setMouseCallback('image', click_event)

# wait for a key to be pressed to exit

cv2.waitKey(0)
print(refPt)
print(np.shape(refPt))
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


'''



  
