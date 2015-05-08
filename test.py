import numpy as np
import cv2

im = cv2.imread("media/img/marker1.JPG")


imgray = cv2.cvtColor(im,cv2.COLOR_RGB2GRAY)
thresh = cv2.adaptiveThreshold(imgray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,7,7)
im, all_contour, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)


cv2.imshow("Coucou", imgray)

cv2.drawContours(imgray, all_contour, -1, (0,255,0),3)

cv2.imshow("CAPTURE", im)
cv2.waitKey(0)
cv2.destroyAllWindows()


print "here"
