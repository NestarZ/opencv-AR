#!/usr/bin/python
#
# ENGR421 -- Applied Robotics, Spring 2013
# OpenCV Python Demo
# Taj Morton <mortont@onid.orst.edu>
#
import sys
import cv2
import time
import numpy
import os

##
# Opens a video capture device with a resolution of 800x600
# at 30 FPS.
##
def open_camera(cam_id = 0):
    cap = cv2.VideoCapture(cam_id)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600);
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800);
    cap.set(cv2.CAP_PROP_FPS, 30);
    return cap

##
# Gets a frame from an open video device, or returns None
# if the capture could not be made.
##
def get_frame(device):
    ret, img = device.read()
    if (ret == False): # failed to capture
        print >> sys.stderr, "Error capturing from video device."
        return None
    return img

##
# Closes all OpenCV windows and releases video capture device
# before exit.
##
def cleanup(cam_id = 0):
    cv2.destroyAllWindows()
    cv2.VideoCapture(cam_id).release()

##
# Creates a new RGB image of the specified size, initially
# filled with black.
##
def new_rgb_image(width, height):
    image = numpy.zeros( (height, width, 3), numpy.uint8)
    return image

##
# Converts an RGB image to grayscale, where each pixel
# now represents the intensity of the original image.
##
def rgb_to_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

##
# Converts an image into a binary image at the specified threshold.
# All pixels with a value <= threshold become 0, while
# pixels > threshold become 1
def do_threshold(image, threshold = 170):
    (thresh, im_bw) = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
    return (thresh, im_bw)

# Finds the outer contours of a binary image and returns a shape-approximation
# of them. Because we are only finding the outer contours, there is no object
# hierarchy returned.
##
def find_contours(image):
    (image, contours, hierarchy) = cv2.findContours(image, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
    return contours

########### Main Program ###########

if __name__ == "__main__":
    # Camera ID to read video from (numbered from 0)
    camera_id = '../media/video/movement1.mp4'
    dev = open_camera(camera_id) # open the camera as a video capture device

    while True:
        img_orig = get_frame(dev) # Get a frame from the camera
        #####################################################s
        # If you have captured a frame from your camera like in the template program above,
        # you can create a bitmap from it as follows:

        img_gray = rgb_to_gray(img_orig) # Convert img_orig from video camera from RGB to Grayscale

        # Converts grayscale image to a binary image with a threshold value of 220. Any pixel with an
        # intensity of <= 220 will be black, while any pixel with an intensity > 220 will be white:
        (thresh, img_threshold) = do_threshold(img_gray, 60)

        cv2.imshow("Threshold", img_threshold)



        #####################################################
        # If you have created a binary image as above and stored it in "img_threshold"
        # the following code will find the contours of your image:
        contours = find_contours(img_threshold)

        # Here, we are creating a new RBB image to display our results on
        results_image = new_rgb_image(img_threshold.shape[1], img_threshold.shape[0])
        #cv2.drawContours(results_image, contours, -1, (255,0,0), 2)
        for x in contours:
            sommets = len(cv2.approxPolyDP(x, 10, True))
            if sommets == 4:
                rect = cv2.minAreaRect(contours[1])
                box = cv2.boxPoints(rect)
                box = numpy.int0(box)
                cv2.drawContours(results_image,[box],0,(0,0,255),2)

        # Display Results
        if img_orig is not None: # if we did get an image
            cv2.imshow("results", results_image)
        else: # if we failed to capture (camera disconnected?), then quit
            break

        if (cv2.waitKey(2) >= 0): # If the user presses any key, exit the loop
            break

    cleanup(camera_id) # close video device and windows before we exit
