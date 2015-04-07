#!/usr/bin/python

import cv2
import numpy
import sys

##
# Opens a video capture device with a resolution of 800x600
# at 30 FPS.
##
def open_video(cam_id = 0):
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

def find_contours(image):
    (image, contours0, hierarchy) = cv2.findContours(image.copy(), mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
    contours = [cv2.approxPolyDP(cnt, 3, True) for cnt in contours0]
    return contours, hierarchy

########### Main Program ###########

if __name__ == "__main__":
    # Camera ID to read video from (numbered from 0)
    video_filename = '../media/video/movement1.mp4'
    dev = open_video(video_filename) # open the camera as a video capture device

    while True:
        img_orig = get_frame(dev) # Get a frame from the camera
        if img_orig is not None: # if we did get an image
            h, w = img_orig.shape[:2]
            img_gray = rgb_to_gray(img_orig) # Convert img_orig from video camera from RGB to Grayscale
            # Converts grayscale image to a binary image with a threshold value of 220. Any pixel with an
            # intensity of <= 220 will be black, while any pixel with an intensity > 220 will be white:
            (thresh, img_threshold) = do_threshold(img_gray, 65)
            contours, hierarchy = find_contours(img_threshold)
            # Here, we are creating a new RBB image to display our results on
            vis = numpy.zeros((h, w, 3), numpy.uint8)
            cv2.drawContours(vis, contours, -1, (128,255,255),
                3, cv2.LINE_AA, hierarchy, 7 )

            # Display Results
            cv2.imshow("results", vis)
            cv2.imshow("Threshold", img_threshold)
            cv2.imshow("video", img_orig) # display the image in a window named "video"
        else: # if we failed to capture (camera disconnected?), then quit
            break

        if (cv2.waitKey(2) >= 0): # If the user presses any key, exit the loop
            break

    cleanup(video_filename) # close video device and windows before we exit
