import numpy as np
import cv2
import glob

# # termination criteria
# criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
# objp = np.zeros((6*7,3), np.float32)
# objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)
#
# # Arrays to store object points and image points from all the images.
# objpoints = [] # 3d point in real world space
# imgpoints = [] # 2d points in image plane.
#
# images = glob.glob('/home/elias/git/opencv-AR/media/img/laptop_camera/chessboard/*.JPG')
#
# for fname in images:
#     img = cv2.imread(fname)
#     gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#
#     # Find the chess board corners
#     ret, corners = cv2.findChessboardCorners(gray, (7,6),None)
#
#     # If found, add object points, image points (after refining them)
#     if ret == True:
#         objpoints.append(objp)
#
#         corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
#         imgpoints.append(corners2)
#
#         # Draw and display the corners
#         img = cv2.drawChessboardCorners(img, (7,6), corners2,ret)
#         ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
#         print(mtx, dist)
#         raw_input()
#         cv2.imshow('img',img)
#         cv2.waitKey(500)
#
# cv2.destroyAllWindows()

#####################################################################################

def draw(img, corners, imgpts):
    print(imgpts)
    imgpts = np.int32(imgpts).reshape(-1,2)

    # draw ground floor in green
    img = cv2.drawContours(img, [imgpts[:4]],-1,(0,255,0),-3)

    # draw pillars in blue color
    for i,j in zip(range(4),range(4,8)):
        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]),(255),3)

    # draw top layer in red color
    img = cv2.drawContours(img, [imgpts[4:]],-1,(0,0,255),3)

    return img

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
axis = np.float32([[0,0,0], [0,3,0], [3,3,0], [3,0,0],
                   [0,0,-3],[0,3,-3],[3,3,-3],[3,0,-3] ])
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)

#mtx, dist = np.array([[ 535.99470079,    0.        ,  334.33378325],[   0.        ,  535.99526309,  239.81140202],[   0.        ,    0.        ,    1.        ]]), np.array([[ -3.11676235e-01,   2.24977910e-01,   7.59760385e-04,-2.39005054e-04,  -1.95189232e-01]])
mtx, dist = (np.array([[ 859.87780302,    0.        ,  682.3173388 ],
       [   0.        ,  799.06029505,  382.57139684],
       [   0.        ,    0.        ,    1.        ]]), np.array([[ 0.2529905 , -0.12695403, -0.01182944, -0.02478326, -0.48389559]]))

#for fname in glob.glob('/home/elias/OpenCV/opencv/samples/data/left*.jpg'):
for fname in glob.glob('/home/elias/git/opencv-AR/media/img/laptop_camera/chessboard/*.JPG'):
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (7,6),None)

    if ret == True:
        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)

        # Find the rotation and translation vectors.
        inliers, rvecs, tvecs = cv2.solvePnP(objp, corners2, mtx, dist)

        # project 3D points to image plane
        imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)

        img = draw(img,corners2,imgpts)
        cv2.imshow('img',img)
        k = cv2.waitKey(0) & 0xff
        if k == 's':
            cv2.imwrite(fname[:6]+'.png', img)

cv2.destroyAllWindows()
