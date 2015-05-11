 #!/usr/bin/python
 # -*- coding: utf-8 -*-

import cv2
import numpy as np
import json

import direct.directbase.DirectStart
from direct.showbase.DirectObject import DirectObject
import direct.gui.DirectGui as directgui
from direct.gui.OnscreenText import OnscreenText
import panda3d.core
import sys

class UserInterface:
    def __init__(self):
        pass

    @property
    def exit(self):
        return cv2.waitKey(1) >= 0

    @classmethod
    def display(cls, caption, img, coordxy=None):
        cv2.imshow(caption, img)
        if coordxy:
            assert isinstance(coordxy, tuple) and len(coordxy) == 2
            assert all(isinstance(coord, int) for coord in coordxy)
            cv2.moveWindow(caption, coordxy[0], coordxy[1])

class Capture(object):
    def __init__(self, material_id):
        """ material_id - Filename or Camera Index """
        self.material_id = material_id
        self.img = None

    def open(self):
        pass

    def get_frame(self):
        pass

    def kill(self):
        pass

class ImageController(Capture):
    def __init__(self, material_id):
        super(ImageController, self).__init__(material_id)
        self.path = '../media/img/'

    def open(self):
        file_path = self.path + self.material_id
        self.img = cv2.imread(file_path)
        self.img = cv2.resize(self.img, (700,500), interpolation=cv2.INTER_AREA)
        if self.img is None:
            raise Exception('No image to read')

    def get_frame(self):
        return self.img

class VideoController(Capture):
    def __init__(self, material_id):
        super(VideoController, self).__init__(material_id)
        self.path = '../media/video/'

    def open(self):
        file_path = self.path + self.material_id
        self.device = cv2.VideoCapture(file_path)

    def get_frame(self):
        ret, self.img = self.device.read()
        if (ret == False): # failed to capture
            print("Fail or end of capture.")
            return None
        return self.img

    def kill(self):
        """ Releases video capture device. """
        cv2.VideoCapture(self.material_id).release()

class CamController(VideoController):
    def open(self):
        cam_id = self.material_id
        self.device = cv2.VideoCapture(cam_id)
        self.device.set(cv2.CAP_PROP_FRAME_HEIGHT, 600);
        self.device.set(cv2.CAP_PROP_FRAME_WIDTH, 800);
        self.device.set(cv2.CAP_PROP_FPS, 30);

class MarkerDetector:
    def __init__(self):
        self.ui = UserInterface()
        self.nb_current_markers = 0
        self.nb_current_quadrangles = 0

    def __call__(self, img_orig):
        w, h = img_orig.shape[1], img_orig.shape[0]
        img_gray = self.rgb_to_gray(img_orig)
        img_threshold = self.canny_algorithm(img_gray)
        self.ui.display('canny', img_threshold, (600, 0))
        contours = self.find_contours(img_threshold)
        img_contours = np.zeros((h, w, 3), np.uint8)
        img_contours = cv2.drawContours(img_contours, contours, -1, (255,255,255), 1)
        self.ui.display('contours', img_contours, (1200, 0))
        markers = self.get_markers(img_gray, contours)
        return markers

    def rgb_to_gray(self, img):
        """ Converts an RGB image to grayscale, where each pixel
        now represents the intensity of the original image. """
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    def do_threshold(self, image, thresh=0):
        (thres, im_thres) = cv2.threshold(image, thresh, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        return thres, im_thres

    def canny_algorithm(self, image):
        kernel = np.ones((3, 3), np.uint8)
        image = cv2.dilate(image, kernel, (-1, -1), iterations=1)
        image = cv2.GaussianBlur(image, (3,3), 0)
        high_thresh_val = cv2.mean(image)[0] * 1.33
        lower_thresh_val = cv2.mean(image)[0] * 0.66
        image = cv2.Canny(image, lower_thresh_val, high_thresh_val)
        image = cv2.dilate(image, kernel, (-1, -1), iterations=1)
        image = cv2.erode(image, kernel,(-1, -1), iterations=1)
        return image

    def find_contours(self, threshold_img):
        (threshold_img, contours, hierarchy) = cv2.findContours(threshold_img,
            mode=cv2.RETR_CCOMP, method=cv2.CHAIN_APPROX_SIMPLE)
        markers_contours = []
        for i in range(len(contours)):
            # Approxime par un polynome
            epsilon = 0.025*cv2.arcLength(contours[i], True)
            approx_curve = cv2.approxPolyDP(contours[i], epsilon, True)
            if not cv2.isContourConvex(approx_curve):
                continue
            if cv2.contourArea(approx_curve) < 500:
                continue
            if len(approx_curve) != 4:
                continue
            if hierarchy[0][i][3] >= 0:
                continue
            markers_contours.append(approx_curve)
        return markers_contours

    def homothetie_marker(self, img_orig, points):
        # Find the perspective transfomation to get a rectangular 2D marker
        corners = self.curve_to_quadrangle(points)
        ideal_corners = np.float32([[0,0],[200,0],[0,200],[200,200]])
        sorted_corners = self.sort_corners(corners)
        M = cv2.getPerspectiveTransform(sorted_corners, ideal_corners)
        marker2D_img = cv2.warpPerspective(img_orig, M, (200,200))
        return marker2D_img

    def get_ref_markers(self):
        fic = open("ref_markers.json", "r")
        _str = fic.read()
        markers_array = json.loads(_str)
        return markers_array

    def get_bit_matrix(self, img, split_coeff):
        """ split_coeff : découpage de l'image en x*x cases
            cell_size : w, h
        """
        assert all(len(row) == len(img) for row in img) #matrice carrée
        cell_size = len(img)/split_coeff, len(img[0])/split_coeff
        bit_matrix = [[0 for x in range(split_coeff)] for y in range(split_coeff)]
        for y in range(split_coeff):
            for x in range(split_coeff):
                cell = img[(x*cell_size[0]):(x+1)*cell_size[0], (y*cell_size[1]):(y+1)*cell_size[1]]
                nb_white = cv2.countNonZero(cell)
                if nb_white > (cell_size[0]**2)/2:
                    bit_matrix[x][y] = 1 #is white
        return bit_matrix

    def get_markers(self, img_gray, positions):
        ref_markers = self.get_ref_markers()
        detected_markers = {}
        for i in range(len(positions)):
            img = self.homothetie_marker(img_gray, positions[i])
            img = self.do_threshold(img, 125)[1]
            detected_mat = self.get_bit_matrix(img, 6)
            for j in range(4):
                try:
                    id_ = ref_markers.index(detected_mat)
                except:
                    detected_mat = np.rot90(np.array(detected_mat)).tolist()
                else:
                    detected_markers[id_] = positions[i]
                    self.ui.display('marker_{} ({})'.format(id_, j), img, (i*300, 800))
                    break

        if self.nb_current_markers != len(detected_markers) or self.nb_current_quadrangles != len(positions):
            self.nb_current_markers = len(detected_markers)
            self.nb_current_quadrangles = len(positions)
            print("Quadrangles detected = {}, Markers detected = {} (id: {})".format(
            len(positions), len(detected_markers), detected_markers.keys()))

        return detected_markers

    def sort_corners(self, corners):
        top_corners = sorted(corners, key=lambda x : x[1])
        top = top_corners[:2]
        bot = top_corners[2:]
        if len(top) == 2 and len(bot) == 2:
    	    tl = top[1] if top[0][0] > top[1][0] else top[0]
    	    tr = top[0] if top[0][0] > top[1][0] else top[1]
    	    br = bot[1] if bot[0][0] > bot[1][0] else bot[0]
    	    bl = bot[0] if bot[0][0] > bot[1][0] else bot[1]
    	    corners = np.float32([tl, tr, br, bl])
            return corners
        raise Exception('len(bot) != 2 or len(top) != 2')

    def curve_to_quadrangle(self, points):
        assert points.size == 8, 'not a quadrangle'
        vertices = [p[0] for p in points]
        return np.float32([x for x in vertices])


class World(DirectObject):
    def __init__(self):
        self.ui = UserInterface()
        self.markerdetector = MarkerDetector()
        self.run = run

    def get_cv_img(self):
        self.img = self.capture.get_frame()
        shape = self.img.shape
        cv2.flip(self.img, 0, self.img) # cv2 image is upside down
        self.tex = panda3d.core.Texture("detect")
        self.tex.setCompression(panda3d.core.Texture.CMOff) # 1 to 1 copying - default, so is unnecessary
        self.tex.setup2dTexture(shape[1], shape[0],
                                    panda3d.core.Texture.TUnsignedByte, panda3d.core.Texture.FRgb)#FRgba8) # 3,4 channel
        self.tex.makeRamImage() # weird flicking results if omit this.
        self.tex.setRamMipmapPointerFromInt(self.img.ctypes.data, 0, 640*480*3)

    def set_capture(self, capture):
        self.capture = capture

    def start(self):
        base.setBackgroundColor(0.5,0.5,0.5)
        sm = panda3d.core.CardMaker('bg')
        sm.setFrameFullscreenQuad()
        self.test = render2d.attachNewNode(sm.generate(),2)
        self.accept('escape', sys.exit)
        taskMgr.add(self.turn, "turn")

    def turn(self, task):
        if self.ui.exit:
            sys.exit()
        self.get_cv_img()
        self.test.setTexture(self.tex)
        img = self.capture.get_frame()
        self.markerdetector(img)
        return task.cont


class Master:
    def __init__(self):
        self.world = World()

    def start(self, mode, material_id):
        self.mode = mode
        if mode == VID_MODE:
            self.capture = VideoController(material_id)
        elif mode == CAM_MODE:
            self.capture = CamController(material_id)
        elif mode == IMG_MODE:
            self.capture = ImageController(material_id)
        self.capture.open()
        self.world.set_capture(self.capture)
        self.world.start()
        self.world.run()

    def cleanup(self):
        """ Closes all OpenCV windows and releases video capture device. """
        cv2.destroyAllWindows()
        self.capture.kill()

# Modes
CAM_MODE = 1
VID_MODE = 2
IMG_MODE = 3
# Devices
IMG_EXAMPLE1 = 'markerQR.png'
VID_EXAMPLE1 = 'movement1.mp4'
CAM_INDEX = 0

def main():
    master = Master()
    #master.start(IMG_MODE, IMG_EXAMPLE1)
    master.start(VID_MODE, VID_EXAMPLE1)
    #master.start(CAM_MODE, CAM_INDEX)
    master.cleanup()

if __name__ == "__main__":
    main()
