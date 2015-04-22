 #!/usr/bin/python
 # -*- coding: utf-8 -*-

import cv2
import numpy
import json

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

class ImageCapture(Capture):
    def __init__(self, material_id):
        super(ImageCapture, self).__init__(material_id)
        self.path = '../media/img/'

    def open(self):
        file_path = self.path + self.material_id
        self.img = cv2.imread(file_path)
        self.img = cv2.resize(self.img, (700,500));
        if self.img is None:
            raise Exception('No image to read')

    def get_frame(self):
        return self.img

class VideoCapture(Capture):
    def __init__(self, material_id):
        super(VideoCapture, self).__init__(material_id)
        self.path = '../media/video/'

    def open(self):
        file_path = self.path + self.material_id
        self.device = cv2.VideoCapture(file_path)
        self.device.set(cv2.CAP_PROP_FRAME_HEIGHT, 600);
        self.device.set(cv2.CAP_PROP_FRAME_WIDTH, 800);
        self.device.set(cv2.CAP_PROP_FPS, 30);

    def get_frame(self):
        ret, self.img = self.device.read()
        if (ret == False): # failed to capture
            print("Fail or end of capture.")
            return None
        return self.img

    def kill(self):
        """ Releases video capture device. """
        cv2.VideoCapture(self.material_id).release()

class CamCapture(VideoCapture):
    def open(self):
        cam_id = self.material_id
        self.device = cv2.VideoCapture(cam_id)

class MarkerDetector:

    def __init__(self):
        self.ui = UserInterface()

    def __call__(self, img_orig):
        self.img_orig = img_orig
        w, h = img_orig.shape[1], img_orig.shape[0]
        img_gray = self.rgb_to_gray(img_orig)
        img_threshold = self.canny_algorithm(img_gray)
        self.ui.display('threshold', img_threshold, (600, 0))
        contours = self.find_contours(img_threshold)
        img_contours = numpy.zeros((h, w, 3), numpy.uint8)
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
        kernel = numpy.ones((3,3),numpy.uint8)
        image = cv2.dilate(image, kernel, (-1,-1), iterations=1)
        image = cv2.GaussianBlur(image, (3,3), 0)
        high_thresh_val = cv2.mean(image)[0]*1.33
        lower_thresh_val = cv2.mean(image)[0]*0.66
        image = cv2.Canny(image, lower_thresh_val, high_thresh_val)
        image = cv2.dilate(image, kernel, (-1,-1), iterations=1)
        image = cv2.erode(image,kernel,(-1,-1),iterations=1)
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
        ideal_corners = numpy.float32([[0,0],[200,0],[0,200],[200,200]])
        sorted_corners = self.sort_corners(corners)
        M = cv2.getPerspectiveTransform(sorted_corners, ideal_corners)
        marker2D_img = cv2.warpPerspective(img_orig, M, (200,200))
        return marker2D_img

    def get_ref_markers(self):
        fic = open("markers_ref.json", "r")
        _str = fic.read()
        markers_array = json.loads(_str)
        print len(markers_array)
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
        print("Quadrangles detected = {}".format(len(positions)))
        markers = self.get_ref_markers()
        detected_markers = {}
        for i in range(len(positions)):
            img = self.homothetie_marker(img_gray, positions[i])
            img = self.do_threshold(img, 125)[1]
            mat = self.get_bit_matrix(img, 8)
            try:
                id_ = markers.index(mat)
                detected_markers[id_] = positions[i]
                self.ui.display('marker2D_{} id={}'.format(i, id_), img, (i*300, 800))
            except:
                pass
        print("Markers detected = {}".format(len(detected_markers)))
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
    	    corners = numpy.float32([tl, tr, br, bl])
            return corners
        raise Exception('len(bot) != 2 or len(top) != 2')

    def curve_to_quadrangle(self, points):
        assert points.size == 8, 'not a quadrangle'
        vertices = [p[0] for p in points]
        return numpy.float32([x for x in vertices])

class MarkerTracker:
    def __init__(self):
        self.ui = UserInterface()


class Master:
    def __init__(self):
        self.ui = UserInterface()
        self.markerdetector = MarkerDetector()

    def start(self, mode, material_id):
        self.mode = mode
        if mode == VID_MODE:
            self.capture = VideoCapture(material_id)
        elif mode == CAM_MODE:
            self.capture = CamCapture(material_id)
        elif mode == IMG_MODE:
            self.capture = ImageCapture(material_id)
        self.capture.open()
        self.main_loop()

    def main_loop(self):
        first = True
        #while True:
        while not self.ui.exit:
            if self.mode == IMG_MODE and not first:
                continue
            img = self.capture.get_frame()
            if img is None: break
            self.ui.display('raw', img)
            cv2.moveWindow('raw', 0, 0)
            self.markerdetector(img)
            first = False

    def cleanup(self):
        """ Closes all OpenCV windows and releases video capture device. """
        cv2.destroyAllWindows()
        self.capture.kill()

# todo : Ameliorer le thresholding
#           Faire en fonction de la luminosité ambiante ?
#        Multiples détections d'un même marqueur, à corriger
#        Détecter si c'est réellement un marqueur, décomposer l'image 2D
#           reconstuire en petits morceaux et voir si les couleurs correspondent?
#        Représentation 3D du marqueur

# Modes
CAM_MODE = 1
VID_MODE = 2
IMG_MODE = 3
# Devices
IMG_EXAMPLE1 = 'markerQR2.png'
VID_EXAMPLE1 = 'rotation1.mp4'
CAM_INDEX = 0

def main():
    master = Master()
    #master.start(IMG_MODE, IMG_EXAMPLE1)
    #master.start(VID_MODE, VID_EXAMPLE1)
    master.start(CAM_MODE, CAM_INDEX)
    master.cleanup()

if __name__ == "__main__":
    main()
