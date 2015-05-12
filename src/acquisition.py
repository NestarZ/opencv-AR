 #!/usr/bin/python
 # -*- coding: utf-8 -*-

import cv2
import numpy as np
import json

import random
from panda3d.core import *
loadPrcFileData("", "window-title AR Mathieu & ELias")
loadPrcFileData("", "win-size 640 360")

from direct.showbase.DirectObject import DirectObject
from direct.gui.OnscreenText import OnscreenText
from direct.showbase.ShowBase import ShowBase


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

    def homothetie_marker(self, img_orig, corners):
        # Find the perspective transfomation to get a rectangular 2D marker
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
            sorted_corners = self.curve_to_quadrangle(positions[i])
            img = self.homothetie_marker(img_gray, sorted_corners)
            img = self.do_threshold(img, 125)[1]
            detected_mat = self.get_bit_matrix(img, 6)
            for j in range(4):
                try:
                    id_ = ref_markers.index(detected_mat)
                except:
                    detected_mat = np.rot90(np.array(detected_mat)).tolist()
                else:
                    detected_markers[id_] = sorted_corners.tolist()
                    self.ui.display('marker_{} ({})'.format(id_, j), img, (i*300, 800))
                    break

        #if self.nb_current_markers != len(detected_markers) or self.nb_current_quadrangles != len(positions):
        #    self.nb_current_markers = len(detected_markers)
        #    self.nb_current_quadrangles = len(positions)
        #    print("Quadrangles detected = {}, Markers detected = {} (id: {})".format(
        #    len(positions), len(detected_markers), detected_markers.keys()))

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


class MeshController:
    dict_mesh = {0:'cube'}
    path = '../media/mesh/'
    def __init__(self, id_=0):
        id_=0
        self.obj_name = MeshController.dict_mesh[id_]
        self.obj = loader.loadModel(MeshController.path + self.obj_name)

    def set(self, pos3d, scale):
        pos3d = pos3d.tolist()
        self.obj.setPos(pos3d[0][0][0],pos3d[1][0][0],1) # to test
        self.obj.setScale(scale)

    def reparentTo(self, parent):
        self.obj.reparentTo(parent)

class World(ShowBase):
    def __init__(self):
        ShowBase.__init__(self)
        self.ui = UserInterface()
        self.markerdetector = MarkerDetector()
        self.run = run
        self.obj_list = {}
        self.title = self.addTitle("AR Mathieu & Elias")


    def set_capture(self, capture):
        self.capture = capture

    def get_cv_img(self):
        img = self.capture.get_frame()
        img = cv2.flip(img, 0)
        w, h = img.shape[0:2]
        tex = Texture("webcam_img")
        tex.setup2dTexture(h, w, Texture.TUnsignedByte, Texture.FRgb)
        tex.setRamImage(img)
        return tex

    def start(self):
        self.camPos = "{} {} {}".format(camera.getX(),camera.getY(),camera.getZ())
        self.inst1 = self.addInstructions(0.24, self.camPos)
        self.objectPos = ""
        self.inst2 = self.addInstructions(0.30, self.objectPos)


        self.tex = self.get_cv_img()

        cm = CardMaker("Background")
        cm.setFrameFullscreenQuad()

        cm.setUvRange(self.tex)

        self.empty = self.render.attachNewNode("BG")
        self.card = self.empty.attachNewNode(cm.generate())
        self.card.setTexture(self.tex)
        self.card.setScale(2)

        self.initializeKey()

        camera.setPos(0.0, -60.0, 0)
        camera.lookAt(0.0, 0.0, 0.0)
        lens = OrthographicLens()
        lens.setFilmSize(4, 4)
        base.cam.node().setLens(lens)

        self.objNode = self.empty.attachNewNode("Objets")

        base.disableMouse()

        alight = AmbientLight('alight')
        alight.setColor(VBase4(0.2, 0.2, 0.2, 1))
        alnp = self.objNode.attachNewNode(alight)
        self.objNode.setLight(alnp)

        taskMgr.add(self.turn, "turn")

    def initializeKey(self):
        self.accept('arrow_up', lambda : self.move_object(camera,0,8,0))
        self.accept('arrow_down', lambda : self.move_object(camera,0,-8,0))
        self.accept('arrow_left', lambda : self.move_object(camera,8,0,0))
        self.accept('arrow_right', lambda : self.move_object(camera,-8,0,0))
        self.accept('w', lambda : self.move_object(camera,0,0,8))
        self.accept('x', lambda : self.move_object(camera,0,0,-8))

    def move_object(self, objet, dx, dy, dz):
        objet.setX(objet.getX() + dx)
        objet.setY(objet.getY() + dy)
        objet.setZ(objet.getZ() + dz)

        self.camPos = "{} {} {}".format(objet.getX(),objet.getY(),objet.getZ())
        self.inst1.setText(self.camPos)

    def add_mesh(self,path, id_obj):
        assert isinstance(path, str)
        if id_obj not in self.obj_list:
            cube = loader.loadModel(path)
            cube.setScale(0.0005)
            cube.reparentTo(self.objNode)
            self.obj_list[id_obj] = cube

    def turn(self, task):
        if self.ui.exit:
            pass
        img = self.capture.get_frame()
        tex = self.get_cv_img()
        self.card.setTexture(tex)
        self.markers = self.markerdetector(img)
        for mesh in self.obj_list.values():
            mesh.hide()
        for id_obj, pos in self.markers.items():
            if id_obj not in self.obj_list:
                self.add_mesh('../media/mesh/cup', id_obj)
	    mesh = self.obj_list[id_obj]
            mesh.show()
	    x, y, z = (pos[0][0]-320)/320.0, 0, (pos[0][1]-180)/-180.0
            mesh.setPos(self.card, x, y, z)
            self.objectPos = "cube : {} {} {}".format(mesh.getX(),mesh.getY(),mesh.getZ())
            self.inst2.setText(self.objectPos)
	print(id_obj, (x,y,z))
        return task.cont

    def addInstructions(self, pos, msg):
        return OnscreenText(text=msg, style=1, fg=(0, 0, 0, 1), shadow=(1, 1, 1, 1),
                            parent=base.a2dTopLeft, align=TextNode.ALeft,
                            pos=(0.08, -pos - 0.04), scale=.06)

    def addTitle(self, text):
        return OnscreenText(text=text, style=1, pos=(-0.1, 0.09), scale=.08,
                            parent=base.a2dBottomRight, align=TextNode.ARight,
                            fg=(1, 1, 1, 1), shadow=(0, 0, 0, 1))

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
IMG_EXAMPLE1 = 'marker1.jpg'
VID_EXAMPLE1 = 'marker_vid.mp4'
CAM_INDEX = 0

def main():
    master = Master()
    #master.start(IMG_MODE, IMG_EXAMPLE1)
    master.start(VID_MODE, VID_EXAMPLE1)
    #master.start(CAM_MODE, CAM_INDEX)
    master.cleanup()

if __name__ == "__main__":
    main()
