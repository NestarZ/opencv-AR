 #!/usr/bin/python
 # -*- coding: utf-8 -*-

import cv2
import numpy as np
import json

import random
from panda3d.core import *
loadPrcFileData("", "window-title AR Mathieu & Elias")
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

    def find(self, img_orig):
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
        M = cv2.getPerspectiveTransform(corners, ideal_corners)
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
            corners = self.curve_to_quadrangle(positions[i])
            sorted_corners = self.sort_corners(corners)
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

        if self.nb_current_markers != len(detected_markers) or self.nb_current_quadrangles != len(positions):
            self.nb_current_markers = len(detected_markers)
            self.nb_current_quadrangles = len(positions)

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
    dict_mesh = {0:'lantern', 1:'lantern', 2:'wall_S'}
    path = '../media/mesh/'
    def __init__(self, id_=0):
        self.obj_name = MeshController.dict_mesh[id_]
        self.obj = loader.loadModel(MeshController.path + self.obj_name)
        self.time_hidden = 0

    def setPosScale(self, pos3d, scale, rot,relative=None):
        if relative:
            self.obj.setPos(relative, pos3d[0],pos3d[1],pos3d[2])
        else:
            self.obj.setPos(pos3d[0],pos3d[1],pos3d[2])
        self.obj.setScale(scale)

        self.obj.setHpr(rot[0],rot[1],rot[2])

    def reparentTo(self, parent):
        self.obj.reparentTo(parent)

    def hide(self, time):
        self.time_hidden += 1
        if self.time_hidden > time:
            self.obj.hide()

    def show(self):
        self.time_hidden = 0
        self.obj.show()

class VideoTexture:
    def __init__(self, capture):
        self.capture = capture
        self.image = None

    def draw_3d_axis(self, corners, imgpts):
        # draw 3d axis on each markers
        corner = tuple(corners[0].ravel())
        self.image = cv2.line(self.image, corner, tuple(imgpts[0].ravel()), (255,0,0), 5)
        self.image = cv2.line(self.image, corner, tuple(imgpts[1].ravel()), (0,255,0), 5)
        self.image = cv2.line(self.image, corner, tuple(imgpts[2].ravel()), (0,0,255), 5)

    def get_cv_img(self, img):
        img = cv2.flip(img, 0)
        w, h = img.shape[0:2]
        tex = Texture("webcam_img")
        tex.setup2dTexture(h, w, Texture.TUnsignedByte, Texture.FRgb)
        tex.setRamImage(img)
        return tex

    def get_tex(self):
        #return current frame as a panda3d texture
        return self.get_cv_img(self.get_image())

    def get_image(self):
        return self.image

    def update(self):
        image = self.capture.get_frame()
        if image is not None:
            self.image = image

class World(ShowBase):
    def __init__(self, capture):
        ShowBase.__init__(self)
        self.run = run
        self.obj_list = {}
        self.ui = UserInterface()
        self.marker_detector = MarkerDetector()
        self.video_texture = VideoTexture(capture)
        self.caption1 = self.addCaption(0.30, "")
        self.title = self.addTitle("AR Mathieu & Elias")
        self.videoNode = self.render.attachNewNode("World")
        self.objNode = self.videoNode.attachNewNode("Objets")

    def set_camera(self, lens_sizeX, lens_sizeY):
        camera.setPos(0.0, -60.0, 0)
        camera.lookAt(0.0, 0.0, 0.0)
        lens = OrthographicLens()
        lens.setFilmSize(lens_sizeX, lens_sizeY)
        base.cam.node().setLens(lens)

    def set_light(self, light_type, name, color, pos=None):
        light = light_type(name)
        light.setColor(VBase4(*color))
        light_np = self.objNode.attachNewNode(light)
        if pos: light_np.setPos(*pos)
        self.objNode.setLight(light_np)

    def create_video_card(self, tex):
        cm = CardMaker("VideoCard")
        cm.setFrameFullscreenQuad()
        cm.setUvRange(tex)
        return self.videoNode.attachNewNode(cm.generate())

    def start(self):
        self.set_camera(2, 2)
        self.set_light(AmbientLight, "alight", (0.2, 0.2, 0.2, 1))
        self.set_light(PointLight, "plight", (0.5, 0.5, 0.5, 1), (10, -60, 0))
        self.video_texture.update()
        tex = self.video_texture.get_tex()
        self.video_card = self.create_video_card(tex)
        self.video_card.setTexture(tex)
        taskMgr.add(self.update_video_texture, "update_video_texture")
        taskMgr.add(self.update_mesh, "update_mesh")
        base.disableMouse()

    def get_detected_markers(self):
        return self.marker_detector.find(self.video_texture.get_image())

    def update_video_texture(self, task):
        self.video_texture.update()
        updated_tex = self.video_texture.get_tex()
        self.video_card.setTexture(updated_tex)
        return task.cont

    def update_mesh(self, task):
        def convertPosMarkerToPosWorld(pos):
            """
            return the position where the mesh should be in world
            using the parameters 'pos' sent by cv2
            """
            def euclid(x,y):
                return abs(x-y)

            #calculate position for mesh
            center = [0,0]
            for elem in pos:
                center[0] += elem[0]/4
                center[1] += elem[1]/4

            x = (center[0] - 320) / 320.0
            y = 0
            z = (center[1] - 180) / -180.0

            echo = (center, pos, x,z)

            #adaptative scale, depending on how far the marker is
            #the further the smaller
            dx12 = euclid(pos[1][0],pos[2][0])
            dx03 = euclid(pos[3][0],pos[0][0])
            dy12 = euclid(pos[1][1],pos[2][1])
            dy03 = euclid(pos[3][1],pos[0][1])
            scale = dx12+dx03+dy12+dy03
            scale = scale/960

            #calculate Rotation for mesh
            rotX = 0
            rotY = 25
            rotZ = 0

            return (x,y,z,scale, rotX, rotY, rotZ)

        self.markers = self.get_detected_markers()
        if self.ui.exit: pass
        for mesh in self.obj_list.values():
            mesh.hide(3) # 3 frames until hide (prevent blink)
        for (id_obj, pos) in self.markers.items():
            if id_obj not in self.obj_list:
                self.obj_list[id_obj] = MeshController(id_obj)
                self.obj_list[id_obj].reparentTo(self.objNode)
            self.obj_list[id_obj].show()
            #Need converter to stick to position, deal with rotation
            (x, y, z, scale, rx, ry, rz) = convertPosMarkerToPosWorld(pos)
            self.obj_list[id_obj].setPosScale((x, y, z), scale, (rx,ry,rz), self.videoNode)
        return task.cont

    def addCaption(self, pos, msg):
        return OnscreenText(text=msg, style=2, fg=(1, 1, 1, 1),
                            parent=base.a2dTopLeft, align=TextNode.ALeft,
                            pos=(0.08, - pos - 0.04), scale=.1)

    def addTitle(self, text):
        return OnscreenText(text=text, style=1, pos=(-0.1, 0.09), scale=.08,
                            parent=base.a2dBottomRight, align=TextNode.ARight,
                            fg=(1, 1, 1, 1), shadow=(0, 0, 0, 1))

class Master:
    def __init__(self):
        self.world = object()
        self.capture = object()
        self.mode = 0

    def start(self, mode, material_id):
        self.mode = mode
        if mode == VID_MODE:
            self.capture = VideoController(material_id)
        elif mode == CAM_MODE:
            self.capture = CamController(material_id)
        elif mode == IMG_MODE:
            self.capture = ImageController(material_id)
        self.capture.open()
        self.world = World(self.capture)
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
