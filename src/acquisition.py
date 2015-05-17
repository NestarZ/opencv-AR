 #!/usr/bin/python
 # -*- coding: utf-8 -*-

import cv2
import numpy as np
import json
import glob
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
        try:
            cv2.imshow(caption, img)
        except:
            print "Empty image, can't display ({})".format(caption)
        if coordxy:
            assert isinstance(coordxy, tuple) and len(coordxy) == 2
            assert all(isinstance(coord, int) for coord in coordxy)
            cv2.moveWindow(caption, coordxy[0], coordxy[1])

class Capture(object):
    DISPLAY_SIZE = (640,360)
    def __init__(self, material_id, device_name):
        """ material_id - Filename or Camera Index """
        self.material_id = material_id
        self.device_name = device_name
        self.img = None

    def open(self):
        pass

    def get_frame(self):
        pass

    def kill(self):
        pass

class ImageController(Capture):
    def __init__(self, material_id, device_name, **kwargs):
        super(ImageController, self).__init__(material_id, device_name)
        self.path = '../media/img/{}/'.format(device_name) if not kwargs.get('absolute', False) else ''
        self.batch = kwargs.get('batch', False)
        self.regex = kwargs.get('regex', '*.jpg')

    def open(self):
        file_path = self.path + self.material_id
        if self.batch:
            self.pack = glob.glob(file_path+self.regex)
        else:
            self.img = cv2.imread(file_path)
            self.img = cv2.resize(self.img, Capture.DISPLAY_SIZE, interpolation=cv2.INTER_AREA)
            if self.img is None:
                raise Exception('No image to read')

    def get_frame(self):
        if self.batch and not self.pack:
            print 'No more image to analyze in {}'.format(self.path+self.material_id)
            return None
        if self.batch:
            self.img = cv2.resize(cv2.imread(self.pack.pop()), Capture.DISPLAY_SIZE, interpolation=cv2.INTER_AREA)
        return self.img

class VideoController(Capture):
    def __init__(self, material_id, device_name):
        super(VideoController, self).__init__(material_id, device_name)
        self.path = '../media/video/{}/'.format(device_name)

    def open(self):
        file_path = self.path + self.material_id
        self.device = cv2.VideoCapture(file_path)
        print(file_path, self.path, self.material_id)

    def get_frame(self):
        ret, image = self.device.read()
        if (ret == False): # failed to capture
            print("Fail or end of capture.")
            return None
        self.img = cv2.resize(image, Capture.DISPLAY_SIZE, interpolation=cv2.INTER_AREA)
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
                    sorted_corners = self.rotate(sorted_corners, -90) # Dont work, il faudrait echanger les positions
                else:
                    detected_markers[id_] = sorted_corners.tolist()
                    #self.ui.display('marker_{} ({})'.format(id_, j), img, (i*300, 800))
                    break

        if self.nb_current_markers != len(detected_markers) or self.nb_current_quadrangles != len(positions):
            self.nb_current_markers = len(detected_markers)
            self.nb_current_quadrangles = len(positions)

        return detected_markers

    def rotate(self, corners, angle):
        assert angle in (-90,90,180,270), "Not an angle, need to be -90,90,180,270"

        if angle == 90:
            rotation_vec = (3,1,2,4) 
        elif angle == 180:
            rotation_vec = (4,3,2,1)
        else:
            rotation_vec = (2,4,1,3)

        rotated = [corners[i-1] for i in rotation_vec]
        print(rotated)

        return np.array(rotated,'f')


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
    dict_mesh = {0:'lantern', 1:'lantern', 2:'wall_S', 3:'plane'}
    path = '../media/mesh/'
    def __init__(self, id_=0):
        id_ = 3
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

class Camera:
  def __init__(self,P):
    """ Initialize P = K[R|t] camera model. """
    self.P = P
    self.K = None # calibration matrix
    self.R = None # rotation
    self.t = None # translation
    self.c = None # camera center


  def project(self,X):
    """  Project points in X (4*n array) and normalize coordinates. """

    x = dot(self.P,X)
    for i in range(3):
      x[i] /= x[2]
    return x

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
        if self.image is not None:
            return self.get_cv_img(self.get_image())
        return None

    def get_image(self):
        return self.image

    def update(self):
        self.image = self.capture.get_frame()

class World(ShowBase):
    def __init__(self, capture, camera_param):
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
        self.camera_matrix, self.distortion_coeffs = camera_param
        self.marker_corners3d = []
        self.marker_corners3d.append((-0.5,-0.5,0))
        self.marker_corners3d.append((+0.5,-0.5,0))
        self.marker_corners3d.append((+0.5,+0.5,0))
        self.marker_corners3d.append((-0.5,+0.5,0))
        self.marker_corners3d = np.array(self.marker_corners3d)

    def get_marker3d_corners(self):
        return self.marker_corners3d

    def set_camera(self, lens_sizeX, lens_sizeY):
        camera.setPos(0.0, -3.75, 0)
        camera.lookAt(0.0, 0.0, 0.0)
        #lens = OrthographicLens() #pos is -60
        lens = PerspectiveLens()
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
        if updated_tex is not None:
            self.video_card.setTexture(updated_tex)
            return task.cont
        return None

    def update_mesh(self, task):
        def convertPosMarkerToPosWorld(obj, pos):
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
            y = -0.4
            z = (center[1] - 180) / -180.0


            pos = np.array(pos)
            marker3d_corners = self.get_marker3d_corners()
            _ret, rot, trans = cv2.solvePnP(marker3d_corners, pos, self.camera_matrix, self.distortion_coeffs)

            #adaptative scale, depending on how far the marker is
            #the further the smaller
            dx12 = euclid(pos[1][0],pos[2][0])
            dx03 = euclid(pos[3][0],pos[0][0])
            dy12 = euclid(pos[1][1],pos[2][1])
            dy03 = euclid(pos[3][1],pos[0][1])
            scale = dx12+dx03+dy12+dy03
            scale = scale/960
            scale = 0.2


            #calculate Rotation for mesh

            return (x,y,z,scale, rot[0], rot[1], rot[2])

        self.markers = self.get_detected_markers()
        if self.ui.exit: pass
        for mesh in self.obj_list.values():
            mesh.hide(4) # 4 frames until hide (prevent blink)
        for (id_obj, pos) in self.markers.items():
            if id_obj not in self.obj_list:
                self.obj_list[id_obj] = MeshController(id_obj)
                self.obj_list[id_obj].reparentTo(self.objNode)
            self.obj_list[id_obj].show()
            #Need converter to stick to position, deal with rotation
            (x, y, z, scale, rx, ry, rz) = convertPosMarkerToPosWorld(self.obj_list[id_obj].obj, pos)
            self.obj_list[id_obj].setPosScale((x, y, z), scale, (rx,ry,rz), self.videoNode)
        return task.cont

    def estimate_camera_position(self):
        cv2.solvePnP(self.markers, imagePoints, self.camera_matrix, self.distortion_coeffs, rvec, vec, useExtrinsicGuess=False)

    def addCaption(self, pos, msg):
        return OnscreenText(text=msg, style=2, fg=(1, 1, 1, 1),
                            parent=base.a2dTopLeft, align=TextNode.ALeft,
                            pos=(0.08, - pos - 0.04), scale=.1)

    def addTitle(self, text):
        return OnscreenText(text=text, style=1, pos=(-0.1, 0.09), scale=.08,
                            parent=base.a2dBottomRight, align=TextNode.ARight,
                            fg=(1, 1, 1, 1), shadow=(0, 0, 0, 1))

class CameraCalibration:
    def __init__(self, device_name):
        self.device_name = device_name
        self.calibration_data = None
        self.criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 30, 0.1)
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        self.objp = np.zeros((6*7,3), np.float32)
        self.objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)
        # Arrays to store object points and image points from all the images.
        self.objpoints = [] # 3d point in real world space
        self.imgpoints = [] # 2d points in image plane.

    def get_calibration_data(self, camera_name):
        try:
            # Load previously saved data
            print("Recherche de données de calibration de la camera ({}) en cours...".format(camera_name))
            with np.load('{}.npz'.format(camera_name)) as X:
                mtx, dist, _, _ = [X[i] for i in ('mtx','dist','rvecs','tvecs')]
            print("Données de calibration de la camera ({}) trouvées !".format(camera_name))
            return True, (mtx, dist)
        except:
            print("Aucune données de calibration n'a été trouvé.".format(camera_name))
            return False, (None, None)

    def add_chessboard_points(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Arrays to store object points and image points from all the images.
        object_corners = [] # 3d point in real world space
        image_corners = [] # 2d points in image plane.
        # 3D Scene Points
        chessboard_size = (9, 6)
        for i in range(chessboard_size[0]):
            for j in range(chessboard_size[1]):
                object_corners.append((i, j, 0.0))

        successes = 0

        # Get the chessboard corners
        found, image_corners = cv2.findChessboardCorners(gray, chessboard_size, np.array(image_corners))
        # Get subpixel accuracy on the corners
        if found:
            image_corners = cv2.cornerSubPix(gray, np.array(image_corners), (5,5), (-1,-1), self.criteria)
            #If we have a good board, add it to our data
            if len(image_corners) == chessboard_size[0]*chessboard_size[1]:
                # Add image and scene points from one view
                # 2D image points from one view
                self.imgpoints.append(image_corners)
                # corresponding 3D scene points
                self.objpoints.append(object_corners)
                successes += 1
                cv2.drawChessboardCorners(image, chessboard_size, image_corners, found)
                UserInterface.display('chessboard_found', image)
                cv2.waitKey(1)
        return successes

    def calibration(self, shape):
        print "Calibration ongoing... "
        ret, mtx, dist, rvecs, tvecs  = cv2.calibrateCamera(np.array(self.objpoints).astype(np.float32), np.array(self.imgpoints).astype(np.float32), shape, None,None)
        print "Calibration OK"
        return mtx, dist, rvecs, tvecs

    def save(self, camera_name, **kwargs):
        with open("{}.npz".format(camera_name), 'w') as outfile:
            print("Sauvegarde des parametres")
            np.savez(outfile, **kwargs)

    def run(self):
        found, self.calibration_data = self.get_calibration_data(self.device_name)
        if not found:
            #capture = ImageController("/home/elias/OpenCV/opencv/samples/data/left", self.device_name, batch=True, absolute=True) # opencv chessboard
            capture = ImageController("chessboard/", self.device_name, batch=True, regex='*') # laptop camera chessboard (img)
            #capture = ImageController("chessboard/", self.device_name, batch=True, regex='*.png') # laptop camera chessboard (img)
            #capture = VideoController("chessboard/chessboard_vid_01.mp4", self.device_name) # sony camera chessboard (video)
            capture.open()
            image = capture.get_frame()
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            shape = gray.shape[::-1]
            calib_iter, i = 0, 0
            while image is not None and calib_iter < 500:
                successes = self.add_chessboard_points(image)
                calib_iter += successes
                i += 1
                print "Calibration sucess rate : {}% ({})".format(100*calib_iter/i, i)
                image = capture.get_frame()
            mtx, dist, rvecs, tvecs = self.calibration(shape)
            assert (mtx, dist, rvecs, tvecs) != (None,)*4
            self.save(self.device_name, mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)
            self.calibration_data = mtx, dist

    def get_data(self):
        assert self.calibration_data is not None, "No data"
        print("Parametres de la camera :\n>> camera_matrix={}\n>> distortion_coeffs={}".format(self.calibration_data[0], self.calibration_data[1]))
        return self.calibration_data

class PoseEstimation:
    def __init__(self, capture, params):
        self.capture = capture
        self.marker_detector = MarkerDetector()
        self.camera_matrix, self.distortion_coeffs = params
        #self.camera_matrix, self.distortion_coeffs = (np.array([[ 859.87780302,0.,682.3173388 ],[0.,799.06029505,382.57139684],[   0.        ,    0.        ,    1.        ]]), np.array([[ 0.2529905 , -0.12695403, -0.01182944, -0.02478326, -0.48389559]]))
        self.marker_corners3d = []
        self.marker_corners3d.append((-0.5,-0.5,0))
        self.marker_corners3d.append((+0.5,-0.5,0))
        self.marker_corners3d.append((-0.5,+0.5,0))
        self.marker_corners3d.append((+0.5,+0.5,0))
        self.marker_corners3d = np.array(self.marker_corners3d)
        self.marker_corners3d = self.marker_corners3d.astype(np.float32)
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        self.axis = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)

    def run(self):
        while True:
            image = self.capture.get_frame()
            if image is not None:
                markers = self.marker_detector.find(image)
                if markers:
                    self.estimate(image, markers)

    def refine_corners(self, gray, markers_corners):
        precise_corners = []
        for corners in markers_corners:
            for c in range(4):
                precise_corners.append(np.array([corners[c]]))
        a = np.array(precise_corners).astype(np.float32)
        precise_corners = cv2.cornerSubPix(gray, a, (6,6), (-1,-1), self.criteria)
        for i, corners in enumerate(markers_corners):
            for c in range(4):
                corners[c] = np.array(precise_corners[i*4 + c])
        return markers_corners

    def estimate(self, image, markers):
        gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        markers_corners = np.array([points for points in markers.values()])
        markers_corners = self.refine_corners(gray, markers_corners)
        # Find the rotation and translation vectors.
        objp = self.marker_corners3d
        for marker_corners in markers_corners:
            _, rvecs, tvecs = cv2.solvePnP(objp, marker_corners, self.camera_matrix, self.distortion_coeffs)
            # project 3D points to image plane
            imgpts, jac = cv2.projectPoints(self.axis, rvecs, tvecs, self.camera_matrix, self.distortion_coeffs)
            try:
                image = self.draw(image,marker_corners,imgpts)
            except:
                pass
        UserInterface.display('pose_estimation', image)
        cv2.waitKey(1)

    def draw(self, img, corners, imgpts):
        corner = tuple(corners[0].ravel().astype(int))
        img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 5)
        img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 5)
        img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 5)
        return img

class Master:
    def __init__(self):
        self.world = object()
        self.capture = object()
        self.mode = 0

    def start(self, device, mode, material_id):
        self.mode = mode
        if mode == VID_MODE:
            self.capture = VideoController(material_id, device)
        elif mode == CAM_MODE:
            self.capture = CamController(material_id, device)
        elif mode == IMG_MODE:
            self.capture = ImageController(material_id, device)
        self.capture.open()
        self.calibration = CameraCalibration(device)
        self.calibration.run()
        params = self.calibration.get_data()
        #self.world = World(self.capture, params)
        #self.world.start()
        #self.world.run()
        self.pose_estimation = PoseEstimation(self.capture, params)
        self.pose_estimation.run()

    def cleanup(self):
        """ Closes all OpenCV windows and releases video capture device. """
        cv2.destroyAllWindows()
        self.capture.kill()

# Modes
CAM_MODE = 1
VID_MODE = 2
IMG_MODE = 3
# Devices Source
DEVICE_NAME = "sony_camera"
#DEVICE_NAME = "laptop_camera"
# Devices Type
IMG_FILE = 'marker1.jpg'
VID_FILE = 'markers/marker_vid_01.mp4'
CAM_INDEX = 0

def main():
    #x = CameraPosition()
    #return
    master = Master()
    #master.start(DEVICE_NAME, IMG_MODE, IMG_FILE)
    master.start(DEVICE_NAME, VID_MODE, VID_FILE)
    #master.start(DEVICE_NAME, CAM_MODE, CAM_INDEX)
    master.cleanup()

if __name__ == "__main__":
    main()
