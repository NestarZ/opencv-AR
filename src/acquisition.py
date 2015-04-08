import cv2
import numpy

class UserInterface:
    def __init__(self):
        pass

    @property
    def exit(self):
        return cv2.waitKey(1) >= 0

    @classmethod
    def display(cls, caption, img):
        cv2.imshow(caption, img)

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
            return None
        return self.img

    def kill(self):
        """ Releases video capture device. """
        cv2.VideoCapture(self.material_id).release()

class MarkerDetector:
    def __init__(self):
        pass

    def __call__(self, img_orig):
        w, h = img_orig.shape[1], img_orig.shape[0]
        img_gray = self.rgb_to_gray(img_orig)
        img_threshold = self.do_threshold(img_gray)
        contours = self.find_contours(img_threshold)
        img_contours = numpy.zeros((h, w, 3), numpy.uint8)
        img_contours = cv2.drawContours(img_contours, contours, -1, (255,255,255), 2)
        return img_contours

    def rgb_to_gray(self, img):
        """ Converts an RGB image to grayscale, where each pixel
        now represents the intensity of the original image. """
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    def do_threshold(self, image):
        #im_bw = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 7, 7)
        (thres, im_thres) = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)
        return im_thres

    def find_contours(self, threshold_img):
        (threshold_img, all_contours, hierarchy) = cv2.findContours(threshold_img,
            mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_NONE)
        contours = []
        for c in all_contours:
            # Approxime par un polynome
            approx_curve = cv2.approxPolyDP(c, c.size*.05, True)
            if not cv2.isContourConvex(approx_curve):
                continue
            #if approx_curve.size == 4:
                #continue
            if c.size < 70:
                continue
            contours.append(c)
        return contours

    def display(self, caption, img):
        cv2.imshow(caption, img)

class Master:
    def __init__(self):
        self.ui = UserInterface()
        self.markerdetector = MarkerDetector()

    def start(self, mode, material_id):
        if mode == VID_MODE or mode == CAM_MODE:
            self.capture = VideoCapture(material_id)
        elif mode == IMG_MODE:
            self.capture = ImageCapture(material_id)
        self.capture.open()
        self.main_loop()

    def main_loop(self):
        while not self.ui.exit:
            img = self.capture.get_frame()
            if img is None: break
            self.ui.display('raw', img)
            result_img = self.markerdetector(img)
            self.ui.display('result', result_img)

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
    master.start(IMG_MODE, IMG_EXAMPLE1)
    #master.start(VID_MODE, VID_EXAMPLE1)
    #master.start(CAM_MODE, CAM_INDEX)
    master.cleanup()

if __name__ == "__main__":
    main()
