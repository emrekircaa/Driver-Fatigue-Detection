from scipy.spatial import distance as dist
from imutils import face_utils
import time
import dlib
import cv2

class perclos:
    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        self.COUNTER = 0
        self.TOTAL = 0
        self.perclos_value = 0
        self.ear = 0
        self.EYE_AR_THRESH = 0.24
        (self.lStart, self.lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (self.rStart, self.rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
        self.leftEye = -1
        self.rightEye = -1
        self.start = time.time()

    @staticmethod
    def eye_aspect_ratio(eye):
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])
        C = dist.euclidean(eye[0], eye[3])
        ear = (A + B) / (2.0 * C)
        return ear

    def load_img(self, img):
        self.img = img
        self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

    def detect_faces(self):
        faces = self.detector(self.img)
        if len(faces) > 0:
            return faces
        self.leftEye = -1
        self.rightEye = -1
        return -1

    def calc_ear(self):
        faces = self.detect_faces()
        if type(faces) is not int :
            shape = self.predictor(self.img, faces[0])
            shape = face_utils.shape_to_np(shape)

            self.leftEye = shape[self.lStart:self.lEnd]
            self.rightEye = shape[self.rStart:self.rEnd]
            leftEAR = perclos.eye_aspect_ratio(self.leftEye)
            rightEAR = perclos.eye_aspect_ratio(self.rightEye)

            self.ear = (leftEAR + rightEAR) / 2.0
            self.TOTAL += 1
            if self.ear >= self.EYE_AR_THRESH:
                self.COUNTER += 1
        return self.ear


    def calc_perclos(self):
        end = time.time()
        if end - self.start >= 10:
            if(self.TOTAL != 0):
                self.perclos_value = ((self.TOTAL-self.COUNTER)/self.TOTAL)*100
                self.TOTAL = 0
                self.COUNTER = 0
                self.start = time.time()
        return self.perclos_value



