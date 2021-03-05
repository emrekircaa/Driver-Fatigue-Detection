import cv2
import dlib
import time
import numpy as np

class DetectSlope:
    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        self.start = time.time()

    def load_img(self, img):
        self.img = img

    def detect_faces(self, img):
        faces = self.detector(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        if len(faces) > 0:
            return faces
        return -1

    @staticmethod
    def get_line_size(p1, p2):
        ls = ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** (1 / 2)
        return ls

    def get_angle(self, p1, p2, p3):
        line1 = self.get_line_size(p1, p2)
        line2 = self.get_line_size(p1, p3)
        line3 = self.get_line_size(p2, p3)
        angle = np.arccos((line3 ** 2 - line1 ** 2 - line2 ** 2) / (-2 * line1 * line2))
        return angle

    @staticmethod
    def center(x1, y1, x2, y2):
        return int((x1 + x2) / 2), int((y1 + y2) / 2)

    def find_slop(self):
        faces = self.detect_faces(self.img)
        if faces != -1:
            for face in faces:
                cv2.rectangle(self.img, (face.left(), face.top()), (face.right(), face.bottom()), (0, 255, 255), 2)
                shape = self.predictor(self.img, face)
                left_x1 = shape.part(37).x
                left_x2 = shape.part(40).x

                left_y1 = shape.part(38).y
                left_y2 = shape.part(41).y

                right_x1 = shape.part(43).x
                right_x2 = shape.part(46).x

                right_y1 = shape.part(44).y
                right_y2 = shape.part(47).y

                nose = (shape.part(34).x, shape.part(34).y)

                left_eye = self.center(left_x1, left_y1, left_x2, left_y2)
                right_eye = self.center(right_x1, right_y1, right_x2, right_y2)

                center_of_eyes = self.center(left_eye[0], left_eye[1], right_eye[0], right_eye[1])
                center_of_top = self.center(face.left(), face.top(), face.right(), face.top())

                angle = self.get_angle(nose, center_of_eyes, center_of_top)
                angle = np.degrees(angle)

                return angle
        return 0
