import cv2
from imutils import face_utils
from scipy.spatial import distance as dist
import dlib

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
MOUTH_AR_THRESH = 0.50
(mStart, mEnd) = (48, 68)
counter = 0
total = 0
kesinti = 0
yawn_counter = 0
rate = 0
ctrl = 0
mouthHull = None
shape = None
mar = 0


def mouth_aspect_ratio(mouth):
    A = dist.euclidean(mouth[2], mouth[10])  # 51, 59
    B = dist.euclidean(mouth[4], mouth[8])  # 53, 57
    C = dist.euclidean(mouth[0], mouth[6])  # 49, 55

    mar = (A + B) / (2.0 * C)

    return mar


def center(x1, y1, x2, y2):
    return int((x1 + x2) / 2), int((y1 + y2) / 2)


def calcDistance(mouth):
    centerOfMouth = center(mouth[0][0], mouth[0][1], mouth[6][0], mouth[6][1])
    distance1 = dist.euclidean(centerOfMouth, mouth[3])
    distance2 = dist.euclidean(centerOfMouth, mouth[9])
    h1 = distance1 / (distance1 + distance2)
    h2 = distance2 / (distance1 + distance2)
    return abs(h1 - h2)
""""
h1-h2 / h1+h2
"""

def extract_face(frame):
    faces = detector(frame)
    if len(faces) > 0:
        return faces[0]
    return -1


def calcMar(frame):
    global ctrl
    global mouthHull
    global shape
    global mar
    rect = extract_face(frame)
    if rect is not -1:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        mouth = shape[mStart:mEnd]
        mouthMAR = mouth_aspect_ratio(mouth)
        ctrl = calcDistance(mouth)
        mouthHull = cv2.convexHull(mouth)
        mar = mouthMAR
        cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)
        cv2.putText(frame, "MAR: {:.2f}".format(mar), (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        #cv2.putText(frame, "ctrl: {:}".format(ctrl), (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "Yawn: " + str(yawn_counter), (10, 210),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return mar
    return -1



def yawnDetect(frame):
    global total
    global counter
    global yawn_counter
    global kesinti
    global shape

    mar = calcMar(frame)
    if mar is not -1:
        total += 1
        if mar > MOUTH_AR_THRESH and ctrl < 0.55:
            counter += 1
            kesinti = 0
        else:
            kesinti += 1
            if kesinti > 2:
                if total > 5:
                    if counter / total > 0.60:
                        yawn_counter += 1
                total = 0
                counter = 0
                kesinti = 0

        for (x, y) in shape:
            cv2.circle(frame, (x, y), 1, (0, 255, 0), 1)
