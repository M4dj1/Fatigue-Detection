import cv2
import dlib
from math import hypot

def frameps():
    (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
    if int(major_ver) < 3:
        fps = cap.get(cv2.cv.CV_CAP_PROP_FPS)
        print("Frames per second : {0}".format(fps))
    else:
        fps = cap.get(cv2.CAP_PROP_FPS)
        print("Frames per second : {0}".format(fps))
    return fps

def blinking_ratio(eyeP, mark):
    leftP = (mark.part(eyeP[0]).x, mark.part(eyeP[0]).y)
    rightP = (mark.part(eyeP[3]).x, mark.part(eyeP[3]).y)
    top = midpoint(mark.part(eyeP[1]), mark.part(eyeP[2]))
    bottom = midpoint(mark.part(eyeP[5]), mark.part(eyeP[4]))
    hor_line = hypot((leftP[0] - rightP[0]), (leftP[1] - rightP[1]))
    ver_line = hypot((top[0] - bottom[0]), (top[1] - bottom[1]))
    ratio = hor_line / ver_line
    return ratio

def midpoint(p1 ,p2):
    return int((p1.x + p2.x)/2), int((p1.y + p2.y)/2)


counter = 0
font = cv2.FONT_HERSHEY_SIMPLEX
cap = cv2.VideoCapture(0)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
fps = frameps()

while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    for face in faces:
        landmarks = predictor(gray, face)

        leftP = (landmarks.part(36).x, landmarks.part(36).y)
        rightP = (landmarks.part(39).x, landmarks.part(39).y)
        top = midpoint(landmarks.part(37), landmarks.part(38))
        bottom = midpoint(landmarks.part(41), landmarks.part(40))

        leftP2 = (landmarks.part(42).x, landmarks.part(42).y)
        rightP2 = (landmarks.part(45).x, landmarks.part(45).y)
        top2 = midpoint(landmarks.part(43), landmarks.part(44))
        bottom2 = midpoint(landmarks.part(47), landmarks.part(46))

        hor_line = cv2.line(frame, leftP, rightP, (0, 0, 255), 2)
        ver_line = cv2.line(frame, top, bottom, (0, 0, 255), 2)

        hor_line2 = cv2.line(frame, leftP2, rightP2, (0, 0, 255), 2)
        ver_line2 = cv2.line(frame, top2, bottom2, (0, 0, 255), 2)

        left_eye_ratio = blinking_ratio([36, 37, 38, 39, 40, 41], landmarks)
        right_eye_ratio = blinking_ratio([42, 43, 44, 45, 46, 47], landmarks)
        ratio = (left_eye_ratio + right_eye_ratio) / 2

    if (ratio > 4):
        counter = counter + 1
        cv2.putText(frame, "Blinking", (1, 25), font, 1, (255, 0, 0), 2)
        print(counter)
        if (counter >= fps):
            cv2.putText(frame, "Alert!!", (50, 150), font, 5, (0, 0, 255), 15)
    else:
        counter = 0

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1)
    if key == 27:
        break
cap.release()
cv2.destroyAllWindows()