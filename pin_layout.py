# import the necessary packages
from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
import cv2
import numpy as np
from math import hypot
import time
import dlib
import imutils
import pyttsx3
import random
import math

auth = 0
OTP = ""

names = ['krishna','2','3','4','5'] #replace with ur names in order

# One time initialization
engine = pyttsx3.init()

# Set properties _before_ you add things to say
engine.setProperty('rate', 125)    # Speed percent (can go over 100)
engine.setProperty('volume', 1)  # Volume 0-1

def eye_aspect_ratio(eye):
	# compute the euclidean distances between the two sets of
	# vertical eye landmarks (x, y)-coordinates
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])

	# compute the euclidean distance between the horizontal
	# eye landmark (x, y)-coordinates
	C = dist.euclidean(eye[0], eye[3])

	# compute the eye aspect ratio
	ear = (A + B) / (2.0 * C)

	# return the eye aspect ratio
	return ear
 
# define two constants, one for the eye aspect ratio to indicate
# blink and then a second constant for the number of consecutive
# frames the eye must be below the threshold for to set off the
# alarm
EYE_AR_THRESH = 0.25
EYE_AR_CONSECUTIVE_FRAMES = 8
WINK_AR_DIFF_THRESH = 0.01
WINK_AR_CLOSE_THRESH = 0.25
WINK_CONSECUTIVE_FRAMES = 10

COUNTER = 0
WINK_COUNTER = 0

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# grab the indexes of the facial landmarks for the left and
# right eye, respectively
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# start the video stream thread


# Keyboard settings
keyboard = np.zeros((400, 1000, 3), np.uint8)
keys_set_1 = {0: "0", 1: "1", 2: "2", 3: "3", 4: "4",
              5: "5", 6: "6", 7: "7", 8: "8", 9: "9",}

user_key = ""

def generateOTP() :
    global OTP
    # Declare a digits variable 
    # which stores all digits
    digits = "0123456789"
    OTP = ""
 
   # length of password can be changed
   # by changing value in range
    for i in range(4) :
        OTP += digits[math.floor(random.random() * 10)]
    print("Generated OTP is:",OTP)
    return OTP

def draw_letters(letter_index, text, letter_light):
    # Keys
    if letter_index == 0:
        x = 0
        y = 0
    elif letter_index == 1:
        x = 200
        y = 0
    elif letter_index == 2:
        x = 400
        y = 0
    elif letter_index == 3:
        x = 600
        y = 0
    elif letter_index == 4:
        x = 800
        y = 0
    elif letter_index == 5:
        x = 0
        y = 200
    elif letter_index == 6:
        x = 200
        y = 200
    elif letter_index == 7:
        x = 400
        y = 200
    elif letter_index == 8:
        x = 600
        y = 200
    elif letter_index == 9:
        x = 800
        y = 200

    width = 200
    height = 200
    th = 3 # thickness

    # Text settings
    font_letter = cv2.FONT_HERSHEY_PLAIN
    font_scale = 10
    font_th = 4
    text_size = cv2.getTextSize(text, font_letter, font_scale, font_th)[0]
    width_text, height_text = text_size[0], text_size[1]
    text_x = int((width - width_text) / 2) + x
    text_y = int((height + height_text) / 2) + y

    if letter_light is True:
        cv2.rectangle(keyboard, (x + th, y + th), (x + width - th, y + height - th), (255, 255, 255), -1)
        cv2.putText(keyboard, text, (text_x, text_y), font_letter, font_scale, (51, 51, 51), font_th)
    else:
        cv2.rectangle(keyboard, (x + th, y + th), (x + width - th, y + height - th), (51, 51, 51), -1)
        cv2.putText(keyboard, text, (text_x, text_y), font_letter, font_scale, (255, 255, 255), font_th)

# Text and keyboard settings
text = ""
keyboard_selected = "left"
last_keyboard_selected = "left"
select_keyboard_menu = True
keyboard_selection_frames = 0
letter_index = 0

# Counters
frames = 0
letter_index = 0
blinking_frames = 0
frames_to_blink = 6
frames_active_letter = 9
u_otp = 0

def chk_otp():
        global u_otp
        global frames
        global frames
        global letter_index
        global blinking_frames
        global frames_to_blink
        global frames_active_letter
        global EYE_AR_THRESH
        global EYE_AR_CONSECUTIVE_FRAMES
        global WINK_AR_DIFF_THRESH
        global WINK_AR_CLOSE_THRESH
        global WINK_CONSECUTIVE_FRAMES
        global COUNTER
        global WINK_COUNTER
        global user_key
        global OTP
        print("[INFO] Starting Video")
        vs = VideoStream(0).start()
        time.sleep(1.0)
        while True:
            blank_image = np.ones((200,800,3), np.uint8)
            frames += 1
            keyboard[:] = (26, 26, 26)
            keys_set = keys_set_1
            active_letter = keys_set[letter_index]

            
            # Display letters on the keyboard
            if select_keyboard_menu is True:
                if frames == frames_active_letter:
                    letter_index += 1
                    frames = 0

                    #time.sleep(0.5)
                if letter_index == 10:
                    letter_index = 0
                for i in range(10):
                    if i == letter_index:
                        light = True
                    else:
                        light = False
                    draw_letters(i, keys_set[i], light)
            frame = vs.read()
            size = frame.shape
            frame = imutils.resize(frame, width=450)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # detect faces in the grayscale frame
            rects = detector(gray, 0)

            # loop over the face detections
            for rect in rects:
                    # determine the facial landmarks for the face region, then
                    # convert the facial landmark (x, y)-coordinates to a NumPy
                    # array
                    shape = predictor(gray, rect)
                    shape = face_utils.shape_to_np(shape)

                    # extract the left and right eye coordinates, then use the
                    # coordinates to compute the eye aspect ratio for both eyes
                    leftEye = shape[lStart:lEnd]
                    rightEye = shape[rStart:rEnd]
                    leftEAR = eye_aspect_ratio(leftEye)
                    rightEAR = eye_aspect_ratio(rightEye)

                    # average the eye aspect ratio together for both eyes
                    ear = (leftEAR + rightEAR) / 2.0
                    # compute the convex hull for the left and right eye, then
                    # visualize each of the eyes and mouth
                    leftEyeHull = cv2.convexHull(leftEye)
                    rightEyeHull = cv2.convexHull(rightEye)
                    cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
                    cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

                    # check to see if the eye and mouth aspect ratio is below the 
                    # threshold, and if so, increment the blink frame counter
                    if ear < EYE_AR_THRESH:
                            COUNTER += 1
                            # if the eyes were closed for a sufficient number of frames
                            # then sound the alarm
                            if COUNTER >= EYE_AR_CONSECUTIVE_FRAMES:
                                print("eye blinked")
                                #print(type(active_letter))

                                user_key += active_letter
                                engine.say(active_letter)
                                # Flush the say() queue and play the audio
                                engine.runAndWait()
                                COUNTER = 0
                                
                    else:
                            COUNTER = 0
                    cv2.putText(blank_image, user_key, (30, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    if(len(user_key) == 4):
                        if user_key == OTP:
                            print("yes")
                            engine.say("Authorized OTP")
                            # Flush the say() queue and play the audio
                            engine.runAndWait()
                            user_key = ""
                            u_otp = 1
                        else:
                            cv2.putText(blank_image, "Invalid OTP", (200, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                            engine.say("Invalid OTP")
                            # Flush the say() queue and play the audio
                            engine.runAndWait()
                            print("Invalid OTP Try Again")
                            user_key = ""

                    cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.putText(frame, "left eye:{}".format(leftEAR), (270, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.putText(frame, "right eye:{}".format(rightEAR), (270, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)  
                            

            # show the frame
            cv2.imshow("Frame", frame)
            cv2.imshow("Virtual keyboard", keyboard)
            cv2.imshow('Selected Key', blank_image)
            key = cv2.waitKey(160)
            if(u_otp):
                break
            if key == 27:
                break
            if cv2.waitKey(1) & 0xFF == ord('s'):
                user_key += active_letter
        cv2.destroyAllWindows()


def chk_pass():
        global auth
        global frames
        global frames
        global letter_index
        global blinking_frames
        global frames_to_blink
        global frames_active_letter
        global EYE_AR_THRESH
        global EYE_AR_CONSECUTIVE_FRAMES
        global WINK_AR_DIFF_THRESH
        global WINK_AR_CLOSE_THRESH
        global WINK_CONSECUTIVE_FRAMES
        global COUNTER
        global WINK_COUNTER
        global user_key
        print("[INFO] Starting Video")
        vs = VideoStream(0).start()
        time.sleep(1.0)
        while True:
            blank_image = np.ones((200,800,3), np.uint8)
            frames += 1
            keyboard[:] = (26, 26, 26)
            keys_set = keys_set_1
            active_letter = keys_set[letter_index]

            
            # Display letters on the keyboard
            if select_keyboard_menu is True:
                if frames == frames_active_letter:
                    letter_index += 1
                    frames = 0

                    #time.sleep(0.5)
                if letter_index == 10:
                    letter_index = 0
                for i in range(10):
                    if i == letter_index:
                        light = True
                    else:
                        light = False
                    draw_letters(i, keys_set[i], light)
            frame = vs.read()
            size = frame.shape
            frame = imutils.resize(frame, width=450)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # detect faces in the grayscale frame
            rects = detector(gray, 0)

            # loop over the face detections
            for rect in rects:
                    # determine the facial landmarks for the face region, then
                    # convert the facial landmark (x, y)-coordinates to a NumPy
                    # array
                    shape = predictor(gray, rect)
                    shape = face_utils.shape_to_np(shape)

                    # extract the left and right eye coordinates, then use the
                    # coordinates to compute the eye aspect ratio for both eyes
                    leftEye = shape[lStart:lEnd]
                    rightEye = shape[rStart:rEnd]
                    leftEAR = eye_aspect_ratio(leftEye)
                    rightEAR = eye_aspect_ratio(rightEye)

                    # average the eye aspect ratio together for both eyes
                    ear = (leftEAR + rightEAR) / 2.0
                    # compute the convex hull for the left and right eye, then
                    # visualize each of the eyes and mouth
                    leftEyeHull = cv2.convexHull(leftEye)
                    rightEyeHull = cv2.convexHull(rightEye)
                    cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
                    cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

                    # check to see if the eye and mouth aspect ratio is below the 
                    # threshold, and if so, increment the blink frame counter
                    if ear < EYE_AR_THRESH:
                            COUNTER += 1
                            # if the eyes were closed for a sufficient number of frames
                            # then sound the alarm
                            if COUNTER >= EYE_AR_CONSECUTIVE_FRAMES:
                                print("eye blinked")
                                #print(type(active_letter))

                                user_key += active_letter
                                engine.say(active_letter)
                                # Flush the say() queue and play the audio
                                engine.runAndWait()
                                COUNTER = 0
                                
                    else:
                            COUNTER = 0
                    cv2.putText(blank_image, user_key, (30, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    if(len(user_key) == 4):
                        if user_key == "1234" or user_key == "5678":
                            print("yes")
                            engine.say("Authorized")
                            # Flush the say() queue and play the audio
                            engine.runAndWait()
                            user_key = ""
                            auth = 1
                        else:
                            cv2.putText(blank_image, "Invalid Password", (200, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                            engine.say("Invalid Password Enter OTP")
                            # Flush the say() queue and play the audio
                            engine.runAndWait()
                            user_key = ""
                            print("Invalid Password")
                            generateOTP()
                            chk_otp()

                    cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.putText(frame, "left eye:{}".format(leftEAR), (270, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.putText(frame, "right eye:{}".format(rightEAR), (270, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)  
                            

            # show the frame
            cv2.imshow("Frame", frame)
            cv2.imshow("Virtual keyboard", keyboard)
            cv2.imshow('Selected Key', blank_image)
            key = cv2.waitKey(160)
            if(auth):
                break
            if key == 27:
                break
            if cv2.waitKey(1) & 0xFF == ord('s'):
                user_key += active_letter
        cv2.destroyAllWindows()

        
        
def subjectchoose():
        now = time.time()  ###For calculate seconds of video
        future = now + 8
        if time.time() < future:
                recognizer = cv2.face.LBPHFaceRecognizer_create()  # cv2.createLBPHFaceRecognizer()
                try:
                    recognizer.read("TrainingImageLabel\Trainer.yml")
                except:
                    e = 'Model not found,Please train model'
                    Notification.configure(text=e, bg="red", fg="black", width=40, font=('times', 15, 'bold'))
                    Notification.place(x=600, y=120)

                harcascadePath = "haarcascade_frontalface_default.xml"
                faceCascade = cv2.CascadeClassifier(harcascadePath)
                cam = cv2.VideoCapture(0)
                font = cv2.FONT_HERSHEY_SIMPLEX
                while True:
                    ret, im = cam.read()
                    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                    faces = faceCascade.detectMultiScale(gray, 1.2, 5)
                    for (x, y, w, h) in faces:
                        global Id
                        global faceval
                        global rfidval
                        Id, conf = recognizer.predict(gray[y:y + h, x:x + w])
                        test = round(100 - conf)
                        #print(test)
                        if (test > 40):
                            faceval = Id
                            Id = names[Id]
                            global tt
                            tt = str(Id)
                            cv2.rectangle(im, (x, y), (x + w, y + h), (0, 260, 0), 7)
                            cv2.putText(im, str(tt), (x + h, y), font, 1, (255, 255, 0,), 4)
                        else:
                            Id = 'Unknown'
                            tt = str(Id)
                            cv2.rectangle(im, (x, y), (x + w, y + h), (0, 25, 255), 7)
                            cv2.putText(im, str(tt), (x + h, y), font, 1, (0, 25, 255), 4)
                    if time.time() > future:
                        break

                    cv2.imshow('Live Testing', im)
                    key = cv2.waitKey(30) & 0xff
                    if key == 27:
                        break
                if Id == 'Unknown':
                    print("Unknown detected")
                else:
                        cam.release()
                        cv2.destroyAllWindows()
                        print("face detected")
                        time.sleep(1)
                        chk_pass()


subjectchoose()
