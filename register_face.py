import tkinter as tk
from tkinter import *
import cv2
import csv
import os
import numpy as np
from PIL import Image,ImageTk
import datetime
import time

#iniciate id counter
Id = 0

# names related to ids
names = ['krishna','2','3','4','5'] #replace with ur names in order

#Main Windows of GUI

window = tk.Tk()
window.title("Face Recognition System")

window.geometry('900x550')
window.configure(background='snow')

##For clear textbox
def clear():
    txt.delete(first=0, last=22)

def clear1():
    txt2.delete(first=0, last=22)
    
def del_sc1():
    sc1.destroy()

def err_screen():
    global sc1
    sc1 = tk.Tk()
    sc1.geometry('300x100')
    sc1.title('Warning!!')
    sc1.configure(background='snow')
    Label(sc1,text='Enrollment & Name required!!!',fg='red',bg='white',font=('times', 16, ' bold ')).pack()
    Button(sc1,text='OK',command=del_sc1,fg="black"  ,bg="lawn green"  ,width=9  ,height=1, activebackground = "Red" ,font=('times', 15, ' bold ')).place(x=90,y= 50)


message = tk.Label(window, text="FACE REGISTRATION", bg="snow", fg="black", width=48,
                   height=2, font=('times', 25, 'italic bold '))
message.place(x=5, y=5)

def testVal(inStr,acttyp):
    if acttyp == '1': #insert
        if not inStr.isdigit():
            return False
    return True

###For take images for datasets
def take_img():
    l1 = txt.get()
    l2 = txt2.get()
    if l1 == '':
        err_screen()
    elif l2 == '':
        err_screen()
    else:
        try:
            cam = cv2.VideoCapture(0)
            detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
            Enrollment = txt.get()
            Name = txt2.get()
            sampleNum = 0
            while (True):
                ret, img = cam.read()
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = detector.detectMultiScale(gray, 1.3, 5)
                for (x, y, w, h) in faces:
                    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    # incrementing sample number
                    sampleNum = sampleNum + 1
                    # saving the captured face in the dataset folder
                    cv2.imwrite("TrainingImage/ " + Name + "." + Enrollment + '.' + str(sampleNum) + ".jpg",
                                gray[y:y + h, x:x + w])
                    cv2.imshow('Frame', img)
                # wait for 100 miliseconds
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                # break if the sample number is morethan 100
                elif sampleNum > 100:
                    break
            cam.release()
            cv2.destroyAllWindows()
            res = "Images Saved for Enrollment : " + Enrollment + " Name : " + Name
            Notification.configure(text=res, bg="SpringGreen3", width=40, font=('times', 15, 'bold'))
            Notification.place(x=600, y=120)
        except FileExistsError as F:
            f = 'Face Data already exists'
            Notification.configure(text=f, bg="SpringGreen3", width=40, font=('times', 15, 'bold'))
            Notification.place(x=600, y=120)

###For train the model
def trainimg():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    global detector
    detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    try:
        global faces,Id
        faces, Id = getImagesAndLabels("TrainingImage")
    except Exception as e:
        l='please make "TrainingImage" folder & put Images'
        Notification.configure(text=l, bg="SpringGreen3", width=40, font=('times', 15, 'bold'))
        Notification.place(x=600, y=120)

    recognizer.train(faces, np.array(Id))
    try:
        recognizer.save("TrainingImageLabel\Trainer.yml")
    except Exception as e:
        q='Please make "TrainingImageLabel" folder'
        Notification.configure(text=q, bg="SpringGreen3", width=40, font=('times', 15, 'bold'))
        Notification.place(x=600, y=120)

    res = "Model Trained"  # +",".join(str(f) for f in Id)
    Notification.configure(text=res, bg="SpringGreen3", width=40, font=('times', 15, 'bold'))
    Notification.place(x=600, y=120)

def getImagesAndLabels(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    # create empth face list
    faceSamples = []
    # create empty ID list
    Ids = []
    # now looping through all the image paths and loading the Ids and the images
    for imagePath in imagePaths:
        # loading the image and converting it to gray scale
        pilImage = Image.open(imagePath).convert('L')
        # Now we are converting the PIL image into numpy array
        imageNp = np.array(pilImage, 'uint8')
        # getting the Id from the image

        Id = int(os.path.split(imagePath)[-1].split(".")[1])
        # extract the face from the training image sample
        faces = detector.detectMultiScale(imageNp)
        # If a face is there then append that in the list as well as Id of it
        for (x, y, w, h) in faces:
            faceSamples.append(imageNp[y:y + h, x:x + w])
            Ids.append(Id)
    return faceSamples, Ids
                
def live_testing():
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

            Id, conf = recognizer.predict(gray[y:y + h, x:x + w])
            test = round(100 - conf)
            print(test)
            if (test > 40):
                Id = names[Id]
                global aa
                global date
                global timeStamp
                aa = Id
                global tt
                tt = str(Id) + '-Face'
                cv2.rectangle(im, (x, y), (x + w, y + h), (0, 260, 0), 7)
                cv2.putText(im, str(tt), (x + h, y), font, 1, (255, 255, 0,), 4)
            else:
                Id = 'Unknown' + '-Face'
                tt = str(Id)
                cv2.rectangle(im, (x, y), (x + w, y + h), (0, 25, 255), 7)
                cv2.putText(im, str(tt), (x + h, y), font, 1, (0, 25, 255), 4)
        cv2.imshow('Live Test', im)
        key = cv2.waitKey(30) & 0xff
        if key == 27:
            break
    cam.release()
    cv2.destroyAllWindows()
        
def on_closing():
    from tkinter import messagebox
    if messagebox.askokcancel("Quit", "Do you want to quit?"):
        window.destroy()
window.protocol("WM_DELETE_WINDOW", on_closing)

Notification = tk.Label(window, text="All things good", bg="Green", fg="white", width=40,
                      height=3, font=('times', 15, 'bold'))


lbl = tk.Label(window, text="Enter Person ID:", width=20, height=1, fg="white", bg="grey", font=('times', 15, ' bold '))
lbl.place(x=30, y=120)

txt = tk.Entry(window, validate="key", width=20, bg="yellow", fg="red", font=('times', 15, ' bold '))
txt['validatecommand'] = (txt.register(testVal),'%P','%d')
txt.place(x=290, y=120)

lbl2 = tk.Label(window, text="Enter Person Name", width=20, fg="white", bg="grey", height=1, font=('times', 15, ' bold '))
lbl2.place(x=30, y=170)

txt2 = tk.Entry(window, width=20, bg="yellow", fg="red", font=('times', 15, ' bold '))
txt2.place(x=290, y=170)

clearButton = tk.Button(window, text="Clear",command=clear,fg="black"  ,bg="deep pink"  ,width=10  ,height=1 ,activebackground = "Red" ,font=('times', 10, ' bold '))
clearButton.place(x=510 ,y=120)

clearButton1 = tk.Button(window, text="Clear",command=clear1,fg="black"  ,bg="deep pink"  ,width=10 ,height=1, activebackground = "Red" ,font=('times', 10, ' bold '))
clearButton1.place(x=510, y=170)

takeImg = tk.Button(window, text="1 Capture Images",command=take_img,fg="white"  ,bg="blue2"  ,width=20  ,height=3, activebackground = "Red" ,font=('times', 15, ' bold '))
takeImg.place(x=90, y=300)

trainImg = tk.Button(window, text="2 Train Images",fg="black",command=trainimg ,bg="lawn green"  ,width=20  ,height=3, activebackground = "Red" ,font=('times', 15, ' bold '))
trainImg.place(x=90, y=450)

liveWindow = tk.Button(window, text="3 Live Test", command=live_testing ,fg="black"  ,bg="lawn green"  ,width=20  ,height=3, activebackground = "Red" ,font=('times', 15, ' bold '))
liveWindow.place(x=400, y=300)

quitWindow = tk.Button(window, text="Quit", command=on_closing  ,fg="black"  ,bg="lawn green"  ,width=20  ,height=3, activebackground = "Red" ,font=('times', 15, ' bold '))
quitWindow.place(x=400, y=450)

window.mainloop()

