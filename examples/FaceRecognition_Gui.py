from tkinter import *
from tkinter import filedialog
from tkinter import messagebox
import tkinter as tk
import face_recognition
import os.path
import time
import os
from sklearn import svm
import pickle
import numpy as np
from PIL import Image, ImageDraw, ImageTk
from examples.DataProcessing import image_data, SaveImage
from face_recognition.face_recognition_cli import image_files_in_folder

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'PNG'}
CountDetect = 0
path_train = "Train"
path_test = "Test"
model_save_path = "svm_means_test.dat"

def OpenFolder():
    datafolder = filedialog.askdirectory()
    path.set(datafolder)
    return datafolder

def Openfile():
    filename = filedialog.askopenfilename()
    openfile.set(filename)
    return filename

def openImage():
    x = Openfile()
    img = Image.open(x)
    img = img.resize((145, 155), Image.ANTIALIAS)
    img = ImageTk.PhotoImage(img)
    lableImage = Label(Gui, image=img, bd=2, borderwidth=2, relief="ridge")
    lableImage.image = img
    lableImage.place(x=10, y=150)

def Train_Test():
    datafolder = path.get()
    path_train = "Train"
    path_test = "Test"
    if not os.path.exists(path_train):
        os.mkdir(path_train)
    if not os.path.exists(path_test):
        os.mkdir(path_test)
    messagebox.showinfo("Notification", "Start Dividing Data")
    X_train, X_test, Y_train, Y_test, path_train, path_test, datafolder = image_data(datafolder, path_train, path_test)
    SaveImage(X_train, X_test, Y_train, Y_test, path_train, path_test)
    train.set(len(X_train))
    test.set(len(X_test))
    print("Dividing Train Test Accssuary !")
    messagebox.showinfo("Notification", "Successfully Divided Data")

def TrainModels():
    print("Start Train Models....")
    messagebox.showinfo("Notification", "Start Model Training")
    start = time.time()
    verbose = False
    names = []
    means = []
    for class_dir in os.listdir(path_train):
        encodings = []
        if not os.path.isdir(os.path.join(path_train, class_dir)):
            continue
        for img_path in image_files_in_folder(os.path.join(path_train, class_dir)):
            image = face_recognition.load_image_file(img_path)
            face_bounding_boxes = face_recognition.face_locations(image)
            if len(face_bounding_boxes) != 1:
                if verbose:
                    print("Image {} not suitable for training: {}".format(img_path, "Didn't find a face" if len(
                        face_bounding_boxes) < 1 else "Found more than one face"))
            else:
                face_encodings = face_recognition.face_encodings(image)[0]
                encodings.append(face_encodings)
                names.append(class_dir)
                means.append([np.mean(x, axis=0) for x in zip(*encodings)])
    clf = svm.SVC(gamma='scale')
    clf.fit(means, names)
    if model_save_path is not None:
        with open(model_save_path, 'wb') as f:
            pickle.dump([clf,means,names], f)
    end = time.time()
    FileNumber_Train = train.get()
    timetrain.set(round(float(str(end - start)) / float(FileNumber_Train), 2))
    print("Train Models Accssuary !")
    messagebox.showinfo("Notification", "Successful model training")
    return clf, means, names

def show_prediction_labels_on_image(X_img_path,path_out,means,names,model_save_path=None,clf= None):
    global CountDetect
    if not os.path.isfile(X_img_path) or os.path.splitext(X_img_path)[1][1:] not in ALLOWED_EXTENSIONS:
        raise Exception("Invalid image path: {}".format(X_img_path))
    if clf is None and model_save_path is None:
        raise Exception("Must supply svm classifier either thourgh knn_clf or model_path")
    pil_image = Image.open(X_img_path).convert("RGB")
    draw = ImageDraw.Draw(pil_image)
    nameout = ""
    X_img = face_recognition.load_image_file(X_img_path)
    start = time.time()
    X_face_locations = face_recognition.face_locations(X_img)
    end = time.time()
    if len(X_face_locations) == 0:
        return []
    CountDetect = CountDetect + len(X_face_locations)
    faces_encodings = face_recognition.face_encodings(X_img, X_face_locations)
    namess = ""
    for (top, right, bottom, left), face_encoding in zip(X_face_locations, faces_encodings):
        matches = face_recognition.compare_faces(means, face_encoding)
        name = "unknown"
        face_distance = face_recognition.face_distance(means, face_encoding)
        best_match_index = np.argmin(face_distance)
        if matches[best_match_index]:
            name = names[best_match_index]
        draw.rectangle(((left, top), (right, bottom)), outline=(255, 0, 0))
        name = name.encode("UTF-8")
        text_width, text_height = draw.textsize(name)
        draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(255, 0, 0), outline=(255, 0, 0))
        draw.text((left + 6, bottom - text_height - 5), name, fill=(255, 255, 255, 255))
        nameout = str(name).replace("b'", "").replace("'", "")
        namess += nameout + " - "
    namespersion.set(namess[:-3])
    del draw
    pil_image.save(path_out)
    timetest.set(round((float(str(end-start)) / CountDetect), 2))
    return nameout

def Detector():
    if os.path.isfile(model_save_path):
        with open(model_save_path, 'rb') as f:
            clf, means, names = pickle.load(f)
    ImageOut = "Out"
    if not os.path.exists(ImageOut):
        os.mkdir(ImageOut)
    CountName = 0
    full_file_path = openfile.get()
    path_out = ImageOut + "/" + os.path.basename(full_file_path)
    name = show_prediction_labels_on_image(full_file_path, path_out, means, names, model_save_path)
    if str(name) == str(os.path.basename(full_file_path).split("_")[0]):
        CountName += 1
    recognition = path_out
    img = Image.open(recognition)
    img = img.resize((145, 155), Image.ANTIALIAS)
    img = ImageTk.PhotoImage(img)
    lableRecognition = Label(Gui, image=img, bd=2, borderwidth=2, relief="ridge")
    lableRecognition.image = img
    lableRecognition.place(x=200, y=150)

Gui = tk.Tk()
Gui.geometry("520x440")
Gui.title("Face Recognition SVM")

Title = Label(Gui, text="Face Recognition SVM", fg="red", font=("Arial", 16)).place(x=150, y=5)

# Thực hiện chức năng mở folder
btnOpenFolder = Button(Gui, text="1.Brower", command=OpenFolder, fg="black").place(x=10, y=40)
path = StringVar()
txtPath = Entry(Gui,textvariable=path, width=70).place(x=80, y=45)

# lấy số lượng ảnh trong tập train
TitleTrain=Label(Gui, text="Data Train", fg="black", font=("Arial", 9)).place(x=10, y=80)
train = StringVar()
txtPathTrain = Entry(Gui,textvariable =train,width=20).place(x=80, y=80)
txtfiletrain = Label(Gui, text="file", fg="black", font=("Arial", 9)).place(x=203, y=80)

# lấy số lượng ảnh trong tập test
TitleTest=Label(Gui, text="Data Test", fg="black", font=("Arial", 9)).place(x=300, y=80)
test = StringVar()
txtPathTest = Entry(Gui,textvariable=test, width=20).place(x=370, y=80)
txtfiletest = Label(Gui, text="file", fg="black", font=("Arial", 9)).place(x=483, y=80)

# thực hiện chức năng mở ảnh lên để test
openfile = StringVar()
btnOpenFile = Button(Gui, text="4. Open Image", command=openImage).place(x=10, y=115)
lableImage = Label(Gui, bd=2, text="Choose photos to check", borderwidth=2, width=20, height=10, relief="ridge").place(x=10, y=150)

recognition = StringVar()
lableRecognition = Label(Gui, bd=2, text="Recognized photo" ,borderwidth=2, width=20, height=10, relief="ridge").place(x=200, y=150)

# thực hiện lấy thời gian train trung bình của một ảnh
timetrain = StringVar()
lableTimeTrain = Label(Gui, bd=2, text="Train time").place(x=355, y=145)
txtTimeTrain = Entry(Gui,textvariable=timetrain, width=8).place(x=430, y=145)
txtstrain = Label(Gui, text="(s)", fg="black", font=("Arial", 9)).place(x=475, y=145)

# thực hiện lấy thời gian test của một ảnh
timetest =StringVar()
labeltest = Label(Gui, bd=2, text="Detect time").place(x=355, y=175)
txttest = Entry(Gui,textvariable=timetest, width=8).place(x=430, y=175)
txtstest= Label(Gui, text="(s)", fg="black", font=("Arial", 9)).place(x=475, y=175)

# thực hiện lấy tên ảnh của người cần nhận diện
namespersion = StringVar()
lableRecognitionFace=Label(Gui, text="Names", fg="black", font=("Arial", 9)).place(x=45, y=320)
txtNames = Entry(Gui, width=65, textvariable=namespersion).place(x=110, y=320)

btnChiaDuLieu = Button(Gui, text="2. Divide data", command=Train_Test, fg="black").place(x=170, y=350)
btnTrain = Button(Gui, text="3. Train", command=TrainModels, fg="black").place(x=280, y=350)
btnRecognition = Button(Gui, text="5.Detector and Recognition", command=Detector, fg="black").place(x=350, y=350)

Label(Gui, text="Electric Power University - Class D13CNPM5", fg="black", font=("Arial", 7)).place(x=10, y=390)
Label(Gui, text="Members: Nguyen Van Nam and Ha Quy Duc and Can Quang Trieu", fg="black", font=("Arial", 7)).place(x=10, y=410)

img = Image.open("MuiTen.png")
img = img.resize((25, 25), Image.ANTIALIAS)
img = ImageTk.PhotoImage(img)
lableMuiTen = Label(Gui, image=img, bd=2, borderwidth=2, relief="ridge")
lableMuiTen.image = img
lableMuiTen.place(x=165, y=220)

Gui.mainloop()