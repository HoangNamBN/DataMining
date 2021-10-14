import face_recognition
import os.path
import time
import os
from sklearn import svm
import pickle
import cv2
import numpy as np
from face_recognition.face_recognition_cli import image_files_in_folder
from examples.DataProcessing import image_data, SaveImage, CheckMaxMin, CountImgaeTrainTest
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg','PNG'}

def train(path_train,model_save_path=None,verbose=False):
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
    return clf, means, names

if __name__ == "__main__":
    datafolder = "DataSetFace"
    path_train = "Train"
    path_test = "Test"
    model_save_path = "svm_means_test.dat"
    if not os.path.exists(path_train):
        os.mkdir(path_train)
    if not os.path.exists(path_test):
        os.mkdir(path_test)
    if len(os.listdir(path_train))!= 0 or len(os.listdir(path_test))!= 0:
        print("Da chia tap train test")
    else:
        X_train, X_test, Y_train, Y_test, path_train, path_test, datafolder = image_data(datafolder,path_train,path_test)
        SaveImage(X_train, X_test, Y_train, Y_test, path_train, path_test)
        CheckMaxMin(path_train)
    print ("Training SVM classifier...")
    start = time.time()
    clf, means, names = train(path_train,model_save_path)
    end = time.time()
    print("Training complete!")
    CountImage = 0
    for path in os.listdir(path_train):
        curr_path = os.path.join(path_train, path)
        for file in os.listdir(curr_path):
            os.path.join(curr_path, file)
            CountImage += 1
    print("Time to train a photo is: ", (float(str(end - start)) / CountImage))