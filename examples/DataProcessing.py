import os
import os.path
import cv2
import random
from shutil import copyfile
from sklearn.model_selection import train_test_split
def image_data(datafolder, path_train, path_test):
    X = []
    Y = []
    print ("Is dividing the train test ...")
    for folder in os.listdir(datafolder):
        curr_path = os.path.join(datafolder, folder)
        for file in os.listdir(curr_path):
            curr_file = os.path.join(curr_path, file)
            image = cv2.imread(curr_file)
            X.append(image)
            Y.append(folder)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state= 42)
    print ("Number of photo train:",len(X_train), "and Number of photo test:",len(X_test))
    return X_train, X_test, Y_train, Y_test, path_train, path_test, datafolder

def SaveImage(X_train, X_test, Y_train, Y_test, path_train, path_test):
    for i in range(len(X_train)):
        if not os.path.exists(path_train + "/" + Y_train[i]):
            os.mkdir(path_train + "/" + Y_train[i])
        pathout_train = path_train + "/" + Y_train[i]
        cv2.imwrite(pathout_train + "/" +  str(i) + ".jpg", X_train[i])
    for i in range(len(X_test)):
        cv2.imwrite(path_test + "/" + Y_test[i] + "_" + str(i) + ".jpg", X_test[i])

def image(datafolder, path_train, path_test):
    for folder in os.listdir(datafolder):
        X= []
        curr_path = os.path.join(datafolder, folder)
        if not os.path.exists(path_train + "/" + folder):
            os.mkdir(path_train + "/" + folder)
        for file in os.listdir(curr_path):
            curr_file = os.path.join(curr_path, file)
            if os.path.getsize(curr_file) !=0:
                X.append(file)
                print ("X: ", X)
            else: print ("zero file lenght")
        train_num = int(len(X) * 0.8)
        file_sh = random.sample(X, len(X))
        train_Set = file_sh[0:train_num]
        test_Set = file_sh[train_num:]
        for f in train_Set:
            s = os.path.join(curr_path,f)
            d = os.path.join(path_train +"/"+ folder,f)
            copyfile(s, d)
        for f in test_Set:
            s = os.path.join(curr_path,f)
            d = os.path.join(path_test, folder +"_"+ f)
            copyfile(s, d)
    return datafolder, path_train, path_test

def CountImgaeTrainTest(path_train, path_test):
    X_train =[]
    for folder in os.listdir(path_train):
        curr_path = os.path.join(path_train, folder)
        for folder_image in os.listdir(curr_path):
            image = os.path.join(curr_path, folder_image)
            image = cv2.imread(image)
            X_train.append(image)
    print(len(X_train))
    for image in os.listdir(path_test):
        os.path.join(path_test,image)
    print (len(os.listdir(path_test)))
    return path_train, path_test

def ImageFileCount(path_train, path_test):
    for path in os.listdir(path_train):
        curr_path = os.path.join(path_train, path)
        CountImage = 0
        for file in os.listdir(curr_path):
            os.path.join(curr_path, file)
            CountImage += 1
        print ("folder", curr_path,"la:", CountImage,"image to train")
    print (CountImage)
    list_name = []
    for image_file in os.listdir(path_test):
        str(image_file.split("_")[0])
        list_name.append(str(image_file.split("_")[0]))
    set_name = set(list_name)
    sorted(set_name)
    for cout_name in set_name:
        result = 0
        for name in list_name:
            if name == cout_name:
                result +=1
        print ("folder",cout_name,"using",result,"image to test")

def CheckMaxMin(path_train):
    NumberFile = path_train + "/" + os.listdir(path_train)[0]
    filemin = len(os.listdir(NumberFile))
    filemax = len(os.listdir(NumberFile))
    foldermin = os.listdir(path_train)[0]
    foldermax = os.listdir(path_train)[0]
    for folder in os.listdir(path_train):
        person = path_train + "/" + folder
        if len(os.listdir(person)) < filemin:
            filemin = len(os.listdir(person))
            foldermin = folder
        if len(os.listdir(person)) > filemax:
            filemax = len(os.listdir(person))
            foldermax = folder
    print ('The least picture folder is the folder', foldermin, 'with image number is:', filemin)
    print ("The folder with the most pictures is the folder", foldermax , 'with image number is:', filemax)