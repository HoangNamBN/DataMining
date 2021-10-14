import face_recognition
from PIL import Image, ImageDraw
import os.path
import time
import os
import numpy as np
import pickle
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg','PNG'}
CountDetect = 0
timer = 0
def show_prediction_labels_on_image(X_img_path,path_out,means,names,model_save_path=None,clf= None):
    global CountDetect
    global timer
    if not os.path.isfile(X_img_path) or os.path.splitext(X_img_path)[1][1:] not in ALLOWED_EXTENSIONS:
        raise Exception("Invalid image path: {}".format(X_img_path))
    if clf is None and model_save_path is None:
        raise Exception("Must supply svm classifier either thourgh svm_clf or model_path")
    pil_image = Image.open(X_img_path).convert("RGB")
    draw = ImageDraw.Draw(pil_image)
    nameout = ""
    X_img = face_recognition.load_image_file(X_img_path)
    start = time.time()
    X_face_locations = face_recognition.face_locations(X_img)
    end = time.time()
    timer = timer + float(str(end-start))
    if len(X_face_locations) == 0:
        return []
    CountDetect = CountDetect + len(X_face_locations)
    faces_encodings = face_recognition.face_encodings(X_img, X_face_locations)
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
    del draw
    pil_image.save(path_out)
    return nameout

if __name__ == "__main__":
    path_train = "Train"
    path_test = "Test"
    model_save_path = "svm_means_test.dat"
    print ("Load model...")
    if os.path.isfile(model_save_path):
        with open(model_save_path, 'rb') as f:
            clf, means, names = pickle.load(f)
    ImageOut = "Out_means"
    if not os.path.exists(ImageOut):
        os.mkdir(ImageOut)
    CountName = 0
    for image_file in os.listdir(path_test):
        full_file_path = os.path.join(path_test, image_file)
        path_out = ImageOut + "/" + image_file
        name = show_prediction_labels_on_image(full_file_path,path_out,means,names,model_save_path)
        if str(name) == str(image_file.split("_")[0]):
            CountName += 1
    print("saved successfully !")
    print("The face number has been detected: ", CountDetect)
    print("Detect accuracy in test set: ", str(CountDetect * 100 / len(os.listdir(path_test))))
    print("The time to detect an image is: ", timer / CountDetect)
    print("Number of correctly identified images: ", (CountName))
    print("Detection accuracy for the detected set: ", str(CountName * 100 / CountDetect))