from os.path import join, exists
from os import mkdir, listdir
import glob
import os
from PIL import Image
import numpy as np
import cv2

def read_classes(filename):
    classes = list()
    with open(filename) as file:
        while (line := file.readline().rstrip()):
            classes.append(line)
    return classes



def detect_faces(frame, model, classes):
    CONFIDENCE_THRESHOLD = 0.5
    NMS_THRESHOLD = 0.4
    classes, scores, boxes = model.detect(frame, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
    faces = list()
    for (classid, score, box) in zip(classes, scores, boxes):
        label = classes[classid]
        x, y, width, height = box
        face = frame[y:y + height, x:x + width]
        if score > CONFIDENCE_THRESHOLD:
            faces.append((face, box))

    return faces

def load_yolo():
    classes_file = './detection/face_classes.txt'
    net_cfg = './detection/yolov3-face.cfg'
    net_weights = './detection/yolov3-wider_16000.weights'
    net = cv2.dnn.readNet(net_cfg, net_weights)
    classes = read_classes(classes_file)

    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
    model = cv2.dnn_DetectionModel(net)
    model.setInputParams(size=(512, 512), scale=1/255, swapRB=True)
    return model, classes

def save_cropped_face(images_root_folder,
                      required_size=(224, 224),
                      cropped_folder='dataset'):


    training_folder = 'training'
    testing_folder = 'testing'

    classes_file = './detection/face_classes.txt'
    net_cfg = './detection/yolov3-face.cfg'
    net_weights = './detection/yolov3-wider_16000.weights'


    if not exists(images_root_folder):
        return Exception("Input Images folder is not exist.")

    # Configuring DNN
    model, classes = load_yolo()


    people = listdir(images_root_folder)
    print(f'People {people}')
    for person in people:
        i = 0
        print(f'Person {person}')
        person_dir = os.path.join(images_root_folder,person)
        if not os.path.isdir(person_dir):
            continue
        for image_name in os.listdir(person_dir):
            image_file = os.path.join(person_dir,image_name)
            print(f"processing {image_file}")
            img = cv2.imread(image_file)
            if img is None:
                continue
            ## detect faces
            results = detect_faces(img, model, classes)
            if not results:
                continue
            face = results[0][0]
            try:
                image = Image.fromarray(face)
            except ValueError:
                continue
            image = image.resize(required_size)
            face_array = np.asarray(image)
            if not exists(cropped_folder):
                mkdir(cropped_folder)
            if not exists(join(cropped_folder, person)):
                mkdir(join(cropped_folder, person))
            output_file_name = f"{person}_{i}.jpeg"
            output_file_path = join(cropped_folder, person, output_file_name)
            print(f"Saving file {output_file_path}")
            cv2.imwrite(
                output_file_path,
                face_array)
            i = i + 1



def get_detected_face(img, detector, classes, required_size=(224, 224)):
    results = detect_faces(img, detector, classes)
    faces = []
    for (face, box) in results:
        image = Image.fromarray(face)
        image = image.resize(required_size)
        face_array = np.asarray(image)
        x, y, width, height = box
        faces.append((face_array, face, x, y, width, height))
    return faces


if __name__ == '__main__':
    save_cropped_face("people")