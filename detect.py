import cv2
import numpy as np
import mtcnn
from train import FaceRecognition, FaceNet
from prepare_images import get_detected_face, load_yolo, detect_faces
import json
import pickle

confidence_t=0.99
recognition_t=0.5
required_size = (224,224)


def load_classes(filepath):
    classes = np.load(filepath, allow_pickle=True).item()
    return classes

def get_face(img, x1, y1, width, height ):
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    face = img[y1:y2, x1:x2]
    face = np.array(cv2.resize(face, required_shape))
    #face = np.expand_dims(face, axis=0)
    return face, (x1, y1), (x2, y2)


def get_face_box(x1, y1, width, height ):
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    return (x1, y1), (x2, y2)


def detect_cnn(img ,model, classes, detector_model, detector_classes):

    faces = get_detected_face(img, detector_model, detector_classes)
    for f in faces:
        face_array, face, x, y, width, height = f
        pt_1, pt_2 = get_face_box(x, y, width, height)
        name, accuracy = FaceRecognition.model_prediction(face_array, model, classes)
        if accuracy < recognition_t:
            cv2.rectangle(img, pt_1, pt_2, (0, 0, 255), 2)
            cv2.putText(img, 'Unknown', pt_1, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
        else:
            cv2.rectangle(img, pt_1, pt_2, (0, 255, 0), 2)
            cv2.putText(img, name + f'__{accuracy:.2f}', (pt_1[0], pt_1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 1,(0, 200, 200), 2)

    return img

def detect_facenet(img ,model, detector_model, detector_classes):
    faces = get_detected_face(img, detector_model, detector_classes)
    for f in faces:
        face_array, face, x, y, width, height = f
        pt_1, pt_2 = get_face_box(x, y, width, height)
        name, distance = model.model_prediction(face)
        print(f'Result of prediction is {name}, {distance}')

        if name == 'unknown':
            cv2.rectangle(img, pt_1, pt_2, (0, 0, 255), 2)
            cv2.putText(img, 'Unknown', pt_1, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
        else:
            cv2.rectangle(img, pt_1, pt_2, (0, 255, 0), 2)
            cv2.putText(img, name + f'__{distance:.2f}', (pt_1[0], pt_1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 1,(0, 200, 200), 2)

    return img




if __name__ == "__main__":

    required_shape = (224,224)
    face_detector, detector_classes = load_yolo()

    # cnn
    # classes = load_classes('./model/face_recognition_class_names.npy')
    # model = FaceRecognition.load_saved_model('./model/face_recognition.h5')

    # facenet
    model = FaceNet()

    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret,frame = cap.read()

        if not ret:
            print("CAM NOT OPEND")
            break

        #frame = detect_cnn(frame , model , classes, face_detector, detector_classes)
        frame = detect_facenet(frame , model, face_detector, detector_classes)

        cv2.imshow('camera', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break




