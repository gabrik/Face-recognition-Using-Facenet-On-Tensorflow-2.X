import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pickle
from architecture import get_model, get_facenet
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import optimizers
from sklearn.preprocessing import Normalizer
from scipy.spatial.distance import cosine
#from face_detection_operation import get_detected_face


class FaceRecognition:

    def __init__(self):
        self.TRAINING_DATA_DIRECTORY = "./dataset/training"
        self.TESTING_DATA_DIRECTORY = "./dataset/testing"
        self.EPOCHS = 10
        self.BATCH_SIZE = 2
        self.NUMBER_OF_TRAINING_IMAGES = 64
        self.NUMBER_OF_TESTING_IMAGES = 42
        self.IMAGE_HEIGHT = 224
        self.IMAGE_WIDTH = 224
        self.model = get_model()
        self.training_generator = None

    @staticmethod
    def plot_training(history):
        plot_folder = "plot"
        plt.plot(history.history['accuracy'], label='accuracy')
        plt.plot(history.history['val_accuracy'], label='val_accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.ylim([0.1, 1])
        plt.legend(loc='lower right')

        if not os.path.exists(plot_folder):
            os.mkdir(plot_folder)

        plt.savefig(os.path.join(plot_folder, "model_accuracy.png"))

    @staticmethod
    def data_generator():
        img_data_generator = ImageDataGenerator(
            rescale=1./255,
            # horizontal_flip=True,
            fill_mode="nearest",
            # zoom_range=0.3,
            # width_shift_range=0.3,
            # height_shift_range=0.3,
            rotation_range=30
        )
        return img_data_generator

    def training(self):
        self.training_generator = FaceRecognition.data_generator().flow_from_directory(
            self.TRAINING_DATA_DIRECTORY,
            target_size=(self.IMAGE_WIDTH, self.IMAGE_HEIGHT),
            batch_size=self.BATCH_SIZE,
            class_mode='categorical'
        )

        testing_generator = FaceRecognition.data_generator().flow_from_directory(
            self.TRAINING_DATA_DIRECTORY,
            target_size=(self.IMAGE_WIDTH, self.IMAGE_HEIGHT),
            class_mode='categorical'
        )

        self.model.compile(
            loss='categorical_crossentropy',
            optimizer=optimizers.SGD(lr=1e-4, momentum=0.9, decay=1e-2 / self.EPOCHS),
            metrics=["accuracy"]
        )

        history = self.model.fit(
            self.training_generator,
            #steps_per_epoch=self.NUMBER_OF_TRAINING_IMAGES//self.BATCH_SIZE,
            epochs=self.EPOCHS,
            validation_data=testing_generator,
            shuffle=True,
            #validation_steps=self.NUMBER_OF_TESTING_IMAGES//self.BATCH_SIZE
        )

        FaceRecognition.plot_training(history)

    def save_model(self, model_name):
        model_path = "./model"
        # if not os.path.exists(model_name):
        #     os.mkdir(model_name)
        if not os.path.exists(model_path):
            os.mkdir(model_path)

        self.model.save(os.path.join(model_path, model_name))
        class_names = self.training_generator.class_indices
        class_names_file_reverse = model_name[:-3] + "_class_names_reverse.npy"
        class_names_file = model_name[:-3] + "_class_names.npy"
        np.save(os.path.join(model_path, class_names_file_reverse), class_names)
        class_names_reversed = np.load(os.path.join(model_path, class_names_file_reverse), allow_pickle=True).item()
        class_names = dict([(value, key) for key, value in class_names_reversed.items()])
        np.save(os.path.join(model_path, class_names_file), class_names)

    @staticmethod
    def load_saved_model(model_path):
        model = load_model(model_path)
        return model

    @staticmethod
    def model_prediction(face_array, model, classes):
        class_name = "Unknown"
        # face_array, face = get_detected_face(image_path)
        # model = load_model(model_path)
        #face_array = face.astype('float32')
        input_sample = np.expand_dims(face_array, axis=0)
        results = model.predict(input_sample)
        result = np.argmax(results, axis=1)
        index = result[0]
        accuracy = results[0][index]
        print(f'Results {results} index {index} accuracy {accuracy}')
        # classes = np.load(class_names_path, allow_pickle=True).item()
        # print(classes, type(classes), classes.items())
        if type(classes) is dict:
            for k, v in classes.items():
                if k == index:
                    class_name = v
        print(f'This is {class_name} with accuracy {accuracy}')
        return class_name, accuracy



class FaceNet():
    def __init__(self):
        self.model = get_facenet()
        self.model.load_weights('./recognition/facenet_keras_weights.h5')
        self.TRAINING_PATH = "./dataset/training"
        self.encodings = None
        self.l2_normalizer = Normalizer('l2')

    def training(self):
        encodes = []
        encoding_dict = dict()

        for face_names in os.listdir(self.TRAINING_PATH):
            person_dir = os.path.join(self.TRAINING_PATH, face_names)
            if not os.path.isdir(person_dir):
                continue

            for image_name in os.listdir(person_dir):
                image_file = os.path.join(person_dir,image_name)
                print(f"processing {image_file}")
                face = cv2.imread(image_file)
                if face is None:
                    continue
                face_d = FaceNet.preprocess(face)
                encode = self.model.predict(face_d)[0]
                encodes.append(encode)
            if encodes:
                encode = np.mean(encodes, axis=0 ) #it was sum
                encode = self.l2_normalizer.transform(np.expand_dims(encode, axis=0))[0]
                encoding_dict[face_names] = encode

        path = './encodings.pkl'
        with open(path, 'wb') as file:
            pickle.dump(encoding_dict, file)

    def model_prediction(self, face):
        recognition_t=0.4
        if self.encodings is None:
            path = './encodings.pkl'
            self.encodings = FaceNet.load_pickle(path)

        face_d = FaceNet.preprocess(face)
        encode = self.model.predict(face_d)[0]
        encode = self.l2_normalizer.transform(encode.reshape(1, -1))[0]
        distance = float("inf")
        name = 'unknown'
        predictions = []
        for db_name, db_encode in self.encodings.items():
            dist = cosine(db_encode, encode)
            predictions.append((db_name, dist))
            if dist < recognition_t:
                predictions.append((db_name, dist))

        if len(predictions) > 0:
            name, distance = min(predictions, key = lambda v: v[1])
        return name, distance


    @staticmethod
    def preprocess(face):
        required_shape = (224,224)
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = FaceNet.normalize(face)
        face = cv2.resize(face, required_shape)
        face_d = np.expand_dims(face, axis=0)
        return face_d


    @staticmethod
    def normalize(face):
        mean, std = face.mean(), face.std()
        return (face - mean) / std

    @staticmethod
    def face_distance(encodings, face):
        return np.linalg.norm(encodings - face)

    @staticmethod
    def load_pickle(path):
        with open(path, 'rb') as f:
            encoding_dict = pickle.load(f)
        return encoding_dict



if __name__ == '__main__':

    # With classes
    #model_name = "face_recognition.h5"
    #face_recognition = FaceRecognition()
    #face_recognition.training()
    #face_recognition.save_model(model_name)

    # facenet
    facenet = FaceNet()
    facenet.training()


