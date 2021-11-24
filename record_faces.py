import cv2
import mtcnn
import sys
import os

confidence_t=0.99
required_shape = (224,224)


def get_face(img, box):
    x1, y1, width, height = box
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    face = img[y1:y2, x1:x2]
    face = cv2.resize(face, required_shape)
    return face, (x1, y1), (x2, y2)


def detect(img, detector):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = detector.detect_faces(img_rgb)
    for res in results:
        if res['confidence'] < confidence_t:
            continue
        face, pt_1, pt_2 = get_face(img_rgb, res['box'])
        return face



if __name__ == "__main__":

    face_detector = mtcnn.MTCNN()
    name = sys.argv[1]
    dir_path = os.path.join('.','faces', name)
    os.mkdir(dir_path)

    testing_dir_path = os.path.join('.','testing', name)
    os.mkdir(testing_dir_path)

    cap = cv2.VideoCapture(0)
    i = 0
    while cap.isOpened():
        if i < 100:
            ret,frame = cap.read()

            if not ret:
                print("CAM NOT OPEND")
                break

            frame= detect(frame, face_detector)

            cv2.imshow('camera', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            filename = name+'_'+str(i)+'.jpg'
            filepath = os.path.join(dir_path, filename)
            print(f'Filename {filename}')
            cv2.imwrite(filepath, frame)
            i=i+1
            print(f'Caputred face n: {i}')
        else:
            break


    i = 0
    while cap.isOpened():

        if i < 25:
            ret,frame = cap.read()

            if not ret:
                print("CAM NOT OPEND")
                break

            frame= detect(frame, face_detector)

            cv2.imshow('camera', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            filename = name+'_'+str(i)+'.jpg'
            filepath = os.path.join(testing_dir_path, filename)
            print(f'Filename {filename}')
            cv2.imwrite(filepath, frame)
            i=i+1
            print(f'Caputred face n: {i}')
        else:
            exit(0)




