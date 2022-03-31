import sys
import numpy
import cv2
import os
import time
import pickle
import mxnet as mx
import config_recognition as params
from tools import face_model

from tools.mtcnn_detector import MtcnnDetector
from face_detection_results import *
from imutils import paths


class Encode:
    def __init__(self):
        if params.GPU:
            ctx = mx.gpu(params.DEVICE)
        else:
            ctx = mx.cpu(params.DEVICE)

        self.detector = MtcnnDetector(
            model_folder=params.PATH_MTCNN,
            ctx=ctx,
            num_worker=1,
            accurate_landmark=True,
            threshold=params.MTCNN_THRES,
        )
        self.insightface_model = face_model.FaceModel()

    def fd_post_processing(self, fd_results, landmarks, image):
        detected_faces = []

        for idx, detection_result in enumerate(fd_results):
            (box_left, box_top, box_right, box_bottom) = (
                int(x) for x in detection_result[:4]
            )
            detected_faces.append(
                FaceDetectionResult(
                    face_mat=image,
                    face_landmark=landmarks[idx],
                    box_left=box_left,
                    box_top=box_top,
                    box_right=box_right,
                    box_bottom=box_bottom,
                )
            )

        return detected_faces

    def saveFace(self, filter_face, user_input):
        face_encoding = []
        print('=====================')
        print(filter_face.feature_vector.shape)
        if params == False and os.path.exists(params.PATH_DATABASE):
            with open(params.PATH_DATABASE, "rb+") as f:
                face_encoding = pickle.load(f)
                face_encoding.append(
                    [user_input, filter_face.feature_vector])
                f.seek(0)
                pickle.dump(face_encoding, f)
                print(" SAVED SUCCESS")
                f.close()
        else:
            with open(params.PATH_DATABASE, "wb") as f:
                face_encoding.append(
                    [user_input, filter_face.feature_vector])
                pickle.dump(face_encoding, f)
                print(" SAVED SUCCESS")
                f.close()

    def getCloset(self, detected_faces, image_w, image_h):
        if len(detected_faces) == 0:
            return -1
        uf = max(
            detected_faces,
            key=lambda f: (
                (f.box_bottom - f.box_top) * (f.box_right - f.box_left)
            ),
            default=detected_faces[0],
        )

        if (uf.box_bottom - uf.box_top) * (
            uf.box_right - uf.box_left
        ) < params.FACESIZE_THRES * image_w * image_h:
            return -1
        return uf

    def encodeFace(self, filter_face):
        face_img = self.insightface_model.preprocess(
            filter_face.face_mat,
            (
                filter_face.box_left,
                filter_face.box_top,
                filter_face.box_right,
                filter_face.box_bottom,
            ),
            filter_face.face_landmark,
        )

        face_feature_vector = self.insightface_model.get_feature(face_img)
        filter_face.feature_vector = face_feature_vector
        return filter_face

    def encode(self):
        lstUsers = [name for name in os.listdir(params.PATH_USER)]
        for user_input in lstUsers:
            pathUser = os.path.join(params.PATH_USER, user_input)
            imagePaths = list(paths.list_images(params.PATH_USER))

            for i, imagePath in enumerate(imagePaths):
                rawImg = cv2.imread(imagePath)
                image_height = rawImg.shape[0]
                image_width = rawImg.shape[1]
                processedImg = rawImg.copy()

                ret = self.detector.detect_face(processedImg, det_type=0)
                if ret is None:
                    continue
                fd_results, landmarks = ret

                detected_faces = self.fd_post_processing(
                    fd_results, landmarks, processedImg
                )

                filter_face = self.getCloset(
                    detected_faces, image_width, image_height)

                if filter_face != -1:
                    filter_face = self.encodeFace(filter_face)
                    self.saveFace(filter_face, user_input)
                    print('Add: '+str(user_input))


# main entry point for program. we'll call main() to do what needs to be done.
if __name__ == "__main__":
    encode = Encode()
    encode.encode()
