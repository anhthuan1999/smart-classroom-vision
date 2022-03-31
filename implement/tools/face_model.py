
import face_preprocess
import sys
import os
import numpy as np
import mxnet as mx
import cv2
import sklearn
from sklearn.decomposition import PCA
from mtcnn_detector import MtcnnDetector
import config_recognition as params
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))


def get_model(ctx, image_size, model_str, layer):
    _vec = model_str.split(',')
    assert len(_vec) == 2
    prefix = _vec[0]
    epoch = int(_vec[1])
    print('loading', prefix, epoch)
    sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)
    all_layers = sym.get_internals()
    gr = mx.visualization.print_summary(sym)
    sym = all_layers[layer+'_output']
    model = mx.mod.Module(symbol=sym, context=ctx, label_names=None)
    model.bind(data_shapes=[('data', (1, 3, image_size[0], image_size[1]))])
    model.set_params(arg_params, aux_params)
    return model


class FaceModel:
    def __init__(self):
        if params.GPU:
            ctx = mx.gpu(params.DEVICE)
        else:
            ctx = mx.cpu(params.DEVICE)
        _vec = params.IMAGE_SIZE
        assert len(_vec) == 2
        image_size = (int(_vec[0]), int(_vec[1]))
        self.model = None

        if len(params.PATH_ENCODE) > 0:
            self.model = get_model(ctx, image_size, params.PATH_ENCODE, 'fc1')
        self.detector = MtcnnDetector(model_folder=params.PATH_MTCNN, ctx=ctx,
                                      num_worker=1, accurate_landmark=True,
                                      threshold=params.MTCNN_THRES)

    def get_input(self, face_img):
        ret = self.detector.detect_face(face_img, det_type=0)
        if ret is None:
            return None
        bboxs, points = ret
        if bboxs.shape[0] == 0:
            return None
        bbox = bboxs[0, :4]
        point = points[0, :].reshape((2, 5)).T
        nimg = face_preprocess.preprocess(
            face_img, bbox, point, image_size='112,112')
        nimg = cv2.cvtColor(nimg, cv2.COLOR_BGR2RGB)
        aligned = np.transpose(nimg, (2, 0, 1))
        return aligned

    def preprocess(self, face_img, face_bbox, face_landmark):
        point = face_landmark.reshape((2, 5)).T
        nimg = face_preprocess.preprocess(
            face_img, face_bbox, point, image_size='112,112')
        nimg = cv2.cvtColor(nimg, cv2.COLOR_BGR2RGB)
        aligned = np.transpose(nimg, (2, 0, 1))
        return aligned

    def get_feature(self, aligned):
        input_blob = np.expand_dims(aligned, axis=0)
        data = mx.nd.array(input_blob)
        db = mx.io.DataBatch(data=(data,))
        self.model.forward(db, is_train=False)
        embedding = self.model.get_outputs()[0].asnumpy()
        embedding = sklearn.preprocessing.normalize(embedding).flatten()
        return embedding
