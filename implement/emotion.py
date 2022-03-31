from keras.models import load_model
from efficientnet.tfkeras import EfficientNetB7
from keras.preprocessing.image import load_img, img_to_array, array_to_img
import config_implement as params
import numpy as np
from skimage.transform import resize
from tools.mtcnn_detector import MtcnnDetector
import mxnet as mx
import tensorflow as tf
import cv2
import glob
import os


class Emotion:
    def __init__(self):
        self.model = load_model(params.PATH_MODEL_EMOTION)

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

    def detectFaces(self, image_path):
        img = load_img(image_path)
        imgArray = img_to_array(img)
        lstImg = []
        ret = self.detector.detect_face(imgArray, det_type=0)
        if ret is None:
            return lstImg
        fd_results, landmark = ret
        for idx, detection_result in enumerate(fd_results):
            (box_left, box_top, box_right, box_bottom) = (
                int(x) for x in detection_result[:4]
            )
            minus = (box_right-box_left)-(box_bottom-box_top)
            if minus < 0:
                box_right = box_right+abs(minus)
            else:
                box_bottom = box_bottom+abs(minus)
            lstImg.append(imgArray[box_top:box_bottom, box_left:box_right, :])

        return lstImg

    def preprocessing(self, img_path):

        lstImgArray = self.detectFaces(img_path)
        lstX_res = []
        if len(lstImgArray) < params.NUM_PEOPLE:
            return lstX_res

        for img in lstImgArray:
            #i = array_to_img(img)
            # i.show()
            try:
                X = tf.image.rgb_to_grayscale(img)
                X = resize(
                    X, (48, 48, 1), anti_aliasing=True)
                X = np.array(X, dtype='float32')
                X = X/255.0

                X_res = np.zeros((1, params.RESIZE, params.RESIZE, 3))
                sample = X.reshape(48, 48)
                image_resized = resize(
                    sample, (params.RESIZE, params.RESIZE), anti_aliasing=True)

                X_res[0, :, :, :] = image_resized.reshape(
                    params.RESIZE, params.RESIZE, 1)
                lstX_res.append(X_res)
            except:
                pass

        return lstX_res

    def predict(self, img_path):
        lstX_res = self.preprocessing(img_path)
        y_pred = []
        if len(lstX_res) == 0:
            return y_pred
        y_pred = [self.model.predict(resizeImg) for resizeImg in lstX_res]
        return y_pred

    def saveCv2(self, path_video, dir):
        cap = cv2.VideoCapture('../test/test.mp4')
        i = 0
        while(cap.isOpened()):
            ret, frame = cap.read()
            if ret == False:
                break
            cv2.imwrite(os.path.join(dir, str(i)+'.jpg'), frame)
            i += 1
            print(i)
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    img = '../test/4.jpg'
    dir = 'frame'
    lstImgPaths = glob.glob(os.path.join(dir, '*.jpg'))
    emotion = Emotion()
    #emotion.saveCv2('../test/test.mp4', 'frame')

    lstemo = []
    for image_path in lstImgPaths:
        # Opens the Video file
        inds = emotion.predict(image_path)
        if len(inds) == 0:
            continue
        print('=======================')
        print([np.argmax(i) for i in inds])
        emo = image_path.split(
            '/')[-1]+' '.join([str(np.argmax(i)) for i in inds])
        lstemo.append(emo)

    with open('emo.txt', 'w') as f:
        for item in lstemo:
            f.write("%s\n" % item)
