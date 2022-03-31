
class FaceDetectionResult():

    def __init__(self, face_mat=None, face_landmark=None, box_left=None, box_right=None, box_top=None, box_bottom=None, match=False, feature_vector=None):
        self.face_mat = face_mat
        self.face_landmark = face_landmark
        self.box_left = box_left
        self.box_right = box_right
        self.box_top = box_top
        self.box_bottom = box_bottom
        self.match = match
        self.feature_vector = feature_vector
