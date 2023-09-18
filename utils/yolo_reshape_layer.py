from keras.layers import Layer
import keras.backend as K

class YoloReshapeLayer(Layer):

    def __init__(self, target_shape):
        super(YoloReshapeLayer, self).__init__()
        self.target_shape = tuple(target_shape)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            "target_shape": self.target_shape
        })
        return config
    
    def call(self, inputs):

        S = [self.target_shape[0], self.target_shape[1]]
        C = 5
        B = 1

        # From 0 1 2 ... 490 values the first
        # 245 (0 ... 224) (7 * 7 * 5) values are for hot encodings / probablities
        # and 245 (224 ... 489) (7 * 7 * 5) values are for the boxes (pc, x, y, w, h)
        classes_limit = S[0] * S[1] * C
        boxes_limit = classes_limit + (S[0] * S[1] * B)

        # K.reshape(tensor, shape)

        # Class probablities (0 -> 244)
        class_probs = K.reshape(
            inputs[:, :classes_limit], 
            (K.shape(inputs)[0], S[0], S[1], C)
        )
        class_probs = K.sigmoid(class_probs)

        # Confidence scores (pc) (244 -> 245)
        confidences = K.reshape(
            inputs[:, classes_limit:boxes_limit], 
            (K.shape(inputs)[0], S[0], S[1], B)
        )
        confidences = K.sigmoid(confidences)

        # Boxes (245 -> 489) (x, y, w, h) 
        boxes = K.reshape(
            inputs[:, boxes_limit:], 
            (K.shape(inputs)[0], S[0], S[1], B * 4)
        )
        boxes = K.sigmoid(boxes)

        # (pc, x, y, w, h, c0, c1, c2, c3, c4, c5)
        outputs = K.concatenate([confidences, boxes, class_probs])

        return outputs

