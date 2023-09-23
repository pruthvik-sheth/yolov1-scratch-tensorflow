import keras.backend as K


def yolo_loss(y_pred, y_true):

    true_pc = y_true[0]
    true_pc = K.expand_dims(true_pc)
    true_box = y_true[..., 1:5]
    true_class = y_true[..., 5:]

    predict_pc = y_pred[..., :2]
    predict_box = y_pred[..., 2:10]
    predict_class = y_pred[..., 10:]
    
    _true_box = K.reshape(true_box, (-1, 7, 7, 1, 4))
    _predict_box = K.reshape(true_box, (-1, 7, 7, 2, 4))

    


