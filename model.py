import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout
from keras.regularizers import l2
from keras.layers import LeakyReLU
from utils.yolo_reshape_layer import YoloReshapeLayer


if __name__ == "__main__":
    model = Sequential()
    lrelu = LeakyReLU(alpha = 0.1)

    architecture_config = [
        #channel_count, kernel, strides, padding
        (64, 7, 2, "same"),
        "M",
        (192, 3, 1, "same"),
        "M",
        (128, 1, 1, "valid"),
        (256, 3, 1, "same"),
        (256, 1, 1, "valid"),
        (512, 3, 1, "same"),
        "M",
        [(256, 1, 1, "valid"), (512, 3, 1, "same"), 4],
        (512, 1, 1, "valid"),
        (1024, 3, 1, "same"),
        "M",
        [(512, 1, 1, "valid"), (1024, 3, 1, "same"), 2],
        (1024, 3, 1, "same"),
        (1024, 3, 2, "same"),
        (1024, 3, 1, "same"),
        (1024, 3, 1, "same")
    ]

    for count, layer_cfg in enumerate(architecture_config):

        if type(layer_cfg) == tuple:
            print("Creating block", layer_cfg)
            model.add(Conv2D(
                filters = layer_cfg[0], 
                kernel_size = layer_cfg[1], 
                strides = layer_cfg[2], 
                padding = layer_cfg[3], 
                activation = lrelu, 
                input_shape = (448, 448, 3) if count == 0 else (), 
                kernel_regularizer = l2(5e-4)
            ))
            
        elif type(layer_cfg) == str:
            print("Creating max block", layer_cfg)
            model.add(MaxPool2D(
                pool_size = 2,
                strides = 2,
                padding = "same"
            ))

        elif type(layer_cfg) == list:
            for j in range(layer_cfg[2]):
                for x in layer_cfg:
                    if type(x) == tuple:
                        model.add(Conv2D(
                            filters = x[0], 
                            kernel_size = x[1], 
                            strides = x[2], 
                            padding = x[3], 
                            activation = lrelu,
                            kernel_regularizer = l2(5e-4)
                        ))
                

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Dense(1024))
    model.add(Dropout(0.5))
    model.add(Dense(490, activation = "sigmoid"))
    model.add(YoloReshapeLayer(target_shape = (7, 7, 10)))
    print(model.summary())