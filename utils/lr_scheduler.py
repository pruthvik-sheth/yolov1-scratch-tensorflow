from keras.callbacks import Callback
import keras.backend as K

class LearningRateScheduler(Callback):

    def __init__(self, schedule):
        super(LearningRateScheduler, self).__init__()
        self.schedule = schedule

    def on_epoch_begin(self, epoch, logs=None):
        if not hasattr(self.model.optimizer, "lr"):
            raise ValueError("Optimizer must have the Lr attribute")
        
        # Fetching the learning rate.
        lr = float(K.get_value(self.model.optimizer.learning_rate))
        # Decided Lr
        scheduled_lr = self.schedule(epoch, lr)
        # Updating the learning rate
        K.set_value(self.model.optimizer.lr, scheduled_lr)
        
        print("\nEpoch %05d: Learning rate is %6.4f." % (epoch, scheduled_lr))


LR_SCHEDULE = [
    (0, 0.01),
    (75, 0.001),
    (105, 0.0001)
]

def lr_schedule(epoch, lr):
    """Helper function to retrieve the scheduled learning rate based on epoch."""
    if epoch < LR_SCHEDULE[0][0] or epoch > LR_SCHEDULE[-1][0]:
        return lr
    for i in range(len(LR_SCHEDULE)):
        if epoch == LR_SCHEDULE[i][0]:
            return LR_SCHEDULE[i][1]
    return lr