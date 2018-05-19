import numpy as np
from sklearn.metrics import roc_auc_score
from keras.callbacks import Callback
#from keras.models = import Model 


class RocAucEvaluation(Callback):
    def __init__(self, validation_data=(), interval=1, fold_idx=0):
        super(Callback, self).__init__()
        self.interval = interval
        self.X_val, self.y_val = validation_data
        self.cv_bestscore = 0
        self.cv_history = np.array([])
        self.fold_idx = fold_idx

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            y_pred = self.model.predict(self.X_val, batch_size=500, verbose=1)
            score = roc_auc_score(self.y_val, y_pred)
            print("¥n ROC-AUC - epoch: {:d} - score: {:.6f}".format(epoch+1, score))
            self.cv_history = np.append(self.cv_history, score)

            if self.cv_bestscore <= score:
                # update cross validation best score
                self.cv_bestscore = score
                if self.fold_idx > 0:
                    self.model.save("best_model.h5")
                else:
                    self.model.save("best_model_no_cross_val.h5")
                print("ROC-AUC score was improved!")
            else:
                print("ROC-AUC score wasn't improved!")