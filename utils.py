import torch


from sklearn.preprocessing import StandardScaler
from sklearn.base import TransformerMixin,BaseEstimator
#https://stackoverflow.com/questions/50125844/how-to-standard-scale-a-3d-matrix
class StandardScaler3D(BaseEstimator,TransformerMixin):
    #batch, sequence, channels
    def __init__(self):
        self.scaler = StandardScaler()

    def fit(self,X,y=None):
        self.scaler.fit(X.reshape(-1, X.shape[2]))
        return self

    def transform(self,X):
        return self.scaler.transform(X.reshape( -1,X.shape[2])).reshape(X.shape)


def evaluate_model(model, loss_func, data_iter):
    model.eval()
    loss_sum, n = 0, 0
    with torch.no_grad():
        for x, y in data_iter:
            y_pred = model(x)
            y_pred = y_pred.squeeze()
            loss = loss_func(y_pred,y)
            loss_sum += loss.item()
            n += 1
        return loss_sum / n

from sklearn.metrics import accuracy_score, confusion_matrix

def cal_accuracy(model, labels, features):
    with torch.no_grad():
        y_hat = model(features)

    yhat = [0 if i<0.5 else 1 for i in y_hat]
    ytrue = labels.numpy()
    ypreds = yhat

    return accuracy_score(ytrue, ypreds), confusion_matrix(ytrue, ypreds)