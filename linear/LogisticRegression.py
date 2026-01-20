import numpy as np 


def sigmoid(x):
    return (1/(1+np.exp(-x)))

class LogisticRegression():
    def __init__(self, lr = 0.01, epochs = 10):
        self.lr = lr 
        self.epochs = epochs
        self.weights = None
        self.bias = None

    def fit(self, X,y):
        n_samples, n_features = X.shape # rows, columns 
        self.weights = np.zeros(n_features)
        self.bias = 0 

        for _ in range(self.epochs):
            y_pred = np.dot(X, self.weights) + self.bias
            y_pred_sigmoid = sigmoid(y_pred)
            
            dw = (1/n_samples) * np.dot(X.T , (y_pred_sigmoid - y))
            db = (1/n_samples) * np.sum(y_pred_sigmoid - y)
            
            self.weights = self.weights - self.lr * dw
            self.bias = self.bias - self.lr * db
        
    def predict(self, X):
        y_pred = np.dot(X, self.weights) + self.bias
        y_pred_sigmoid = sigmoid(y_pred)
        class_pred = [0 if y<=0.5 else 1 for y in y_pred_sigmoid]
        return class_pred 
    
if __name__ == "__main__":
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn import datasets
    import matplotlib.pyplot as plt
    from linear.LogisticRegression import LogisticRegression

    bc = datasets.load_breast_cancer()
    X,y = bc.data, bc.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

    clf = LogisticRegression()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    def accuracy(y_pred, y_test):
        return np.sum(y_pred == y_test) / len(y_test)

    acc = accuracy(y_pred, y_test)
    print(acc)


