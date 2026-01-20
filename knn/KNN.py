import numpy as np 
from collections import Counter

class KNN():
    def __init__(self, k = None):
        self.k = k
        
    def fit(self, X_train,y_train):
        self.X_train = X_train
        self.y_train = y_train

        self.X_train_pca, self.eigenvalues, self.eigenvectors, self.column_means_train = self.PCA(X_train, k = 3)

    def PCA(self, X, k = None):
        # 1. center data around X
        X_centered = X.copy()
        column_means_train = []

        for col in range(X.shape[1]):  # for col in X.shape[columns]
            column = X[:,col]
            column_mean = np.mean(column) # all rows in col 
            column_means_train.append(column_mean) # save it because we need it for testing 

            X_centered[:, col] = X[:, col] - column_mean

        n_samples = X.shape[0]
        covariance_matrix = (1/(n_samples-1)) * np.dot(X_centered.T, X_centered)
        # now compute eigenvectors and eigenvalues of S
        eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
        sorted_indices = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:,sorted_indices]

        if k is not None:
            eigenvectors = eigenvectors[:, :k]
            eigenvalues = eigenvalues[:k]

        X_pca = np.dot(X_centered, eigenvectors)

        return X_pca, eigenvalues, eigenvectors, column_means_train



    def predict(self, X):
        return [self._predict(x) for x in X]

    def _predict(self, X):
        # compute the distance
        
        X_centered = X - self.column_means_train
        X_pca_test = np.dot(X_centered, self.eigenvectors)
            
            
        distances = [self.get_euclidean_distance(X_pca_test, x_train_pca) for x_train_pca in self.X_train_pca]

    
        # get the closest k
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]

        # majority vote
        most_common = Counter(k_nearest_labels).most_common()
        return most_common[0][0]

    def get_euclidean_distance(self, x1,x2): 
        return np.sqrt(np.sum((x1-x2)**2))
    
if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split

    # 1. Load the Iris dataset
    iris = load_iris()
    X = iris.data          # shape: (150, 4)
    y = iris.target        # shape: (150,)

    # 2. Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 3. Initialize and train KNN with PCA
    model = KNN(k=3)   # You can try different k values
    model.fit(X_train, y_train)

    # --- PCA 3D Plot ---
    X_pca_3d = model.X_train_pca[:, :3]  # first 3 PCA components

    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111, projection='3d')

    for class_label in np.unique(model.y_train):
        indices = model.y_train == class_label
        ax.scatter(
            X_pca_3d[indices, 0],
            X_pca_3d[indices, 1],
            X_pca_3d[indices, 2],
            label=f"Class {class_label}",
            s=60
        )

    ax.set_xlabel("PCA Component 1")
    ax.set_ylabel("PCA Component 2")
    ax.set_zlabel("PCA Component 3")
    ax.set_title("Iris Dataset PCA Projection (Training Data)")
    ax.legend()
    plt.show()

    # 4. Make predictions on the test data
    y_pred = model.predict(X_test)

    # 5. Evaluate accuracy
    accuracy = np.mean(y_pred == y_test)
    print(f" Accuracy on Iris test set: {accuracy * 100:.2f}%")

    # 6. Print a few sample predictions vs actual
    print("Predicted labels:", y_pred[:10])
    print("True labels:     ", y_test[:10])
