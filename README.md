# Machine Learning Algorithms From Scratch

This project contains from-scratch implementations of core machine learning algorithms using only NumPy and standard Python libraries.
The goal of the project is to understand how popular ML algorithms work internally without relying on high-level libraries such as scikit-learn.

## Implemented Algorithms

- K-Nearest Neighbors (KNN) with PCA dimensionality reduction
- Linear Regression (Gradient Descent)
- Logistic Regression (Binary Classification)
- Decision Tree Classifier (Entropy & Information Gain)

Each algorithm is implemented in its own Python file and includes an executable example in a `__main__` block.

## Repository Structure
```bash
.
│
├── knn/
│   └── KNN.py                                  # Implements KNN algorithm with PCA
├── linear/
│   ├── LinearRegression.py                     # Implements LinearRegression algorithm
│   └── LogisticRegression.py                   # Implements LogisticRegression algorithm
├── trees/
│   └── DecisionTree.py                         # Implements DecisionTree algorithm
├── README.md
└── requirements.txt
```

## Getting Started
```bash
pip install -r requirements.txt
```

## How to Run the Code

### Decision Tree
```bash
python -m trees.DecisionTree
```
Runs the Decision Tree classifier on the Breast Cancer dataset and prints classification accuracy.

### K-Nearest Neighbors (with PCA)
```bash
python -m knn.KNN
```
- Trains a KNN classifier on the Iris dataset
- Applies PCA for dimensionality reduction
- Displays a 3D PCA visualization
- Prints test accuracy and sample predictions

### Linear Regression
```bash
python -m linear.LinearRegression
```
- Trains a linear regression model on a synthetic dataset
- Plots the data and regression line
- Prints Mean Squared Error (MSE)

### Logistic Regression
```bash
python -m linear.LogisticRegression
```
- Trains a logistic regression classifier on the Breast Cancer dataset
- Prints classification accuracy