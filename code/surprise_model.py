from surprise import Dataset, Reader
from surprise import SVD
from surprise.model_selection import train_test_split
from surprise.model_selection import KFold
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
import pickle

df = pd.read_csv('reduced_rating.csv')

# Create a Reader object to specify the range of ratings
reader = Reader(rating_scale=(df['rating'].min(), df['rating'].max()))

# Load the dataset from the dataframe
data = Dataset.load_from_df(df[['userId', 'movieId', 'rating']], reader)

# Split the dataset into training and testing sets
trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

# Define the SVD model
algo = SVD(n_factors=10, reg_all=0.1, n_epochs=25)

# Train the model
algo.fit(trainset)

# Predict on the test set
predictions = algo.test(testset)

# Calculate the MSE
y_true = np.array([pred.r_ui for pred in predictions])
y_pred = np.array([pred.est for pred in predictions])
mse = mean_squared_error(y_true, y_pred)
print(f"Test MSE: {mse}")

# Implementation of K-Fold Cross-Validation with Early Stopping
kf = KFold(n_splits=5)
train_losses = []
test_losses = []
best_loss = float('inf')
no_improvement = 0
patience = 5

for trainset_cv, testset_cv in kf.split(data):
    algo.fit(trainset_cv)
    
    # Predictions on the training set
    train_predictions = algo.test(trainset_cv.build_testset())
    y_true_train = np.array([pred.r_ui for pred in train_predictions])
    y_pred_train = np.array([pred.est for pred in train_predictions])
    train_mse = mean_squared_error(y_true_train, y_pred_train)
    
    # Predictions on the test set
    test_predictions = algo.test(testset_cv)
    y_true_test = np.array([pred.r_ui for pred in test_predictions])
    y_pred_test = np.array([pred.est for pred in test_predictions])
    test_mse = mean_squared_error(y_true_test, y_pred_test)
    
    train_losses.append(train_mse)
    test_losses.append(test_mse)
    
    print(f"Train MSE: {train_mse}, Test MSE: {test_mse}")
    
    # Early stopping
    if test_mse < best_loss:
        best_loss = test_mse
        no_improvement = 0
    else:
        no_improvement += 1
        if no_improvement >= patience:
            print("Early stopping triggered.")
            break

# Plot the losses (MSE)
plt.plot(train_losses, label='Train MSE')
plt.plot(test_losses, label='Test MSE')
plt.xlabel('Fold')
plt.ylabel('MSE')
plt.legend()
plt.show()

plt.savefig('svd_loss_curve.jpg', dpi=300, bbox_inches='tight')

    
# Save the model parameters to a file
with open('svd_model.pkl', 'wb') as f:
    pickle.dump(algo, f)
