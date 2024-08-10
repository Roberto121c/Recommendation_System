import pickle
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

# Load pre-saved dictionaries that map user-movie ratings
with open('user2movierating.pkl', 'rb') as f:
    user2movierating = pickle.load(f)

with open('movie2userrating.pkl', 'rb') as f:
    movie2userrating = pickle.load(f)

with open('usermovie2rating.pkl', 'rb') as f:
    usermovie2rating = pickle.load(f)
    
with open('movie2userrating_test.pkl', 'rb') as f:
    movie2userrating_test = pickle.load(f)

# Initialize parameters for the model: user factors W, user biases b, item factors U, item biases c
def initialize_parameters(N, M, K):
    W = np.random.rand(N, K)
    b = np.zeros(N)
    U = np.random.rand(M, K)
    c = np.zeros(M)
    return W, b, U, c

# Calculate the loss function (mean squared error) over the provided data
def get_loss(W, U, b, c, mu, m2u):
    N = 0
    sse = 0
    for j, (u_ids, r) in m2u.items():
        p = W[u_ids].dot(U[j]) + b[u_ids] + c[j] + mu  # Predicted ratings
        delta = p - r
        sse += delta.dot(delta)  # Sum of squared errors
        N += len(r)
    return sse / N  # Return mean squared error

# Update the user factors (W) and biases (b) with movie factors as constant
def update_user_factors(W, U, b, c, mu, user2movierating, reg, K):
    N = len(W)
    for i in range(N):
        m_ids, r = user2movierating[i]
        matrix = U[m_ids].T.dot(U[m_ids]) + np.eye(K) * reg  # Regularized least squares matrix for user i
        vector = (r - b[i] - c[m_ids] - mu).dot(U[m_ids])  # Target vector for user i
        bi = (r - U[m_ids].dot(W[i]) - c[m_ids] - mu).sum()  # Update the bias term for user i

        W[i] = np.linalg.solve(matrix, vector)
        b[i] = bi / (len(m_ids) + reg)
    return W, b

# Update the item factors (U) and biases (c) with user factor as constant
def update_item_factors(W, U, b, c, mu, movie2userrating, reg, K):
    M = len(U)
    for j in range(M):
        try:
            u_ids, r = movie2userrating[j]
            matrix = W[u_ids].T.dot(W[u_ids]) + np.eye(K) * reg  # Regularized least squares matrix for item j
            vector = (r - b[u_ids] - c[j] - mu).dot(W[u_ids])  # Target vector for item j
            cj = (r - W[u_ids].dot(U[j]) - b[u_ids] - mu).sum()  # Update the bias term for item j

            U[j] = np.linalg.solve(matrix, vector)  # Solve for the new item factors U[j]
            c[j] = cj / (len(u_ids) + reg)  # Update the bias for item j using regularization
        except KeyError:
            pass  # If the item has no ratings, skip it
    return U, c

# Train the model using alternating least squares with early stopping
def train_model(user2movierating, movie2userrating, movie2userrating_test, K=10, reg=0.1, epochs=25, patience=5):
    N = len(user2movierating)  # Number of users
    M = len(movie2userrating)  # Number of items
    W, b, U, c = initialize_parameters(N, M, K)  # Initialize parameters
    mu = np.mean(list(usermovie2rating.values()))  # Global average rating

    train_losses = []
    test_losses = []
    best_loss = float('inf')
    no_improvement = 0

    for epoch in range(epochs):  # Iterate through epochs
        print(f'######\nEpoch: {epoch}')
        start_time = datetime.now()

        # Update user factors W and biases b
        W, b = update_user_factors(W, U, b, c, mu, user2movierating, reg, K)
        print(f"Updated W and b: {datetime.now() - start_time}")

        # Update item factors U and biases c
        U, c = update_item_factors(W, U, b, c, mu, movie2userrating, reg, K)
        print(f"Updated U and c: {datetime.now() - start_time}")

        # Calculate and store training and testing losses
        train_loss = get_loss(W, U, b, c, mu, movie2userrating)
        test_loss = get_loss(W, U, b, c, mu, movie2userrating_test)
        train_losses.append(train_loss)
        test_losses.append(test_loss)

        print(f"Train loss: {train_loss}, Test loss: {test_loss}")

        # Early stopping condition based on test loss
        if test_loss < best_loss:
            best_loss = test_loss
            no_improvement = 0
        else:
            no_improvement += 1
            if no_improvement >= patience:
                print(f"Early stopping on epoch {epoch}")
                break

    # Plot the training and testing loss over epochs
    plt.plot(train_losses, label='train')
    plt.plot(test_losses, label='test')
    plt.legend()
    plt.show()
    
    plt.savefig('MF_loss_curve.jpg', dpi=300, bbox_inches='tight')

    return W, U, b, c, mu, train_losses, test_losses

# Train the model with specified parameters
W, U, b, c, mu, train_losses, test_losses = train_model(user2movierating, movie2userrating, movie2userrating_test, K=10, reg=0.1, epochs=25, patience=5)

# Save the model parameters to a file
with open('MF_model', 'wb') as f:
    pickle.dump({'W': W, 'U': U, 'b': b, 'c': c, 'mu': mu}, f)