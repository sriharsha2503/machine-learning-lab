#!/usr/bin/env python
# coding: utf-8

# # QUESTION 1

# In[20]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



# In[21]:


data = {
    'mother': [58,62,60,64,67,70],
    'daughter': [60,60,58,60,70,72]
}

df = pd.DataFrame(data)

csv_file = 'height.csv'
df.to_csv(csv_file, index=False)
df


# In[22]:


X=df['mother'].values
Y=df['daughter'].values
errors = []


# # gradientdecent

# In[24]:


def gradient_descent(X, Y, epochs, alpha, b1, b0):
    for _ in range(epochs):  
        for x, y in zip(X, Y):
            y_pred = b1 * x + b0
            error = y_pred - y
            b1 -= alpha * error * x
            b0 -= alpha * error
            errors.append(abs(y_pred-y))
        
    
    return b1, b0


# In[25]:


b1,b0 = gradient_descent(X,Y,4,0.0001,0,0)
y_pred=[b1*x + b0 for x in X]

plt.scatter(X, Y, color='blue', label='Original Data')
plt.plot(X, y_pred, color='red', label=f'Result from 20 epochs')
plt.legend()



# In[26]:


plt.plot(list(range(len(errors))), errors,linestyle='-',label='Gradient Descent')
plt.legend()


# In[27]:


print(f"Daughter's height when moms height is 63 : {b1*63 +b0}")


# 
# Predictions using sckit-learn
# 

# Question 2

# In[28]:


data = {
    'hours': [1,2,3,4,5,6,7,8],
    'pass': [0,0,0,0,1,1,1,1]
}

df = pd.DataFrame(data)

csv_file = 'passed.csv'
df.to_csv(csv_file, index=False)
df


# In[29]:


X=df['hours'].values
Y=df['pass'].values
losses = []



# In[30]:


def calc_loss(y_pred,y):
    return -(y*np.log(y_pred) + (1-y)*np.log(1-y_pred))



# In[31]:


def logistic_regression(X,Y,epochs,alpha,b1,b0):
    for _ in range(epochs):
        for x,y in zip(X,Y):
            z=b1*x + b0
            y_pred=1/(1+np.exp(-z))
            error=y_pred-y
            b1-=(alpha*error*x)
            b0-=(alpha*error)
        losses.append(calc_loss(y_pred,y))
    
    return b1,b0


# In[32]:


b1,b0 = logistic_regression(X,Y,500,0.01,0,0)
y_probs=[1/(1+np.exp(-(b1*x+b0))) for x in X]
y_binary = [1 if prob > 0.5 else 0 for prob in y_probs]

print(y_binary)


# In[33]:


plt.scatter(X,Y,color='blue',label='Actual')
plt.plot(X,y_binary,color='red',label='Predicted')
plt.legend()
plt.show()


# In[37]:


print(f"Probability of passing when studying for 3.5hrs : {1/(1+np.exp(-(w*3.5 +b)))}")
print(f"Probability of passing when studying for 7.5hrs : {1/(1+np.exp(-(w*7.5 +b)))}")


# In[35]:


plt.plot(list(range(len(losses))),losses,color='blue',label='LogLoss')
plt.legend()


# question 3

# In[38]:


data = {'x1':[4,2,1,3,1,6],'x2':[1,8,0,2,4,7],'y':[1,0,1,0,0,0]}
df = pd.DataFrame(data)
df.to_csv('q3data.csv')


# In[39]:


# Initialize parameters
b0 = 0.0
b1 = 0.0
b2 = 0.0
learning_rate = 0.01
iterations = 1000

# Lists to track log losses
log_losses = []

# Stochastic Gradient Descent
for iteration in range(iterations):
    for i, j, k in zip(df['x1'], df['x2'], df['y']):
        # Compute the prediction using the logistic function
        predict = 1 / (1 + np.exp(-(b0 + b1 * i + b2 * j)))
        error = predict - k
        
        # Update the parameters using the gradient of the loss function
        b0 -= learning_rate * error
        b1 -= learning_rate * error * i
        b2 -= learning_rate * error * j

    # Compute the log loss for this iteration
    predictions = 1 / (1 + np.exp(-(b0 + b1 * df['x1'] + b2 * df['x2'])))
    epsilon = 1e-15  # Small value to avoid log(0)
    predictions = np.clip(predictions, epsilon, 1 - epsilon)  # Clip predictions to avoid log(0)
    log_loss = -np.mean(df['y'] * np.log(predictions) + (1 - df['y']) * np.log(1 - predictions))
    log_losses.append(log_loss)

# Print the optimized parameters
print(f"Optimized b0: {b0}")
print(f"Optimized b1: {b1}")
print(f"Optimized b2: {b2}")

# Make predictions
df['predicted_prob'] = 1 / (1 + np.exp(-(b0 + b1 * df['x1'] + b2 * df['x2'])))

# Convert probabilities to binary outcomes
threshold = 0.5
df['predicted_class'] = (df['predicted_prob'] >= threshold).astype(int)

# Compute accuracy
correct_predictions = (df['predicted_class'] == df['y']).sum()
total_predictions = len(df)
accuracy = correct_predictions / total_predictions

print(f"Accuracy: {accuracy:.2f}")

# Plotting sigmoid function vs. x1 while keeping x2 fixed
x1_values = np.linspace(df['x1'].min() - 1, df['x1'].max() + 1, 100)

# Choose a fixed value for x2
fixed_x2 = df['x2'].mean()  # or choose any specific value from df['x2']

# Compute sigmoid function
y_values = 1 / (1 + np.exp(-(b0 + b1 * x1_values + b2 * fixed_x2)))

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(x1_values, y_values, label=f'Sigmoid Function (x2={fixed_x2:.2f})', color='blue')
plt.scatter(df['x1'], df['y'], color='red', label='Data points')
plt.xlabel('x1')
plt.ylabel('Probability')
plt.title('Sigmoid Function vs. x1 with x2 Fixed')
plt.legend()
plt.grid(True)
plt.show()

# Plotting log loss vs. iteration
plt.figure(figsize=(10, 6))
plt.plot(range(iterations), log_losses, color='blue', label='Log Loss')
plt.xlabel('Iteration')
plt.ylabel('Log Loss')
plt.title('Log Loss vs. Iteration')
plt.legend()
plt.grid(True)
plt.show()


# In[ ]:




