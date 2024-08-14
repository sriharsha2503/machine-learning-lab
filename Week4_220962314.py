#!/usr/bin/env python
# coding: utf-8

# In[32]:


import  pandas as pd
import matplotlib.pyplot as plt


# In[33]:


data = {'Temp':[50,50,50,70,70,70,80,80,80,90,90,90,100,100,100],'Yield':[3.3,2.8,2.9,2.3,2.6,2.1,2.5,2.9,2.4,3.0,3.1,2.8,3.3,3.5,3.0]}


# In[34]:


df = pd.DataFrame(data)
df.to_csv('file.csv')


# In[35]:


import numpy as np
import pandas as pd

df = pd.read_csv('file.csv')

X = df['Temp'].values
y = df['Yield'].values

#Pedhazur formula
X_mean = np.mean(X)
y_mean = np.mean(y)
XY_mean = np.mean(X * y)
X_squared_mean = np.mean(X ** 2)

# Compute B1 (slope) and B0 (intercept)
B1 = (XY_mean - X_mean * y_mean) / (X_squared_mean - X_mean ** 2)
B0 = y_mean - B1 * X_mean

# Compute predicted responses
y_pred = B0 + B1 * X

# Compute RMSE
RMSE = np.sqrt(np.mean((y - y_pred) ** 2))


print(f"Intercept (B0): {B0:.2f}")
print(f"Slope (B1): {B1:.2f}")
print(f"RMSE: {RMSE:.2f}")
print(X_mean,y_mean,XY_mean, X_squared_mean)


# In[36]:


import matplotlib.pyplot as plt

plt.scatter(X, y, color='red', label='Data Points')

plt.plot(X, y_pred, color='blue', label='Regression Line')

plt.xlabel('Time')
plt.ylabel('Yield')
plt.title('Time vs Yield')
plt.legend()
plt.grid(True)
plt.show()


# In[37]:


import numpy as np
import pandas as pd

df = pd.read_csv('file.csv')

X = df['Temp'].values
y = df['Yield'].values


# mat1=np.array([[len(X),np.sum(X)],[np.sum(X),np.sum(X**2)]])
# mat2=np.array([[np.sum(y),np.sum(X*y)]])
# coeffs=np.dot(np.linalg.inv(mat1),mat2.T)
# b0_mat,b1_mat=coeffs[0,0],coeffs[1,0]

# y_mat = b0_mat + b1_mat * X
# squared_errors = (y - y_mat) ** 2
# rmse_mat = np.sqrt(np.mean(squared_errors))

# print(f"{b1_mat}x + {b0_mat}")
# print(y_mat)
# print(rmse_mat)


mat1 = np.array([[len(X),np.sum(X),np.sum(X**2)],[np.sum(X),np.sum(X**2),np.sum(X**3)],[np.sum(X**2),np.sum(X**3),np.sum(X**4)]])
mat2 = np.array([[np.sum(y),np.sum(X*y),np.sum((X**2)*y)]])
print(mat1)
print(mat2)
coeffs = np.dot(np.linalg.inv(mat1),mat2.T)
b0,b1,b2 = coeffs[0,0],coeffs[1,0],coeffs[2,0]
y_prd = b0 + b1*X + b2*(X**2)
squared_err = (y-y_prd)**2
rmse_mat = np.sqrt(np.mean(squared_err))
print(np.mean(squared_err))
print(f"{b0}+{b1}x+{b2}x^2")
print(y_prd)
print(rmse_mat)
print(len(X))
print(np.sum(X))
print(np.sum(X**2))
print(np.sum(X**3))
print(np.sum(X**4))
print(np.sum(y))
print(np.sum(X*y))
print(np.sum((X**2)*y))


# In[38]:


import matplotlib.pyplot as plt

plt.scatter(X, y, color='red', label='Data Points')

plt.plot(X, y_prd, color='blue', label='Regression Line')
plt.xlabel('Time')
plt.ylabel('Yield')
plt.title('Time vs Yield')
plt.legend()
plt.grid(True)
plt.show()


# # Q2
# 

# In[39]:


df = pd.read_csv('heartattack.csv')
df.head()


# In[42]:


X1 = df['Area']
X2 = df['X2']
y = df['Infarc']
X3 = df['X3']

mat1 = np.array([[len(X1),np.sum(X1),np.sum(X2),np.sum(X3)],
                 [np.sum(X1),np.sum(X1**2),np.sum(X1*X2),np.sum(X1*X3)],
                 [np.sum(X2),np.sum(X1*X2),np.sum(X2**2),np.sum(X3*X2)],
                 [np.sum(X3),np.sum(X1*X3),np.sum(X2*X3),np.sum(X3**2)]])
mat2 = np.array([[np.sum(y),np.sum(X1*y),np.sum(X2*y),np.sum(X3*y)]])
print(mat1)
print(mat2)
coeffs = np.dot(np.linalg.inv(mat1),mat2.T)
print(mat1)
print(mat2)
b0,b1,b2,b3 = coeffs[0,0],coeffs[1,0],coeffs[2,0],coeffs[3,0]
y_prd = b0 + b1*X1+ b2*(X2) +b3*X3
squared_err = (y-y_prd)**2
rmse_mat = np.sqrt(np.mean(squared_err))
print(np.mean(squared_err))
print(f"{b0}+{b1}x1+{b2}x2+{b3}x3")
print(y_prd)
print(rmse_mat)
print(len(X))
print(np.sum(X1))
print(np.sum(X2))
print(np.sum(X3))
print(np.sum(X2*X3))
print(np.sum(y))
print(np.sum(X1*y))
print(np.sum((X2*y)))
print(np.sum(X3*y))
print(np.sum(X1*X2))
print(np.sum(X1*X3))


# In[28]:


import matplotlib.pyplot as plt

plt.scatter(X1, y, color='red', label='Data Points')

plt.plot(X1, y_prd, color='blue', label='Regression Line')

plt.xlabel('Time')
plt.ylabel('Yield')
plt.title('Time vs Yield')
plt.legend()
plt.grid(True)
plt.show()


# In[43]:


X1 = df['Area']
X2 = df['X2']
y = df['Infarc']
X3 = df['X3']
X4 = 
mat1 = np.array([[len(X1),np.sum(X1),np.sum(X2),np.sum(X3),np.sum(X4)],
                 [np.sum(X1),np.sum(X1**2),np.sum(X1*X2),np.sum(X1*X3),np.sum(X1*X4)],
                 [np.sum(X2),np.sum(X1*X2),np.sum(X2**2),np.sum(X3*X2),np.sum(X4*X2)],
                 [np.sum(X3),np.sum(X1*X3),np.sum(X2*X3),np.sum(X3**2),np.sum(X4*X3)],
                 [np.sum(X4),np.sum(X1*X4),np.sum(X2*X4),np.sum(X3*X4),np.sum(X4**2)]])
mat2 = np.array([[np.sum(y),np.sum(X1*y),np.sum(X2*y),np.sum(X3*y),np.sum(X4*y)]])
print(mat1)
print(mat2)
coeffs = np.dot(np.linalg.inv(mat1),mat2.T)
print(mat1)
print(mat2)
b0,b1,b2,b3 = coeffs[0,0],coeffs[1,0],coeffs[2,0],coeffs[3,0]
y_prd = b0 + b1*X1+ b2*(X2) +b3*X3
squared_err = (y-y_prd)**2
rmse_mat = np.sqrt(np.mean(squared_err))
print(np.mean(squared_err))
print(f"{b0}+{b1}x1+{b2}x2+{b3}x3")
print(y_prd)
print(rmse_mat)
print(len(X))
print(np.sum(X1))
print(np.sum(X2))
print(np.sum(X3))
print(np.sum(X2*X3))
print(np.sum(y))
print(np.sum(X1*y))
print(np.sum((X2*y)))
print(np.sum(X3*y))
print(np.sum(X1*X2))
print(np.sum(X1*X3))


# In[ ]:




