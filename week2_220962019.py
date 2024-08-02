#!/usr/bin/env python
# coding: utf-8

# In[8]:


#Q1
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_axes([0, 0, 1, 1])
x = [0, 20, 40, 60, 80, 100]
y = [0, 50,100, 150, 200,250] 
ax.plot(x, y)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('Title')
plt.show()


# In[9]:


#Q2
import matplotlib.pyplot as plt
x = [0, 20, 40, 60, 80, 100]
y = [0, 50, 100, 150, 200, 250]
fig = plt.figure()
ax1 = fig.add_axes([0, 0, 1, 1])
ax1.plot(x, y)
ax1.set_title('Axes 1')
ax2 = fig.add_axes([0.2, 0.5, 0.2, 0.2])
ax2.plot(x, y)
ax2.set_title('Axes 2')
plt.show()


# In[16]:


#Q3
import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv('company_sales_data.csv')
months = df['month_number']
total_profit= df['total_profit']
plt.figure(figsize=(6, 6))
plt.plot(months, total_profit, marker='o', linestyle='-', color='b')
plt.xlabel('Month Number')
plt.ylabel('Total Profit')
plt.title('Total Profit for Each Month')
plt.grid(True)
plt.show()


# In[19]:


#Q4
import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv('company_sales_data.csv')
months = df['month_number']
total_profit = df['total_profit']
plt.figure(figsize=(10, 6))
plt.plot(months, total_profit, 
         linestyle='--',         # Dotted line style
         color='red',            # Line color
         marker='o',             # Circle marker
         markerfacecolor='red',  # Marker color
         markeredgewidth=1,      # Marker edge width
         linewidth=3,           # Line width
         label='Total Profit')   # Label for the legend
plt.xlabel('Month Number')
plt.ylabel('Sold Units Number')  # Updated y label as per request
plt.title('Total Profit for Each Month')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()


# In[23]:


#ADDITIONAL QUESTION 1
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('company_sales_data.csv')

months = df['month_number']
facecream = df['facecream']
facewash = df['facewash']
bathingsoap = df['bathingsoap']
shampoo = df['shampoo']
moisturizer = df['moisturizer']

plt.figure(figsize=(12, 8))


plt.plot(months, facecream, marker='o', linestyle='-', color='b', label='Face Creams')
plt.plot(months, facewash, marker='s', linestyle='--', color='g', label='Face Wash')
plt.plot(months, bathingsoap, marker='^', linestyle='-.', color='r', label='Bathing Soaps')
plt.plot(months, shampoo, marker='D', linestyle=':', color='c', label='Shampoo')
plt.plot(months, moisturizer, marker='x', linestyle='-', color='m', label='Moisturizer')

plt.xlabel('Month Number')
plt.ylabel('Units Sold')
plt.title('Monthly Sales Data for Each Product')
plt.legend(loc='best')
plt.grid(True)
plt.show()


# In[26]:


#ADDITIONAL QUESTION 2
import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv('company_sales_data.csv')
latest_month = df['month_number'].max()
df_last_year = df[df['month_number'] == latest_month]
total_sales_last_year = {
    'Face Cream': df_last_year['facecream'].sum(),
    'Face Wash': df_last_year['facewash'].sum(),
    'Bathing Soap': df_last_year['bathingsoap'].sum(),
    'Shampoo': df_last_year['shampoo'].sum(),
    'Moisturizer': df_last_year['moisturizer'].sum()
}
plt.figure(figsize=(8, 8))
plt.pie(total_sales_last_year.values(), 
        labels=total_sales_last_year.keys(),
        autopct='%1.1f%%', 
        startangle=140,   
        colors=['r', 'b', 'g', 'c', 'm'], 
        wedgeprops={'edgecolor': 'black'})
plt.title('Total Sales Distribution for the Last Year')
plt.show()


# In[ ]:




