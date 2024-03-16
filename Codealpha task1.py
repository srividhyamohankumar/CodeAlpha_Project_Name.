#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


# In[12]:


# Load the dataset
data = pd.read_csv("C:\\Users\\srivi\\OneDrive\\Desktop\\CODEALPHA\\archive (2)\\updated dataset.csv")

# Handle missing values
data.dropna(inplace=True)

# Encode categorical variables
le = LabelEncoder()
data['S.no'] = le.fit_transform(data['S.no'])
data['Age'] = le.fit_transform(data['Age'])
data['Sex'] = le.fit_transform(data['Sex'])
data['Job'] = le.fit_transform(data['Job'])
data['Housing'] = le.fit_transform(data['Housing'])
data['Saving accounts'] = le.fit_transform(data['Saving accounts'])
data['Checking account'] = le.fit_transform(data['Checking account'])
data['Credit amount'] = le.fit_transform(data['Credit amount'])
data['Duration'] = le.fit_transform(data['Duration'])
data['Purpose'] = le.fit_transform(data['Purpose'])

# Split the data into features (X) and target (y)
X = data.drop('Credit amount', axis=1)
y = data['Credit amount']


# In[13]:


from sklearn.ensemble import RandomForestClassifier

# Initialize the model
model = RandomForestClassifier(n_estimators=100, random_state=42)


# In[14]:


from sklearn.model_selection import train_test_split

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model.fit(X_train, y_train)


# In[17]:


from sklearn.metrics import accuracy_score, confusion_matrix

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)

# Calculate the confusion matrix
confusion_matrix = confusion_matrix(y_test, y_pred)


# In[36]:


from sklearn.model_selection import GridSearchCV, StratifiedKFold

# Define the parameter grid
param_grid = {
    'n_estimators': [10, 50, 100],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Define the cross-validation strategy
cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)

# Initialize the grid search
grid_search = GridSearchCV(model, param_grid, cv=cv, scoring='accuracy', verbose=2, n_jobs=-1)

# Fit the grid search to the training set
grid_search.fit(X_train, y_train)

# Print the best parameters and the best score
print("Best parameters: ", grid_search.best_params_)
print("Best score: ", grid_search.best_score_)


# In[ ]:




