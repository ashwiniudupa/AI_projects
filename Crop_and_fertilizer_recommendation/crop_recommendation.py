import os
import pandas as pd
import numpy as np

from sklearn import metrics
from sklearn.metrics import mean_squared_error
from sklearn import feature_selection
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import random

"""
# Visualization
import matplotlib
# %matplotlib inline
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from plotly import graph_objects as go

crop = pd.read_csv(r"Crop_recommendation.csv")
crop.head()

corr=crop.drop(['label'],axis=1).corr()
corr

sns.heatmap(corr, annot = True, cbar = True, cmap = 'coolwarm')
crop['label'].value_counts()

# Distribution of 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall' features
features = ['P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
for feature in features:
    sns.histplot(crop[feature], kde=True)
    plt.title(f'Distribution of {feature}')
    plt.show()

# Boxplot for 'N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall' features
features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
for feature in features:
    sns.boxplot(x=crop[feature])
    plt.title(f'Boxplot of {feature}')
    plt.show()


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Setting plot aesthetics
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = [10, 8]


# Box Plots for a variable against crop types
sns.boxplot(x='label', y='N', data=crop)
plt.xticks(rotation=90)
plt.title('Nitrogen Content across Different Crops')
plt.show()

nutrients = ['N', 'P', 'K']
avg_nutrients = crop.groupby('label')[nutrients].mean().reset_index()

fig = go.Figure()

for i in range(len(avg_nutrients)):
    fig.add_trace(go.Scatterpolar(
        r=avg_nutrients.iloc[i, 1:],
        theta=nutrients,
        fill='toself',
        name=avg_nutrients['label'][i]
    ))

fig.update_layout(
    polar=dict(radialaxis=dict(visible=True,)),
    showlegend=True
)
fig.show()
"""

###################### FEATURE ENGINEERING  #############################
## Converting Categorical varibales to a integer format
crop_dict = {
    'rice': 1,
    'maize': 2,
    'jute': 3,
    'cotton': 4,
    'coconut': 5,
    'papaya': 6,
    'orange': 7,
    'apple': 8,
    'muskmelon': 9,
    'watermelon': 10,
    'grapes': 11,
    'mango': 12,
    'banana': 13,
    'pomegranate': 14,
    'lentil': 15,
    'blackgram': 16,
    'mungbean': 17,
    'mothbeans': 18,
    'pigeonpeas': 19,
    'kidneybeans': 20,
    'chickpea': 21,
    'coffee': 22
}
crop['crop_num']=crop['label'].map(crop_dict)
crop.head()
crop=crop.drop('label',axis=1)

"""## Get `TRAIN` and `TEST` dataset"""
X=crop.drop('crop_num',axis=1)
Y=crop['crop_num']

X.head()
Y.head()
X.shape
Y.shape

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42)

X_train.shape
X_test.shape

#####  Scale the features using MinMaxScaler  ###################
X_train
"""### `MinMaxScaler`, which transforms the data into the range [0, 1] by default."""
from sklearn.preprocessing import MinMaxScaler
ms = MinMaxScaler()
ms.fit(X_train)     # estimate the minimum and maximum observable values through fit()
X_train = ms.transform(X_train)    # Transform the data 
X_test = ms.transform(X_test)

##### ASHWINI: Why below needed? Ignore 
X_train
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(X_train)
X_train = sc.transform(X_train)
X_test = sc.transform(X_test)
X_train

########### MODEL TRAINING ##################
## Performance of multiple classifiers on a given dataset based on their accuracy scores
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import ExtraTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score

# create instances of all models
models = {
    'Logistic Regression': LogisticRegression(),
    'Naive Bayes': GaussianNB(),
    'Support Vector Machine': SVC(),
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'Bagging': BaggingClassifier(),
    'AdaBoost': AdaBoostClassifier(),
    'Gradient Boosting': GradientBoostingClassifier(),
    'Extra Trees': ExtraTreeClassifier(),
}

for name, md in models.items():
    md.fit(X_train,Y_train)
    ypred = md.predict(X_test)
    print(f"{name}  with accuracy : {accuracy_score(Y_test,ypred)}")

"""## Model Selection for Crop Recommendation Project
After applying multiple machine learning models to the crop recommendation project, Naive Bayes and Random Forest performed better. 
Naive Bayes is selected. (Read proj report pdf)
"""

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.naive_bayes import GaussianNB

# Instantiate the Naive Bayes classifier
nb_classifier = GaussianNB()

# Fit the classifier to the training data
nb_classifier.fit(X_train, Y_train)

# Predict the labels of the test set
y_pred = nb_classifier.predict(X_test)

# Calculate evaluation metrics
precision = precision_score(Y_test, y_pred, average='weighted')
recall = recall_score(Y_test, y_pred, average='weighted')
f1 = f1_score(Y_test, y_pred, average='weighted')
accuracy = accuracy_score(Y_test, y_pred)

# Display evaluation metrics
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1-score: {f1}')
print(f'Accuracy: {accuracy}')

from sklearn.metrics import confusion_matrix
import seaborn as sns

# Calculate the confusion matrix
conf_matrix = confusion_matrix(Y_test, y_pred)

# Create a heatmap for the confusion matrix
plt.figure(figsize=(10, 12))
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='d',
            xticklabels=crop['crop_num'].unique(),
            yticklabels=crop['crop_num'].unique())
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix for Naive Bayes Classifier')
plt.show()


######## ADVANCE MODEL IMPLEMENTATION  #######
##### NEURAL NETWORK - Multi-layer Perceptron classifier  ##########
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

# Load your data
crop_data = pd.read_csv('Crop_recommendation.csv')
# Preprocessing
features = crop_data.drop('label', axis=1)
target = crop_data['label']
# Encoding the target variable
label_encoder = LabelEncoder()
Y_encoded = label_encoder.fit_transform(target)
# Scaling the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)

# Splitting the dataset
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y_encoded, test_size=0.2, random_state=42)
# Neural Network Model
mlp = MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu', solver='adam', max_iter=100, random_state=42)
# Training the model
mlp.fit(X_train, Y_train)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# Predictions
Y_pred = mlp.predict(X_test)
# Compute evaluation metrics
accuracy = accuracy_score(Y_test, Y_pred)
precision = precision_score(Y_test, Y_pred, average='weighted')
recall = recall_score(Y_test, Y_pred, average='weighted')
f1 = f1_score(Y_test, Y_pred, average='weighted')

# Print evaluation metrics
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1-score: {f1}")

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Compute the confusion matrix
conf_matrix = confusion_matrix(Y_test, Y_pred)

# Display the confusion matrix using seaborn heatmap
plt.figure(figsize=(10, 12))
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='g', cbar=False)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix - Neural Network')
plt.show()

import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import log_loss

# Load the dataset
df = pd.read_csv('Crop_recommendation.csv')

# Separate features and target variable
X = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
y = df['label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

from sklearn.metrics import classification_report
import tracemalloc

# Function to train and evaluate a model
def train_and_evaluate_model(model, X_train, y_train, X_test, y_test, model_name):
    tracemalloc.start()  # Start tracing the memory allocation

    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time

    ######  Memory usage  ##################
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()  # Stop tracing the memory allocation

    # Predictions and probabilities on test set
    y_pred = model.predict(X_test)
    y_probs = model.predict_proba(X_test)

    # Log Loss
    model_log_loss = log_loss(y_test, y_probs)

    print(f"{model_name} training time: {training_time:.4f} seconds")
    print(f"{model_name} memory usage: {current / 10**6:.4f} MB; Peak: {peak / 10**6:.4f} MB")
    print(f"{model_name} Log Loss: {model_log_loss:.4f}\n")

    # Detailed classification report
    print(classification_report(y_test, y_pred))

# Train and evaluate Naive Bayes
nb_model = GaussianNB()
train_and_evaluate_model(nb_model, X_train_scaled, y_train, X_test_scaled, y_test, 'Naive Bayes')

# Train and evaluate Neural Network
nn_model = MLPClassifier(random_state=42)
train_and_evaluate_model(nn_model, X_train_scaled, y_train, X_test_scaled, y_test, 'Neural Network')

# Import libraries
import pandas as pd
import matplotlib.pyplot as plt

# Create a DataFrame with the data
data = {
    'Model': ['Naive Bayes', 'Neural Network'],
    'Training Time (s)': [0.0348, 7.2994],
    'Memory Usage (MB)': [0.0092, 0.2391],
    'Peak Memory Usage (MB)': [0.1701, 0.8956],
    'Log Loss': [0.0165, 0.0854]
}

df = pd.DataFrame(data)

# Create a bar chart for training time
plt.figure(figsize=(10, 6))
plt.bar(df['Model'], df['Training Time (s)'], color=['blue', 'green'])
plt.xlabel('Model')
plt.ylabel('Training Time (s)')
plt.title('Training Time Comparison')
plt.yscale('log')  # Logarithmic scale for y-axis
plt.grid(axis='y')
plt.show()

# Create a bar chart for memory usage
plt.figure(figsize=(10, 6))
plt.bar(df['Model'], df['Memory Usage (MB)'], color=['blue', 'green'])
plt.xlabel('Model')
plt.ylabel('Memory Usage (MB)')
plt.title('Memory Usage Comparison')
plt.grid(axis='y')
plt.show()

# Create a bar chart for peak memory usage
plt.figure(figsize=(10, 6))
plt.bar(df['Model'], df['Peak Memory Usage (MB)'], color=['blue', 'green'])
plt.xlabel('Model')
plt.ylabel('Peak Memory Usage (MB)')
plt.title('Peak Memory Usage Comparison')
plt.grid(axis='y')
plt.show()

# Create a bar chart for log loss
plt.figure(figsize=(10, 6))
plt.bar(df['Model'], df['Log Loss'], color=['blue', 'green'])
plt.xlabel('Model')
plt.ylabel('Log Loss')
plt.title('Log Loss Comparison')
plt.grid(axis='y')
plt.show()

"""################### Comparative Analysis of Naive Bayes and Neural Network Models"""
import matplotlib.pyplot as plt
import pandas as pd

# Data
models = ['Naive Bayes', 'Neural Network']
precision = [0.9958, 0.9715, ]
recall = [0.9955, 0.9682]
f1_score = [0.9954,0.9687]
accuracy = [0.9955, 0.9682]

# Creating DataFrame
df = pd.DataFrame({'Precision': precision, 'Recall': recall, 'F1-score': f1_score, 'Accuracy': accuracy}, index=models)

# Plotting
ax = df.plot(kind='bar', figsize=(12, 7))
ax.set_title('Model Performance Comparison')
ax.set_ylabel('Scores')
ax.set_xlabel('Models')
plt.xticks(rotation=0)
plt.grid(True)

# Adjusting y-axis scale to better visualize differences
ax.set_ylim(0.96, 1.0)

plt.show()

"""############### Naive Bayes is better than Neural N/w because of memory, training time, etc """

## Export model to Pickle file
# import pickle
# pickle_out = open('naive_bayes_model.pkl', 'wb')
# pickle.dump(nb_classifier,pickle_out)
# pickle_out.close()