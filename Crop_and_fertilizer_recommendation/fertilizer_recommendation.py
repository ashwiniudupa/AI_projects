import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv(r"Fertilizer.csv")

df.head()
df.describe()
df['Fertilizer Name'].unique()
# Basic information about the dataset
print(df.info())
# Statistical summary of the dataset
print(df.describe())

"""## Exploratory Data Analysis (EDA)
"""

import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(22,6))
sns.countplot(x='Fertilizer Name', data = df)

# Setting aesthetics for better readability of plots
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = [10, 8]

# Plotting histograms for each feature
df.hist(bins=15, figsize=(15, 10))
plt.suptitle('Distribution of Features')
plt.show()

"""## Histogram Analysis of Soil Nutrient Data
"""
# Box plots for each feature
df.plot(kind='box', subplots=True, layout=(3,3), sharex=False, sharey=False, figsize=(15, 10))
plt.suptitle('Box Plot for Each Feature')
plt.show()

"""## Box Plot Analysis for Soil Nutrients
"""
corr = df.drop(columns=['Fertilizer Name']).corr()
corr
sns.heatmap(corr, annot = True, cbar = True, cmap = 'coolwarm')
# Plotting the distribution graphs of the variables
plt.figure(figsize=(15, 5))

# Enumerating through each numeric column for distribution plot
for i, column in enumerate(['Nitrogen', 'Potassium', 'Phosphorous'], start=1):
    plt.subplot(1, 3, i)
    sns.histplot(df[column], bins=20, kde=True)
    plt.title(f'Distribution of {column}')

plt.tight_layout()
plt.show()

# Removing the non-numeric column for correlation analysis
numeric_data = df.drop('Fertilizer Name', axis=1)

X = df.drop(columns=['Fertilizer Name'])
y = df['Fertilizer Name']

############### FEATURE ENGG ###############################
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,train_size=0.7,shuffle=True,random_state=42)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

X_train[0]

"""## Utilizing Random Forest in Fertilizer Recommendation Project
The decision to implement a Random Forest classifier, with hyperparameter tuning via GridSearchCV, in the Fertilizer Recommendation project is 
informed by several compelling advantages of this model.
### Why Random Forest?
**Robustness and Versatility**:
Random Forest is a robust and versatile ensemble learning method, suitable for both classification and regression tasks. 
**Handling of Complex Interactions**:
Unlike Naive Bayes, which assumes feature independence, Random Forest can capture complex interactions between features, 
which is often the case in agricultural datasets where factors such as nutrient levels and soil conditions may interact in complex ways to influence fertilizer requirements.
**Reduction of Overfitting**:
By utilizing multiple decision trees, Random Forest reduces the risk of overfitting, which can be a common problem with single decision trees. 
This makes the model more generalizable to new, unseen data.
"""

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=100,criterion = 'gini', random_state=42)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix,accuracy_score
cm = confusion_matrix(y_test,y_pred)
print(cm)
accuracy_score(y_test,y_pred)

from sklearn.preprocessing import LabelEncoder
encode_ferti = LabelEncoder()
df['Fertilizer Name']=encode_ferti.fit_transform(df['Fertilizer Name'])
#creating the dataframe
Fertilizer = pd.DataFrame(zip(encode_ferti.classes_,encode_ferti.transform(encode_ferti.classes_)),columns=['original','Encoded'])
Fertilizer = Fertilizer.set_index('original')
Fertilizer

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(df.drop(columns=['Fertilizer Name']),df['Fertilizer Name'],test_size=0.2,random_state=1)
print('Shape of Splitting :')
print('x_train = {}, y_train = {}, x_test = {}, y_test = {}'.format(x_train.shape,y_train.shape,x_test.shape,y_test.shape))

x_train.info()

## Evaluation Metrics
rand = RandomForestClassifier(random_state = 42)
rand.fit(x_train,y_train)

pred_rand = rand.predict(x_test)

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report

params = {
    'n_estimators':[300,400,500],
    'max_depth':[5,6,7],
    'min_samples_split':[2,5,8]
}
grid_rand = GridSearchCV(rand,params,cv=3,verbose=3,n_jobs=-1)
grid_rand.fit(x_train,y_train)
pred_rand = grid_rand.predict(x_test)
print(classification_report(y_test,pred_rand))
print('Best score : ',grid_rand.best_score_)
print('Best params : ',grid_rand.best_params_)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Compute evaluation metrics
accuracy_rand = accuracy_score(y_test, pred_rand)
precision_rand = precision_score(y_test, pred_rand, average='weighted')
recall_rand = recall_score(y_test, pred_rand, average='weighted')
f1_rand = f1_score(y_test, pred_rand, average='weighted')

# Print evaluation metrics
print(f"Random Forest Classifier Metrics with GridSearchCV:")
print(f"Accuracy: {accuracy_rand}")
print(f"Precision: {precision_rand}")
print(f"Recall: {recall_rand}")
print(f"F1-score: {f1_rand}")

# Compute the confusion matrix
conf_matrix_rand = confusion_matrix(y_test, pred_rand)

# Display the confusion matrix using seaborn heatmap
plt.figure(figsize=(10, 12))
sns.heatmap(conf_matrix_rand, annot=True, cmap='Blues', fmt='g', cbar=False)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix - Random Forest Classifier with GridSearchCV')
plt.show()

"""## Utilizing Neural Network with MLPClassifier in Fertilizer Recommendation Project
"""
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
fertilizer_data =  pd.read_csv("Fertilizer.csv")
# Preprocessing
X = fertilizer_data.drop(columns=['Fertilizer Name'])
y = fertilizer_data['Fertilizer Name']
# Encoding the target variable
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
# Scaling the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)
# Neural Network Model
mlp = MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu', solver='adam', max_iter=100, random_state=42)

# Training the model
mlp.fit(X_train, y_train)
# Predictions
y_pred = mlp.predict(X_test)

# Compute evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

# Print evaluation metrics
print("Neural Network Model Metrics:")
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1-score: {f1}")

# Compute the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Display the confusion matrix using seaborn heatmap
plt.figure(figsize=(10, 12))
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='g', cbar=False)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix - Neural Network')
plt.show()

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, log_loss
import tracemalloc

# Load the dataset
df = pd.read_csv('Fertilizer.csv')

# Encode the target variable using LabelEncoder
encoder = LabelEncoder()
df['Fertilizer Name'] = encoder.fit_transform(df['Fertilizer Name'])

# Separate features and target variable after encoding
X = df[['Nitrogen', 'Potassium', 'Phosphorous']]
y = df['Fertilizer Name']

# Split the data into training and testing sets with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Function to train and evaluate a model
def train_and_evaluate_model(model, X_train_scaled, y_train, X_test_scaled, y_test, model_name):
    tracemalloc.start()  # Start tracing the memory allocation

    start_time = time.time()
    model.fit(X_train_scaled, y_train)
    training_time = time.time() - start_time

    # Memory usage
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()  # Stop tracing the memory allocation

    # Predictions and probabilities on test set
    y_pred = model.predict(X_test_scaled)
    y_probs = model.predict_proba(X_test_scaled) if hasattr(model, "predict_proba") else None

    # Log Loss
    model_log_loss = log_loss(y_test, y_probs) if y_probs is not None else "Not Applicable"

    print(f"{model_name} Training Time: {training_time:.4f} seconds")
    print(f"{model_name} Memory Usage: {current / 10**6:.4f} MB; Peak: {peak / 10**6:.4f} MB")
    print(f"{model_name} Log Loss: {model_log_loss}\n")

    # Detailed classification report
    print(classification_report(y_test, y_pred, labels=np.unique(y_train), target_names=encoder.classes_))

# Initialize models
rf_model = RandomForestClassifier(random_state=42)
nn_model = MLPClassifier(random_state=42)

# Train and evaluate Random Forest Classifier
print("Random Forest Classifier:")
train_and_evaluate_model(rf_model, X_train_scaled, y_train, X_test_scaled, y_test, 'Random Forest Classifier')

# Train and evaluate Neural Network
print("\nNeural Network:")
train_and_evaluate_model(nn_model, X_train_scaled, y_train, X_test_scaled, y_test, 'Neural Network')

# Import libraries
import pandas as pd
import matplotlib.pyplot as plt

# Create a DataFrame with the data
data = {
    'Model': ['Random Forest', 'Neural Network'],
    'Training Time (s)': [0.8821, 0.2411],
    'Memory Usage (MB)': [0.2875, 0.1895],
    'Peak Memory Usage (MB)': [0.3337, 0.4129],
    'Log Loss': [0.031587, 0.168100]
}

df = pd.DataFrame(data)
# Create a bar chart for training time
plt.figure(figsize=(10, 6))
plt.bar(df['Model'], df['Training Time (s)'], color=['blue', 'green'])
plt.xlabel('Model')
plt.ylabel('Training Time (s)')
plt.title('Training Time Comparison')
# plt.yscale('log')  # Logarithmic scale for y-axis is not necessary here
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

import matplotlib.pyplot as plt
import pandas as pd

# Data
models = ['Random forest', 'Neural Network']
precision = [0.9958, 1.0]
recall = [0.9955, 0.95]
f1_score = [0.9954, 0.96]
accuracy = [0.9876, 0.95]

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
ax.set_ylim(0.9, 1.0)  # Adjusted y-axis limits

plt.show()

"""
## Analysis of Model Performance
**The superior performance of Random Forest in this context can be attributed to several factors:**
1. **Data Characteristics:** The dataset might have features and relationships well-captured by the decision trees in Random Forest.
2. **Overfitting Avoidance:** Random Forest naturally avoids overfitting better than Neural Networks, especially if the dataset isn't massive.
3. **Complexity Balance:** Random Forest strikes a balance between handling complex relationships and not becoming too complex itself, unlike Neural Networks which can become overly complex.<br>

The Neural Network's slightly lower scores might be due to overfitting, the need for more data, or complexity that isn't necessary for this specific dataset.

# CONCLUSION
- In the fertilizer recommendation project, the choice between Random Forest and Neural Network should consider the dataset's nature and the project's specific needs. 
Random Forest emerges as a more balanced choice, offering robust performance with less risk of overfitting and a good handle on complex data relationships. 
Its ability to provide high accuracy with less computational complexity makes it suitable for a variety of scenarios.

- Neural Networks, while powerful, may require more data and careful tuning to achieve their full potential. 
They are more suited to scenarios where the complexity and size of the dataset justify their use.
"""

## Export model to Pickle file
import pickle
pickle_out = open('random_forest_model.pkl', 'wb')
pickle.dump(grid_rand,pickle_out)
pickle_out.close()