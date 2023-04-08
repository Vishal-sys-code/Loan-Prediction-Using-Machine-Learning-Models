#!/usr/bin/env python
# coding: utf-8

# <h1 style = "color: blue">Loan Prediction Using KNN, Decision Tree, SVM, Logistic Regression, Naive Bayes, Random Forest</h1>

# # Reading the Dataset 

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# # Exploratory Data Analytics (EDA)

# ## Importing the Libraries

# In[2]:


import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# ## Importing the train dataset

# In[3]:


df = pd.read_csv('/kaggle/input/loan-prediction-problem-dataset/train_u6lujuX_CVtuZ9i.csv')
df.head(10)


# ## Importing the test dataset

# In[4]:


test = pd.read_csv('/kaggle/input/loan-prediction-problem-dataset/test_Y3wMUE5_7gLdaTN.csv')
test.head(10)


# ## Gathering information of dataset

# In[5]:


df.info()


# ## Computing the mode of each categorical variable

# In[6]:


print('Gender Mode: ', df['Gender'].mode())
print('Married  Mode: ', df['Married'].mode())
print('Self_Employed Mode: ', df['Self_Employed'].mode())
print('Credit_History Mode: ', df['Credit_History'].mode())


# ## Creating a bar plot using the Seaborn library

# In[7]:


sns.barplot(x = df['Loan_Amount_Term'], y = df['LoanAmount'])


# ## Boolean Indexing

# In[8]:


df[['Loan_Amount_Term', 'LoanAmount']][df['Loan_Amount_Term'].isnull()]


# ## Quick overview of the distribution of values

# In[9]:


df['Dependents'].value_counts()


# ## Value Substitution

# In[10]:


df['Dependents'].replace('3+',3,inplace = True)
df['Dependents'].value_counts()


# ## Boolean Indexing

# In[11]:


df[['Dependents', 'Married']][df['Dependents'].isnull()]


# ## Missing value imputation

# In[12]:


df['Gender'].fillna('Male', inplace = True)
df['Married'].fillna('Yes', inplace = True)
df['Self_Employed'].fillna('No', inplace = True)
df['Credit_History'].fillna('1.0', inplace = True)
df['LoanAmount'].fillna((df['LoanAmount'].mean()), inplace = True)
df['Loan_Amount_Term'].fillna('84', inplace = True)
df['Dependents'].fillna(0, inplace = True)


# ## DataType conversion 

# In[13]:


df['Dependents'] = df['Dependents'].astype('int')
df['Dependents'].dtype


# ## Missing Value Detection

# In[14]:


df.isnull().sum()


# ## Column Removal or Column Dropping

# In[15]:


df.drop('Loan_ID', axis = 1, inplace = True)


# ## Computing the number of distinct values in each column

# In[16]:


df.nunique()


# ## Summary Statistics Computation 

# In[17]:


df.describe()


# ## Box Plot Visualization

# In[18]:


plt.figure(figsize = (10,4))
sns.catplot(data = df, kind = 'box')
plt.xticks(rotation = 90)
plt.grid()
plt.show()


# ## Frequency Plot

# In[19]:


fig, axs = plt.subplots(figsize = (25,6), ncols = 6, nrows = 2)
sns.countplot(x = df['Loan_Status'], ax = axs[0,0])
sns.countplot(x = df['Gender'], hue = df['Loan_Status'], ax = axs[0,1])
sns.countplot(x = df['Married'], hue = df['Loan_Status'], ax = axs[0,2])
sns.countplot(x = df['Dependents'], hue = df['Loan_Status'], ax = axs[0,3])
sns.countplot(x = df['Education'], hue = df['Loan_Status'], ax = axs[0,4])
sns.countplot(x = df['Self_Employed'], hue = df['Loan_Status'], ax = axs[0,5])

sns.countplot(x = df['Credit_History'], hue = df['Loan_Status'], ax = axs[1,0])
sns.countplot(x = df['Property_Area'], hue = df['Loan_Status'], ax = axs[1,1])
sns.countplot(x = df['Gender'], hue = df['Dependents'], ax = axs[1,2])
sns.countplot(x = df['Loan_Amount_Term'], hue = df['Loan_Status'], ax = axs[1,3])
sns.countplot(x = df['Married'], hue = df['Dependents'], ax = axs[1,4])
sns.countplot(x = df['Education'], hue = df['Self_Employed'], ax = axs[1,5])


# ## Data Visualization 

# In[20]:


fig, axs = plt.subplots(figsize = (20,3), ncols = 5)
sns.countplot(x = df['ApplicantIncome'], hue = df['Loan_Status'], fill = True ,ax = axs[0])
sns.countplot(x = df['CoapplicantIncome'], hue = df['Loan_Status'], fill = True ,ax = axs[1])
sns.countplot(x = df['LoanAmount'], hue = df['Loan_Status'], fill = True ,ax = axs[2])
sns.countplot(x = df['Loan_Amount_Term'], hue = df['Loan_Status'], fill = True ,ax = axs[3])
sns.countplot(x = df['ApplicantIncome'], hue = df['Gender'], fill = True ,ax = axs[4])
plt.show()


# ## Pair Plot

# In[21]:


sns.pairplot(df, hue = 'Loan_Status')


# ## Identifying which columns in the DataFrame contain non-numeric data

# In[22]:


obj_col = df.select_dtypes('object').columns
obj_col


# ## Converting Object to Numeric data type

# In[23]:


from sklearn.preprocessing import OrdinalEncoder
oe = OrdinalEncoder()
df[obj_col] = df[obj_col].astype(str)
df[obj_col] = oe.fit_transform(df[obj_col])
df.head(3)


# ## Changing Variable 💀

# In[24]:


data = df


# ## Statistical Summary of the Dataset

# In[25]:


sns.catplot(data = df, kind = 'boxen')
plt.xticks(rotation = 90)
plt.show()


# ## Summary Statistics Computation 

# In[26]:


df.describe()


# ## Creating HeatMap

# In[27]:


plt.figure(figsize = (20,5))
sns.heatmap(df.corr(), annot = True)
plt.show()


# ## Standardization

# In[28]:


from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
df.iloc[:,:-1] = ss.fit_transform(df.iloc[:,:-1])
df.head()


# ## Splitting the Data

# In[29]:


x = df.iloc[:,:-1]
y = df.iloc[:,-1]
x.head()


# # Train-Test split on the dataset

# In[30]:


from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x,y,random_state = 4, test_size = 0.25, stratify = y)


# # Making Machine Learning Model that fits and evaluates on training and testing data

# In[31]:


def mymodel(model):
    model.fit(xtrain,ytrain)
    ypred = model.predict(xtest)
    train_accuracy = model.score(xtrain,ytrain)
    test_accuracy = model.score(xtest, ytest)
    print(str(model)[:-2], 'Accuracy')
    print('Accuracy: ', accuracy_score(ytest,ypred), "\nClassification Report: \n", classification_report(ytest, ypred), '\nConfusion Matrix: \n', confusion_matrix(ytest, ypred))
    print(f'Training Accuracy: {train_accuracy}\nTesting Accuracy: {test_accuracy}')
    print()
    print()
    return model
    


# # Importing Libraries foe Accuracy score, Confusion Matrix and Classification Report

# In[32]:


from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# # Using KNN (K-Nearest Neighbors)

# In[33]:


from sklearn.neighbors import KNeighborsClassifier
knn = mymodel(KNeighborsClassifier())


# # Using SVM Model

# In[34]:


from sklearn.svm import SVC
svc = mymodel(SVC())


# # Using Decision Tree Model

# In[35]:


from sklearn.tree import DecisionTreeClassifier
dt= mymodel(DecisionTreeClassifier())


# # Using Logistic Regression Model

# In[36]:


from sklearn.linear_model import LogisticRegression
lr = mymodel(LogisticRegression())


# # Using Gaussian Naive Bayes Model

# In[37]:


from sklearn.naive_bayes import GaussianNB
gnb = mymodel(GaussianNB())


# # Using Random Forest Model

# In[38]:


from sklearn.ensemble import RandomForestClassifier
rfc = mymodel(RandomForestClassifier(n_estimators = 80, max_depth = 10, min_samples_leaf = 12))

