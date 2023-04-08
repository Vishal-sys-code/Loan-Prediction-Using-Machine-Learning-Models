# Loan Prediction Using Machine Learning Models

## **Overview**

This project is aimed at predicting whether a loan application will be approved or not based on the applicant's information. We use six different machine learning algorithms to train our models and predict the outcome. The algorithms used are K-Nearest Neighbors (KNN), Decision Tree, Support Vector Machine (SVM), Logistic Regression, Naive Bayes, and Random Forest.

The dataset used for this project is the Loan Prediction Dataset, which contains various attributes such as applicant's income, loan amount, credit history, and so on.

## **Dependencies**

This project requires the following Python libraries to be installed:
* pandas
* numpy
* scikit-learn
* seaborn
* matplotlib

You can install these libraries by running ```pip install <library name>``` in your terminal or command prompt.

## **Files**

The following files are included in this project:

* ```loan_prediction.ipynb```: Jupyter Notebook containing the code for data exploration, preprocessing, feature engineering, and model building.
* ```loan_prediction.py```: Python file containing the same code as in the Jupyter Notebook, but in a script format.
* ```Loan_Prediction.csv```: Dataset file containing loan application data.

## **How to run**

To run the project, follow these steps:

* Clone or download the project repository.
* Install the required dependencies (mentioned above).
* Open the ```loan_prediction.ipynb``` file in Jupyter Notebook or ```loan_prediction.py``` file in any Python IDE or command prompt.
* Run the cells or script to execute the code.
* Once executed, the code will train the models and output the accuracy score for each algorithm.


## **Conclusion**

The project's objective is to predict whether a loan application will be approved or not using six different machine learning algorithms. We have explored the dataset, preprocessed the data, engineered features, and built models using the six algorithms. Based on the accuracy score, we can see that the Random Forest algorithm performs the best among all the models.
