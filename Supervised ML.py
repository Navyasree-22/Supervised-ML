#importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#load the data into the program
url='http://bit.ly/w-data'
data_load = pd.read_csv(url)
print("Sucessfully imported into console")
#use head function
data_load.head(6)
#To know number of rows and columns
data_load.shape
#To see summary statistics
data_load.describe()
#To find if any null value is present
data_load.isnull()
#view the data
data_load.dtypes
#enter distribution scores &plot them according to the requirement
data_load.plot(x='Hours', y='Scores', style='o')
plt.title('Hours vs Percentage')
plt.xlabel('The number of Hours studied')
plt.ylabel('The Percentage of the student')
plt.show()
#process dividing data into attributes and labels
x=data_load.iloc[:, :-1].values
y=data_load.iloc[:, 1].values
#splitting of data into the training & test sets
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test= train_test_split(x, y, test_size=0.2, random_state=0)
#training the algorithm i.e  using Linear Regression
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train, y_train)
print("Training...Completed!:)")
#implementing the plotting data using previous trained test data
line = regressor.coef_*x+regressor.intercept_
plt.scatter(x,y)
plt.plot(x, line);
plt.show()
#predicting the scores for the model
print(x_test)
y_pred = regressor.predict(x_test)
#comparing actual vs predicted model
data_load = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
data_load
#test our model with sample testing hours
hours= [[9.25]] #Instead of 9.25 hours use your data i.e hours
own_pred = regressor.predict(hours)
print("Number of hours={}".format(hours))
print("Prediction Score={}".format(own_pred[0]))
