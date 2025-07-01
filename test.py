import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.svm import SVR  #SVR regression
from sklearn.ensemble import RandomForestRegressor #random forest regression class
from sklearn.neural_network import MLPRegressor
from sklearn_extensions.extreme_learning_machines.elm import GenELMRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import BaggingRegressor
from sklearn_extensions.extreme_learning_machines.random_layer import RBFRandomLayer, MLPRandomLayer
from sklearn_extensions.extreme_learning_machines.elm import GenELMClassifier

dataset = pd.read_csv("Dataset/Airline_dataset.csv",nrows=200)
dataset.fillna(0, inplace = True)
dataset.drop(['Date_of_Journey'], axis = 1,inplace=True)
cols = ['Airline','Source','Destination','Route','Additional_Info','Dep_Time','Arrival_Time','Duration','Total_Stops']
le = LabelEncoder()

for i in range(len(cols)):
    dataset[cols[i]] = pd.Series(le.fit_transform(dataset[cols[i]].astype(str)))

dataset = dataset.values
Y = dataset[:,dataset.shape[1]-1]
X = dataset[:,0:dataset.shape[1]-1]
Y = Y.reshape(-1, 1)

sc = MinMaxScaler(feature_range = (0, 1))

X = sc.fit_transform(X)
Y = sc.fit_transform(Y)


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
srhl_tanh = MLPRandomLayer(n_hidden=100, activation_func='tanh')
rf_regression = DecisionTreeRegressor()#MLPRegressor()#SVR(kernel="linear")#BaggingRegressor()#LinearRegression()#GenELMRegressor()#RandomForestRegressor()
rf_regression.fit(X, Y.ravel())
predict = rf_regression.predict(X_test)
predict = predict.reshape(predict.shape[0],1)
predict = sc.inverse_transform(predict)
predict = predict.ravel()
labels = sc.inverse_transform(y_test)
labels = labels.ravel()
print("Original Stock Index : "+str(labels[i])+" Random Forest Predicted Stock Index : "+str(predict[i])+"\n")
rf_rae = mean_squared_error(labels,predict)
print("\nRandom Forest RAE Error : "+str(rf_rae)+"\n\n")
print("Random Forest Accuracy : "+str(rf_regression.score(X_train, y_train)))

plt.plot(labels, color = 'red', label = 'Test Stock Index')
plt.plot(predict, color = 'green', label = 'Random Forest Predicted Stock Index')
plt.title('Random Forest Stock Index Prediction')
plt.xlabel('Test Stock Index')
plt.ylabel('Random Forest Predicted Stock Index')
plt.legend()
plt.show()

