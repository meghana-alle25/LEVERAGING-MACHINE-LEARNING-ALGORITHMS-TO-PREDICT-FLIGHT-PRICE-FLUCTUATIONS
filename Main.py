from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
import matplotlib.pyplot as plt
from tkinter.filedialog import askopenfilename
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR  #SVR regression
from sklearn.ensemble import RandomForestRegressor #random forest regression class
from sklearn.neural_network import MLPRegressor
from sklearn_extensions.extreme_learning_machines.elm import GenELMRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import BaggingRegressor
from sklearn.preprocessing import LabelEncoder
import timeit

main = Tk()
main.title("Leveraging Machine Learning to predict Flight Price Fluctuations")
main.geometry("1300x1200")


global filename
global dataset
global X, Y, accuracy, le
sc = MinMaxScaler(feature_range = (0, 1))
global X_train, X_test, y_train, y_test

def uploadDataset():
    global filename
    global dataset
    text.delete('1.0', END)
    filename = askopenfilename(initialdir = "Dataset")
    tf1.insert(END,str(filename))
    text.insert(END,"Dataset Loaded\n\n")
    dataset = pd.read_csv(filename,nrows=200)
    text.insert(END,str(dataset.head()))
    
    
def preprocessDataset():
    global dataset
    global X, Y, sc, le
    global X_train, X_test, y_train, y_test
    text.delete('1.0', END)
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
    text.insert(END,str(X)+"\n\n")
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

def predictPrice(algorithm, predict, test, algorithm_name, execution_time):
    global X, Y
    predict = predict.reshape(predict.shape[0],1)
    predict = sc.inverse_transform(predict)
    predict = predict.ravel()
    labels = sc.inverse_transform(test)
    labels = labels.ravel()
    acc = algorithm.score(X,Y.ravel())
    accuracy.append(acc)
    text.insert(END,algorithm_name+" Accuracy : "+str(acc)+"\n\n")
    text.insert(END,algorithm_name+" Execution Time : "+str(execution_time)+" Seconds\n\n")
    for i in range(0,20):
        text.insert(END,"Original Price : "+str(labels[i])+algorithm_name+" Predicted Price : "+str(predict[i])+"\n")    
    plt.plot(labels, color = 'red', label = 'Original Test Prices')
    plt.plot(predict, color = 'green', label = algorithm_name+' Predicted Prices')
    plt.title(algorithm_name+' Predicted Prices')
    plt.xlabel('Original Test Prices')
    plt.ylabel(algorithm_name+' Airfare Predicted & Test Prices Comparison')
    plt.legend()
    plt.show()
    
def runMLP():
    global X, Y, accuracy
    global X_train, X_test, y_train, y_test              
    accuracy = []
    text.delete('1.0', END)
    start = timeit.default_timer()
    mlp = MLPRegressor()
    mlp.fit(X, Y.ravel())
    predict = mlp.predict(X_test)
    end = timeit.default_timer()
    predictPrice(mlp, predict, y_test, "MLP Algorithm", (end - start))

def runELM():
    text.delete('1.0', END)
    start = timeit.default_timer()
    elm_algorithm = GenELMRegressor()
    elm_algorithm.fit(X, Y.ravel())
    predict = elm_algorithm.predict(X_test)
    end = timeit.default_timer()
    predictPrice(elm_algorithm, predict, y_test, "ELM Algorithm", (end - start))    
  

def runRandomForest():
    text.delete('1.0', END)
    start = timeit.default_timer()
    rf_algorithm = RandomForestRegressor()
    rf_algorithm.fit(X, Y.ravel())
    predict = rf_algorithm.predict(X_test)
    end = timeit.default_timer()
    predictPrice(rf_algorithm, predict, y_test, "Random Forest Algorithm", (end - start)) 
    

def runRegressionTree():
    text.delete('1.0', END)
    start = timeit.default_timer()
    rt_algorithm = DecisionTreeRegressor()
    rt_algorithm.fit(X, Y.ravel())
    predict = rt_algorithm.predict(X_test)
    end = timeit.default_timer()
    predictPrice(rt_algorithm, predict, y_test, "Regression Tree Algorithm", (end - start)) 

def runBagging():
    text.delete('1.0', END)
    start = timeit.default_timer()
    br_algorithm = BaggingRegressor()
    br_algorithm.fit(X, Y.ravel())
    predict = br_algorithm.predict(X_test)
    end = timeit.default_timer()
    predictPrice(br_algorithm, predict, y_test, "Bagging Regressor Algorithm", (end - start)) 

def runPolySVM():
    text.delete('1.0', END)
    start = timeit.default_timer()
    poly_svm_algorithm = SVR(kernel="poly")
    poly_svm_algorithm.fit(X, Y.ravel())
    predict = poly_svm_algorithm.predict(X_test)
    end = timeit.default_timer()
    predictPrice(poly_svm_algorithm, predict, y_test, "Polynomial SVM Algorithm", (end - start))

def runLinearSVM():
    text.delete('1.0', END)
    start = timeit.default_timer()
    linear_svm_algorithm = SVR(kernel="linear")
    linear_svm_algorithm.fit(X, Y.ravel())
    predict = linear_svm_algorithm.predict(X_test)
    end = timeit.default_timer()
    predictPrice(linear_svm_algorithm, predict, y_test, "Linear SVM Algorithm", (end - start))
    
def runLinearRegression():
    text.delete('1.0', END)
    start = timeit.default_timer()
    linear_algorithm = LinearRegression()
    linear_algorithm.fit(X, Y.ravel())
    predict = linear_algorithm.predict(X_test)
    end = timeit.default_timer()
    predictPrice(linear_algorithm, predict, y_test, "LinearRegression Algorithm", (end - start))


def graph():
    height = accuracy
    bars = ('MLP','ELM','Random Forest','Regression Tree', 'Bagging Regressor', 'Poly SVM', 'Linear SVM', 'Linear Regression')
    y_pos = np.arange(len(bars))
    plt.bar(y_pos, height)
    plt.xticks(y_pos, bars)
    plt.title("All Algorithms Accuracy Comparison")
    plt.show()

font = ('times', 15, 'bold')
title = Label(main, text='Leveraging Machine Learning to predict Flight Price Fluctuations')
title.config(bg='HotPink4', fg='yellow2')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 13, 'bold')
ff = ('times', 12, 'bold')

l1 = Label(main, text='Dataset Location:')
l1.config(font=font1)
l1.place(x=50,y=100)

tf1 = Entry(main,width=60)
tf1.config(font=font1)
tf1.place(x=230,y=100)

uploadButton = Button(main, text="Upload Airfare Prices Dataset", command=uploadDataset, bg='#ffb3fe')
uploadButton.place(x=50,y=150)
uploadButton.config(font=font1)

preprocessButton = Button(main, text="Preprocess Dataset", command=preprocessDataset, bg='#ffb3fe')
preprocessButton.place(x=470,y=150)
preprocessButton.config(font=font1)

mlpButton = Button(main,text="Run MLP Algorithm", command=runMLP, bg='#ffb3fe')
mlpButton.place(x=790,y=150)
mlpButton.config(font=font1)

elmButton = Button(main,text="Run ELM Algorithm", command=runELM, bg='#ffb3fe')
elmButton.place(x=50,y=200)
elmButton.config(font=font1)

rfButton = Button(main,text="Train Random Forest Algorithm", command=runRandomForest, bg='#ffb3fe')
rfButton.place(x=470,y=200)
rfButton.config(font=font1)

rtButton = Button(main,text="Run Regression Tree Algorithm", command=runRegressionTree, bg='#ffb3fe')
rtButton.place(x=790,y=200)
rtButton.config(font=font1)

brButton = Button(main,text="Run BaggingRegressor Algorithm",command=runBagging, bg='#ffb3fe')
brButton.place(x=50,y=250)
brButton.config(font=font1)

polysvmButton = Button(main,text="Run Polynomial SVM", command=runPolySVM, bg='#ffb3fe')
polysvmButton.place(x=470,y=250)
polysvmButton.config(font=font1)

linearsvmButton = Button(main,text="Run Linear SVM", command=runLinearSVM, bg='#ffb3fe')
linearsvmButton.place(x=790,y=250)
linearsvmButton.config(font=font1)

linearButton = Button(main,text="Run LinearRegression Algorithm", command=runLinearRegression, bg='#ffb3fe')
linearButton.place(x=50,y=300)
linearButton.config(font=font1)

graphButton = Button(main,text="Accuracy Comparison Graph", command=graph, bg='#ffb3fe')
graphButton.place(x=790,y=300)
graphButton.config(font=font1)

font1 = ('times', 13, 'bold')
text=Text(main,height=20,width=130)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10,y=350)
text.config(font=font1)

main.config(bg='plum2')
main.mainloop()
