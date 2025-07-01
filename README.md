# LEVERAGING-MACHINE-LEARNING-ALGORITHMS-TO-PREDICT-FLIGHT-PRICE-FLUCTUATIONS

# Leveraging Machine Learning to Predict Flight Price Fluctuations
This project uses machine learning algorithms to analyze and predict flight ticket prices based on historical data and relevant flight features. By training predictive models on features such as airline, source, destination, date of journey, duration, and number of stops, this system provides users with insights into expected price fluctuations — helping them make smarter travel decisions.


# Features

Data Preprocessing: Cleaned and transformed real-world flight data for ML use.

Feature Engineering: Extracted useful features like journey day/month, total travel duration, and time of booking.

Modeling Techniques: Implemented and compared multiple models such as:

Linear Regression

Random Forest

Decision Tree

XGBoost

Evaluation Metrics: Assessed models using RMSE, MAE, and R² scores.

Interactive UI: Built using Tkinter, allowing users to input flight details and get predicted prices instantly.

Visualization: Data distribution, feature importance, and prediction vs actual price graphs for interpretability.

# Tech Stack
Languages: Python

Libraries: Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, XGBoost

UI Framework: Tkinter

# Model Performance
Random Forest achieved the highest accuracy with an R² score ~0.85

Feature importance revealed duration, airline, and journey date as key predictors

# Future Enhancements
Incorporate real-time flight data using APIs

Deploy the model as a web application

Add hyperparameter tuning for optimized performance

# Dataset Source
The dataset used was publicly available on Kaggle and includes details of domestic flights in India.
