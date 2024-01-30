# Bike Sharing Utilization Prediction

## Overview

This project aims to predict the hourly utilization of rental bikes using the Bike Sharing Dataset. The dataset spans a two-year period, providing information on various attributes such as date, season, year, month, hour, weather conditions, temperature, humidity, wind speed, and the count of total rental bikes.

## Exploratory Data Analysis (EDA)

As a preliminary step, an extensive exploratory data analysis was conducted on the dataset. The analysis includes visualizations of the relationships between rental bike utilization (`cnt`) and both numerical and categorical variables. Key observations include temperature and wind speed impacting utilization, lower utilization in winter, higher utilization during working weekdays, and peak hours.

## Data Preprocessing

The dataset, containing 17,379 entries, has no missing values. Categorical variables are encoded as integers, and numerical variables are normalized. Correlation analysis revealed a high correlation between 'temp' and 'atemp,' leading to the exclusion of 'temp' from the considered features.

## Predictive Modeling

To predict bike utilization, regression models were trained and evaluated using cross-validation. The following models were considered:
- Linear Regression
- Lasso Regression
- Ridge Regression
- Support Vector Regression
- XGBoost Regression

The models were evaluated using the Average Mean Absolute Deviation (MAD) of cross-validated predictions. XGBoost Regression demonstrated the highest accuracy, with a MAD of 40.95, making it the chosen model for this business case.

## Unit Testing

To ensure the model's reliability, a simple unit test was implemented. The test checks that the MAD of the model is consistently lower than the MAD of the dataset with its mean.

## Usage

In this repo there is a Jupyter notebook with the exploratory data analysis, a python script for training and making predictions with the ML model and a project report.

Feel free to explore, modify, and contribute to the project. Your feedback and contributions are highly appreciated!
