# Sales-Analysis-Forecasting-using-Machine-Learning
This project focuses on analyzing and forecasting sales data across multiple restaurants. It combines data preprocessing, exploratory data analysis (EDA), and machine learning models to uncover insights and predict future sales trends.


---

### 🎯 Objectives

* Perform data cleaning and integration from multiple datasets
* Analyze sales patterns across time, stores, and products
* Build machine learning models to forecast future sales
* Compare model performance using evaluation metrics

---

## 🔍 Methodology

### 1. 🧹 Data Preparation

* Imported and examined datasets for structure and outliers
* Merged multiple datasets into a unified dataset containing:

  * Date, Item ID, Price, Item Count
  * Item Name, Calories (kcal)
  * Store ID and Store Name

---

### 2. 📊 Exploratory Data Analysis (EDA)

* Analyzed **date-wise sales trends**
* Studied **weekly and monthly sales patterns**
* Examined **quarterly sales distribution**
* Compared performance across restaurants
* Identified:

  * Most popular items overall and per store
  * Highest revenue-generating stores
  * Relationship between sales volume and revenue
  * Most expensive items and their calorie values

---

### 3. 🤖 Forecasting using Machine Learning

* Engineered time-based features:
  * Day, Month, Year, Quarter, Day of Week
    
* Built and compared models:
  * Linear Regression
  * Random Forest
  * XGBoost
    
* Used last 6 months as test data
* Evaluated models using **RMSE (Root Mean Square Error)**
* Selected best-performing model for **next-year sales forecasting**

---

## 📊 Results & Insights

* Identified key sales trends across time and locations
* Discovered top-performing stores and products
* Built predictive models for accurate sales forecasting
* Improved decision-making for inventory and marketing strategies

---

## 🛠️ Tech Stack

* Python
* Pandas, NumPy
* Matplotlib, Seaborn
* Scikit-learn
* XGBoost

---

