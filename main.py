import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

from xgboost import XGBRegressor

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_percentage_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense


def main():
                                  #### Data Science ####
                                  # Part 1 #
   #import the csv files
   items = pd.read_csv('/Users/ujjwalmanikyanath/Desktop/Simplilearn_Data Science course 2025/Module_7_Capstone project/Datasets/Capstone 3/items.csv')
   resturants = pd.read_csv('/Users/ujjwalmanikyanath/Desktop/Simplilearn_Data Science course 2025/Module_7_Capstone project/Datasets/Capstone 3/resturants.csv')
   sales = pd.read_csv('/Users/ujjwalmanikyanath/Desktop/Simplilearn_Data Science course 2025/Module_7_Capstone project/Datasets/Capstone 3/sales.csv')
   
   # print the files
   print('\n\n items.csv:\n', items)
   print('\n\n resturants.csv: \n',resturants)
   print('\n\n sales.csv: \n',sales)
   
   # print the shape of the files
   print("Shape of items.csv:", items.shape)
   print("Shape of resturants.csv:", resturants.shape)
   print("Shape of sales.csv:", sales.shape)

    # outlier calculation for items.csv
   columns_to_select = ['kcal', 'cost'] 

   for col in columns_to_select:
       Q1 = items[col].quantile(0.25)
       Q3 = items[col].quantile(0.75)
       IQR = Q3 - Q1
       
       lower_bound = Q1 - 1.5 * IQR
       upper_bound = Q3 + 1.5 * IQR
       
       outliers = items[(items[col] < lower_bound) | (items[col] > upper_bound)]
       print(f"\n\nOutliers of items.csv[{col}]:\n", outliers)
  
  
    # list the columns of sales.csv
   columns = sales.columns.tolist()
   print('\n\n colums of sales.csv:', columns)

   # outlier calculation for sales.csv
   columns_to_select = ['price', 'item_count'] 

   for col in columns_to_select:
       Q1 = sales[col].quantile(0.25)
       Q3 = sales[col].quantile(0.75)
       IQR = Q3 - Q1
       
       lower_bound = Q1 - 1.5 * IQR
       upper_bound = Q3 + 1.5 * IQR
       
       outliers = sales[(sales[col] < lower_bound) | (sales[col] > upper_bound)]
       print(f"\n\nOutliers of sales.csv[{col}]:\n", outliers)

    
    # rename the id column of items.csv
   items.rename(columns={'id': 'item_id'}, inplace=True) 

    # Merge sales with items on item_id
   merged_data = pd.merge(sales, items, on='item_id', how='left')

    # rename the id column of resturants.csv
   resturants.rename(columns={'id': 'store_id'}, inplace=True) 

    # Merge sales with items on item_id
   merged_data = pd.merge(merged_data, resturants, on='store_id', how='left')

   # Display result
   print('\n\n',merged_data.head())

    # Save to new CSV
   merged_data.to_csv('/Users/ujjwalmanikyanath/Desktop/Simplilearn_Data Science course 2025/Module_7_Capstone project/Datasets/Capstone 3/merged_dataset.csv', index=False)

                                # Part 2 #
   # Step 1- Convert date column to datetime
   merged_data['date'] = pd.to_datetime(merged_data['date'])

    # Step 2- Create total sales column
   merged_data['total_sales'] = merged_data['price'] * merged_data['item_count']
    
    # Step 3- Calculate Date-wise Sales
   date_sales = merged_data.groupby('date')['total_sales'].sum().reset_index()

   print('\n\n',date_sales.head())

   # Step 4- Plot Sales Trend
   plt.figure()
   plt.plot(date_sales['date'], date_sales['total_sales'])
   plt.xlabel("Date")
   plt.ylabel("Total Sales")
   plt.title("Date-wise Sales Trend")
   plt.xticks(rotation=45)
   plt.show()


   # Find out how sales fluctuate across different days of the week

   # Step 1- Extract Day of Week
   merged_data['day_of_week'] = merged_data['date'].dt.day_name()

   # Step 2- Group by Day of Week
   weekday_sales = merged_data.groupby('day_of_week')['total_sales'].sum().reset_index()

   print('\n\nweekday_sales:',weekday_sales)

   # Step 3- Sort in Proper Week Order
   order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday',
         'Friday', 'Saturday', 'Sunday']

   weekday_sales['day_of_week'] = pd.Categorical(
    weekday_sales['day_of_week'],
    categories=order,
    ordered=True
    )

   weekday_sales = weekday_sales.sort_values('day_of_week')

   # Step 4- Plot Weekly Sales Pattern
   plt.figure()
   plt.plot(weekday_sales['day_of_week'], weekday_sales['total_sales'])
   plt.xlabel("Day of Week")
   plt.ylabel("Total Sales")
   plt.title("Sales Fluctuation Across Weekdays")
   plt.xticks(rotation=45)
   plt.show()


   # Look for any noticeable trends in the sales data for different months of the year
   # step 1- Extract Month
   merged_data['month'] = merged_data['date'].dt.month
   merged_data['month_name'] = merged_data['date'].dt.month_name()

   # step 2- Group by Month
   monthly_sales_named = merged_data.groupby('month_name')['total_sales'].sum().reset_index()

    # step- 3 Sort months correctly
   order = ['January', 'February', 'March', 'April', 'May', 'June',
         'July', 'August', 'September', 'October', 'November', 'December']

   monthly_sales_named['month_name'] = pd.Categorical(
        monthly_sales_named['month_name'],
        categories=order,
        ordered=True
    )

   monthly_sales_named = monthly_sales_named.sort_values('month_name')

   plt.figure()
   plt.plot(monthly_sales_named['month_name'], monthly_sales_named['total_sales'])
   plt.xticks(rotation=45)
   plt.xlabel("Month")
   plt.ylabel("Total Sales")
   plt.title("Monthly Sales Pattern")
   plt.show()

    # Examine the sales distribution across different quarters averaged over the years. Identify any noticeable patterns
   # step 1: Extract year and Quarter
   
   #merged_data['year'] = merged_data['date'].dt.year
   merged_data['quarter'] = merged_data['date'].dt.quarter

   # step 2: Calculate Quarterly Sales Per Year
   quarterly_sales = merged_data.groupby(['quarter'])['total_sales'].sum().reset_index()

   # step 3: Average Across Years
   quarterly_avg = quarterly_sales.groupby('quarter')['total_sales'].mean().reset_index()

   print('\n\n',quarterly_avg)

   # step 4: Plot Quarterly Distribution
   plt.figure()
   plt.plot(quarterly_avg['quarter'], quarterly_avg['total_sales'])
   plt.xlabel("Quarter")
   plt.ylabel("Average Sales")
   plt.title("Average Sales Distribution Across Quarters")
   plt.show()

    # Bar Chart
   plt.figure()
   plt.bar(quarterly_avg['quarter'], quarterly_avg['total_sales'])
   plt.xlabel("Quarter")
   plt.ylabel("Average Sales")
   plt.title("Average Quarterly Sales")
   plt.show()


   # Compare the performances of the different restaurants. Find out which restaurant had the most sales and look at the sales for each restaurant across different years, months, and days
   # step 1 - Restaurant With Most Sales
   restaurant_sales = merged_data.groupby('name_y')['total_sales'].sum().reset_index()

    # Sort descending
   restaurant_sales = restaurant_sales.sort_values('total_sales', ascending=False)

   print('\n\n',restaurant_sales)

   # Highest Performing Restaurant
   print("\n\n Top Restaurant:")
   print(restaurant_sales.iloc[0])


   # Plot Overall Restaurant Performance
   plt.figure()
   plt.bar(restaurant_sales['name_y'], restaurant_sales['total_sales'])
   plt.xticks(rotation=45)
   plt.xlabel("Restaurant")
   plt.ylabel("Total Sales")
   plt.title("Total Sales by Restaurant")
   plt.show()

   # step- 2: Sales Across Different Years
   merged_data['year'] = merged_data['date'].dt.year

   yearly_sales = merged_data.groupby(['name_y', 'year'])['total_sales'].sum().reset_index()

   print('\n\n yearly sales: ',yearly_sales)


   # Plot Example (One Restaurant at a Time)
   restaurant_name = restaurant_sales.iloc[0]['name_y']

   data = yearly_sales[yearly_sales['name_y'] == restaurant_name]

   plt.figure()
   plt.plot(data['year'], data['total_sales'])
   plt.xlabel("Year")
   plt.ylabel("Sales")
   plt.title(f"Yearly Sales - {restaurant_name}")
   plt.show()


   # step 3: Sales Across Months
   merged_data['month'] = merged_data['date'].dt.month

   monthly_sales = merged_data.groupby(['name_y', 'month'])['total_sales'].sum().reset_index()

   print('\n\n ',monthly_sales.head())


   # step 4: Sales Across Days of Week
   merged_data['day'] = merged_data['date'].dt.day_name()

   weekday_sales = merged_data.groupby(['name_y', 'day'])['total_sales'].sum().reset_index()

   print('\n\n',weekday_sales.head())

   order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday',
         'Friday', 'Saturday', 'Sunday']

   weekday_sales['day'] = pd.Categorical(
        weekday_sales['day'],
        categories=order,
        ordered=True
    )

   weekday_sales = weekday_sales.sort_values('day')


                        #### Machine Learning #####

                        # Part 1 #

    # Identify the most popular items overall and the stores where they are being sold. Also, findout the most popular item at each store
    
    # step 1- Most Popular Items Overall
   popular_items = merged_data.groupby('name_x')['item_count'].sum().reset_index()

    # step 2- Sort descending
   popular_items = popular_items.sort_values('item_count', ascending=False)

   print('\n\n most poular items:',popular_items.head())

   # step 3 - Stores Where These Items Are Sold
   item_store_sales = merged_data.groupby(['name_x','name_y'])['item_count'].sum().reset_index()

    # step 4- Sort to see top selling combinations
   item_store_sales = item_store_sales.sort_values('item_count', ascending=False)

   print('\n\n most poular items and store: ',item_store_sales.head(10))

   # step 5- Most Popular Item at Each Store
   store_item_sales = merged_data.groupby(['name_y','name_x'])['item_count'].sum().reset_index()

    # step 6 - Get top item for each store
   top_item_per_store = store_item_sales.loc[
     store_item_sales.groupby('name_y')['item_count'].idxmax()
    ]

   print('\n\n top_item_per_store:',top_item_per_store)

   # step 7- Visualize Most Popular Items

   top10_items = popular_items.head(10)

   plt.figure()
   plt.bar(top10_items['name_x'], top10_items['item_count'])
   plt.xticks(rotation=45)
   plt.xlabel("Item")
   plt.ylabel("Total Quantity Sold")
   plt.title("Top 10 Most Popular Items")
   plt.show()


   # Determine if the store with the highest sales volume is also making the most money per day

   # Step -1: Find Store With Highest Sales Volume
   volume_sales = merged_data.groupby('name_y')['item_count'].sum().reset_index()

   volume_sales = volume_sales.sort_values('item_count', ascending=False)

   print('\n\nvolume_sales:',volume_sales)

   # step - 2 : Top store by volume:
   print("Store with highest sales volume:")
   print(volume_sales.iloc[0])


   # step -3 : Calculate Revenue Per Day for Each Store
   daily_sales = merged_data.groupby(['name_y','date'])['total_sales'].sum().reset_index()

   # Now compute average revenue per day per store
   avg_daily_sales = daily_sales.groupby('name_y')['total_sales'].mean().reset_index()

   avg_daily_sales = avg_daily_sales.sort_values('total_sales', ascending=False)

   print('\n\navg_daily_sales: ',avg_daily_sales)

   # Top store by daily revenue:
   print("\nStore with highest average daily revenue:")
   print(avg_daily_sales.iloc[0])

   # step 4- Compare Results

   print("\nHighest volume store:", volume_sales.iloc[0]['name_y'])
   print("\nHighest revenue per day store:", avg_daily_sales.iloc[0]['name_y'])

   # step 5-  Visualization
 

   plt.figure()
   plt.bar(avg_daily_sales['name_y'], avg_daily_sales['total_sales'])
   plt.xticks(rotation=45)
   plt.xlabel("Store")
   plt.ylabel("Average Daily Revenue")
   plt.title("Average Daily Revenue by Store")
   plt.show()

   # Identify the most expensive item at each restaurant and find out its calorie count

   # step 1- Find Most Expensive Item at Each Store
   expensive_items = merged_data.loc[
    merged_data.groupby('name_y')['cost'].idxmax(),
      ['name_y', 'name_x', 'cost', 'kcal']
      ]

   print('\nexpensive_items',expensive_items)

                        # Part 2 #
   ## Forecasting using machine learning algorithms ##

   # step 1 - Feature Engineering
   date_sales['year'] = date_sales['date'].dt.year
   date_sales['month'] = date_sales['date'].dt.month
   date_sales['day'] = date_sales['date'].dt.day
   date_sales['day_of_week'] = date_sales['date'].dt.dayofweek
   date_sales['quarter'] = date_sales['date'].dt.quarter

   # step 2- Features list:
   features = ['year','month','day','day_of_week','quarter']
   X = date_sales[features]
   y = date_sales['total_sales']

   # step 3- Train/Test Split
   split_date = date_sales['date'].max() - pd.DateOffset(months=6)

   train = date_sales[date_sales['date'] < split_date]
   test = date_sales[date_sales['date'] >= split_date]

   X_train = train[features]
   y_train = train['total_sales']

   X_test = test[features]
   y_test = test['total_sales']


   # Step 4: Model 1 — Linear Regression

   lr = LinearRegression()

   lr.fit(X_train, y_train)

   lr_pred = lr.predict(X_test)

   lr_rmse = np.sqrt(mean_squared_error(y_test, lr_pred))

   print("\nLinear Regression RMSE:", lr_rmse)



   # Step 5: Model 2 — Random Forest

   rf = RandomForestRegressor(n_estimators=200, random_state=42)

   rf.fit(X_train, y_train)

   rf_pred = rf.predict(X_test)

   rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))

   print("\nRandom Forest RMSE:", rf_rmse)



   # Step 6: Model 3 — XGBoost
   xgb = XGBRegressor(n_estimators=200, learning_rate=0.05)

   xgb.fit(X_train, y_train)

   xgb_pred = xgb.predict(X_test)

   xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_pred))

   print("\nXGBoost RMSE:", xgb_rmse)


   # Step 7: Compare Model Performance
   print("\nModel Comparison")
   print("\nLinear Regression RMSE:", lr_rmse)
   print("\nRandom Forest RMSE:", rf_rmse)
   print("\nXGBoost RMSE:", xgb_rmse)


   # Step 8: Forecast Next Year
   future_dates = pd.date_range(
      start= date_sales['date'].max()+pd.Timedelta(days=1),
      periods=365
      )

   future = pd.DataFrame({'date':future_dates})

   future['year'] = future['date'].dt.year
   future['month'] = future['date'].dt.month
   future['day'] = future['date'].dt.day
   future['day_of_week'] = future['date'].dt.dayofweek
   future['quarter'] = future['date'].dt.quarter

   # Step 9: Predict using best model
   future['forecast_sales'] = xgb.predict(future[features])

   # Step 10: Plot Forecast
   plt.figure()

   plt.plot(date_sales['date'], date_sales['total_sales'])
   plt.plot(future['date'], future['forecast_sales'])

   plt.xlabel("Date")
   plt.ylabel("Sales")
   plt.title("Sales Forecast for Next Year")

   plt.show()


                  ### Deep Learning ###

                  # part 1 # 
   # Forecasting using deep learning algorithms:
     
   # step 1 - Sort by date:
   date_sales = date_sales.sort_values('date')

   # step 2- Scale the Data (LSTM models perform better with normalized data)
   scaler = MinMaxScaler()

   sales_scaled = scaler.fit_transform(date_sales[['total_sales']])

   # step 3- Create Time Series Sequences (Define a function to create input sequences)
   def create_sequences(data, seq_length):
    
        X = []
        y = []
    
        for i in range(len(data) - seq_length):
            X.append(data[i:i+seq_length])
            y.append(data[i+seq_length])
        
        return np.array(X), np.array(y)
   
   # Use 30 days of history to predict the next day

   seq_length = 30
   X, y = create_sequences(sales_scaled, seq_length)


   # step 4- Define Train and Test Series (Use last 12 months as test data)
   test_days = 365

   X_train = X[:-test_days]
   y_train = y[:-test_days]

   X_test = X[-test_days:]
   y_test = y[-test_days:]

   # step 5-  Generate Synthetic Data (Augmentation)
   noise = np.random.normal(0, 0.01, X_train.shape)

   X_train_augmented = X_train + noise

   # step 6- Build the LSTM Model

   model = Sequential()

   model.add(LSTM(50, activation='relu', input_shape=(seq_length,1)))
   model.add(Dense(1))

   model.compile(
       optimizer='adam',
      loss='mse'
      )
   
   # step 7- Train the Model

   model.fit(
      X_train_augmented,
      y_train,
      epochs=20,
      batch_size=32
      )
   
   # step 8- Make Predictions on Test Data

   predictions = model.predict(X_test)

   # Convert predictions back to original scale

   predictions = scaler.inverse_transform(predictions)

   y_test_actual = scaler.inverse_transform(y_test)

   # step 9- Calculate MAPE

   mape = mean_absolute_percentage_error(
      y_test_actual,
      predictions
      )

   print("\nMAPE:", mape)
   print("MAPE (%):", mape * 100)

   # step- 10: Plot Actual vs Predicted

   plt.figure()

   plt.plot(y_test_actual, label="Actual")
   plt.plot(predictions, label="Predicted")

   plt.legend()

   plt.title("LSTM Sales Prediction")

   plt.show()


   # step 11: Train Model on Entire Series (train using all data to forecast future values)

   model.fit(
       X,
       y,
       epochs=20,
       batch_size=32
      )
   
   # step - 12: Forecast Next 3 Months (Predict 90 days ahead)

   future_predictions = []

   last_sequence = sales_scaled[-seq_length:]

   current_seq = last_sequence.reshape(1, seq_length, 1)

   for i in range(90):
       

       pred = model.predict(current_seq)

       future_predictions.append(pred[0][0])

            # reshape prediction properly
       pred = pred.reshape(1,1,1)

       current_seq = np.append(
            current_seq[:,1:,:],
            pred,
            axis=1
            )
   
   # Convert back to original scale

   future_predictions = scaler.inverse_transform(
      np.array(future_predictions).reshape(-1,1)
      )
   
   # step 13: Plot Forecast

   plt.figure()

   plt.plot(date_sales['total_sales'], label="Historical")

   plt.plot(
      range(len(date_sales), len(date_sales)+90),
      future_predictions,
      label="Forecast"
      )

   plt.legend()

   plt.title("3-Month Sales Forecast")

   plt.show()

   
   
   
   
   # Develop another model using the entire series for training, and use it to forecast for the next three months

   # Build ELU LSTM Model

   model = Sequential()

   model.add(
       LSTM(
           50,
            activation='elu',
            input_shape=(seq_length,1)
         )
   )

   model.add(Dense(1))

   model.compile(
      optimizer='adam',
      loss='mse'
   )







if __name__ == "__main__":
    main()