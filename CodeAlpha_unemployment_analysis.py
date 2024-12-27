import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA

#load the dataset
file_path = ["Unemployment in india.csv", "Unemployment_Rate_upto_11_2020.csv"]
for file in file_path:
  print(pd.read_csv(file).head())
  print(pd.read_csv(file).info())  # Overview of data types and non-null values
  print(pd.read_csv(file).describe())  # Summary statistics for numerical columns
  print(pd.read_csv(file).isnull().sum())  # Check for missing values


#Assigning the list index to two variables representing both dataset.
df_1 = pd.read_csv(file_path[0])
df_2 = pd.read_csv(file_path[1])



#Data cleaning

# Fill missing values (example, replace with mean or other strategies if needed)
#df_1.fillna(df_1.mean(), inplace=True)

# Drop missing values
df_1 = df_1.dropna(axis=0)
print(df_1.isnull().sum())
print(df_1.head())

#Analyzing the columns
print(df_1.columns)
print(df_2.columns)


# Remove leading and trailing spaces from all column names for both dataset.
df_1.columns = df_1.columns.str.strip()
df_2.columns = df_2.columns.str.strip()



# Convert 'Date' column to datetime
df_1['Date'] = pd.to_datetime(df_1['Date'], errors='coerce')
df_2['Date'] = pd.to_datetime(df_2['Date'], errors='coerce')


#visualize Unemployment Rate
plt.figure(figsize=(12, 6))

#Dataset 1. Check one at a time. plot or scattered plot.
#plt.plot(df_1['Date'], df_1['Estimated Unemployment Rate (%)'], marker='o', linestyle='-', color='blue')
#sns.scatterplot(data=df_1, x='Date', y='Estimated Unemployment Rate (%)', marker='o', color='blue')

#Dataset 2
#plt.plot(df_2['Date'], df_2['Estimated Unemployment Rate (%)'], marker='o', linestyle='-', color='blue')
#sns.scatterplot(data=df_2, x='Date', y='Estimated Unemployment Rate (%)', marker='o', color='blue')

#plt.title('Unemployment Rate Over Time')
#plt.xlabel('Date')
#plt.ylabel('Estimated Unemployment Rate (%)')
#plt.xticks(rotation=45)
#plt.tight_layout()
#plt.grid()
#plt.show()

#Distribution of unemployment rate
#sns.histplot(df_1['Estimated Unemployment Rate (%)'], kde=True, bins=20)
#plt.title('Distribution of Unemployment Rate')
#plt.xlabel('Estimated Unemployment Rate (%)')
#plt.ylabel('Frequency')
#plt.grid()
#plt.show()

#For second dataset
#sns.histplot(df_2['Estimated Unemployment Rate (%)'], kde=True, bins=20)
#plt.title('Distribution of Unemployment Rate')
#plt.xlabel('Estimated Unemployment Rate (%)')
#plt.ylabel('Frequency')
#plt.grid()
#plt.show()



#Visualize Unemployment Rate with Moving Average
#df_1['Unemployment_MA'] = df_1['Estimated Unemployment Rate (%)'].rolling(window=12).mean()
#df_2['Unemployment_MA'] = df_2['Estimated Unemployment Rate (%)'].rolling(window=12).mean()

# Assuming there's a column for time and unemployment rate
#plt.figure(figsize=(12, 6))

#sns.lineplot(data=df_1, x='Date', y='Estimated Unemployment Rate (%)', marker='o', label='Unemployment Rate')
#sns.lineplot(data=df_1, x='Date', y='Unemployment_MA', label='Moving Average')

#sns.lineplot(data=df_2, x='Date', y='Estimated Unemployment Rate (%)', marker='o', label='Unemployment Rate')
#sns.lineplot(data=df_2, x='Date', y='Unemployment_MA', label='Moving Average')

#plt.title('Unemployment Rate and 12-Month Moving Average Over Time')
#plt.legend()
#plt.show()


# Data Filtering
# Filter data for a specific region
#print(df_1['Region'].unique())
Region_name = ['Andhra Pradesh', 'Assam', 'Bihar', 'Chhattisgarh', 'Delhi', 'Goa', 'Gujarat', 'Haryana', 'Himachal Pradesh', 'Jammu & Kashmir', 'Jharkhand', 'Karnataka', 'Kerala', 'Madhya Pradesh', 'Maharashtra', 'Meghalaya', 'Odisha', 'Puducherry', 'Punjab', 'Rajasthan', 'Sikkim', 'Tamil Nadu', 'Telangana', 'Tripura', 'Uttar Pradesh', 'Uttarakhand', 'West Bengal', 'Chandigarh']
#print(len(Region_name))


# Run through the region names
#for region_of_interest in Region_name:
  #if df_1[df_1['Region'] == region_of_interest].shape[0] > 0:
    #filtered_data = df_1[df_1['Region'] == region_of_interest]
    #plt.figure(figsize=(12, 6))
    #plt.plot(filtered_data['Date'], filtered_data['Estimated Unemployment Rate (%)'], marker='o', linestyle='-', color='blue')
    #plt.title(f'Unemployment Rate Over Time for {region_of_interest}')
    #plt.xlabel('Date')
    #plt.ylabel('Estimated Unemployment Rate (%)')
    #plt.xticks(rotation=45)
    #plt.tight_layout()
    #plt.grid()
    #plt.show()




#Data filtering: Dataset 2.
#print(df_2['Region'].unique())
Region_name_2 = ['Andhra Pradesh', 'Assam', 'Bihar', 'Chhattisgarh', 'Delhi', 'Goa', 'Gujarat', 'Haryana', 'Himachal Pradesh', 'Jammu & Kashmir', 'Jharkhand', 'Karnataka', 'Kerala', 'Madhya Pradesh', 'Maharashtra', 'Meghalaya', 'Odisha', 'Puducherry', 'Punjab', 'Rajasthan', 'Sikkim', 'Tamil Nadu', 'Telangana', 'Tripura', 'Uttar Pradesh', 'Uttarakhand', 'West Bengal']
#print(len(Region_name_2))

# Run through the region names
#for region_of_interest in Region_name_2:
  #if df_2[df_2['Region'] == region_of_interest].shape[0] > 0:
    #filtered_data = df_2[df_2['Region'] == region_of_interest]
    #plt.figure(figsize=(12, 6))
    #plt.plot(filtered_data['Date'], filtered_data['Estimated Unemployment Rate (%)'], marker='o', linestyle='-', color='blue')
    #plt.title(f'Unemployment Rate Over Time for {region_of_interest}')
    #plt.xlabel('Date')
    #plt.ylabel('Estimated Unemployment Rate (%)')
    #plt.xticks(rotation=45)
    #plt.tight_layout()
    #plt.grid()
    #plt.show()


#Average unemployment by Region for both dataset
#state_avg = [df_1.groupby('Region')['Estimated Unemployment Rate (%)'].mean(), df_2.groupby('Region')['Estimated Unemployment Rate (%)'].mean()]
#print(state_avg)

# Plotting
#for state in state_avg:
  #state.plot(kind='bar', figsize=(12, 6), color='skyblue')
  #plt.title('Average Unemployment Rate by Region')
  #plt.xlabel('Region')
  #plt.ylabel('Estimated Unemployment Rate (%)')
  #plt.grid()
  #plt.show()



#Comparing areas
#Area_name = ['Rural', 'Urban']

# Run through the area names
#for area_of_interest in Area_name:
  #if df_1[df_1['Area'] == area_of_interest].shape[0] > 0:
    #filtered_data = df_1[df_1['Area'] == area_of_interest]
    #plt.figure(figsize=(12, 6))
    #sns.lineplot(data=filtered_data, x='Date', y='Estimated Unemployment Rate (%)', hue='Area')
    #plt.title(f'Unemployment Rate Over Time for {area_of_interest}')
    #plt.xlabel('Date')
    #plt.ylabel('Estimated Unemployment Rate (%)')
    #plt.xticks(rotation=45)
    #plt.legend(title='Area')
    #plt.tight_layout()
    #plt.show()

#Comparing Area: Dataset 2
#print(df_2['Region.1'].unique())
#Area_name_2 = ['South', 'Northeast', 'East', 'West', 'North']
#Run through the area names
#for area_of_interest in Area_name_2:
  #if df_2[df_2['Region.1'] == area_of_interest].shape[0] > 0:
    #filtered_data = df_2[df_2['Region.1'] == area_of_interest]
    #plt.figure(figsize=(12, 6))
    #sns.lineplot(data=filtered_data, x='Date', y='Estimated Unemployment Rate (%)', hue='Region.1')
    #plt.title(f'Unemployment Rate Over Time for {area_of_interest}')
    #plt.xlabel('Date')
    #plt.ylabel('Estimated Unemployment Rate (%)')
    #plt.xticks(rotation=45)
    #plt.legend(title='Area')
    #plt.tight_layout()
    #plt.show()

#Monthly Average dataset 1
#df_1['Month'] = df_1['Date'].dt.month
#monthly_avg = df_1.groupby('Month')['Estimated Unemployment Rate (%)'].mean().reset_index()

#plt.figure(figsize=(10, 6))
#sns.barplot(x='Month', y='Estimated Unemployment Rate (%)', data=monthly_avg)
#plt.title('Average Unemployment Rate by Month')
#plt.xlabel('Month')
#plt.ylabel('Estimated Unemployment Rate (%)')
#plt.show()

#Monthly Average dataset 2
#df_2['Month'] = df_2['Date'].dt.month
#monthly_avg_2 = df_2.groupby('Month')['Estimated Unemployment Rate (%)'].mean().reset_index()

#plt.figure(figsize=(10, 6))
#sns.barplot(x='Month', y='Estimated Unemployment Rate (%)', data=monthly_avg_2)
#plt.title('Average Unemployment Rate by Month')
#plt.xlabel('Month')
#plt.ylabel('Estimated Unemployment Rate (%)')
#plt.show()


#Correlation Analysis
#numerical_df = df_1.select_dtypes(include=['int64', 'float64'])
#correlation_matrix = numerical_df.corr()
# Visualize as a heatmap
#sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
#plt.title("Correlation Heatmap")
#plt.show()


#dataset 2
#correlation_matrix = df_2[['Estimated Unemployment Rate (%)', 'Estimated Employed', 'Estimated Labour Participation Rate (%)']].corr()
#print(correlation_matrix)
# Visualize as a heatmap
#sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
#plt.title("Correlation Heatmap")
#plt.show()





#Using a model to predict unemployment rate

# Set 'Date' as the index (important for time series)
# Convert 'Date' to datetime
df_1['Date'] = pd.to_datetime(df_1['Date'], format='%d-%m-%Y')
df_1.set_index('Date', inplace=True)

# Extract the target variable (Unemployment Rate)
unemployment_rate = df_1['Estimated Unemployment Rate (%)']
#print(unemployment_rate.index)

# Sort the index to make it monotonic
unemployment_rate = unemployment_rate.sort_index()

# Convert the index to a proper datetime index with monthly frequency
unemployment_rate.index = pd.to_datetime(unemployment_rate.index).to_period('M').to_timestamp()


# Generate datetime index for the forecast
last_date = unemployment_rate.index[-1]  # Last date in the actual data


# Perform ADF test
#result = adfuller(unemployment_rate)
#print("ADF Statistic:", result[0])
#print("p-value:", result[1])
#if result[1] <= 0.05:
    #print("The series is stationary.")
#else:
    #print("The series is not stationary.")


#if the series is not stationary, we can use differencing
# Differencing to make data stationary (if needed)
#diff_data = unemployment_rate.diff().dropna()

# Plot the differenced data
#plt.figure(figsize=(10, 6))
#plt.plot(diff_data, label='Differenced Unemployment Rate')
#plt.title('Stationary Unemployment Rate')
#plt.xlabel('Date')
#plt.ylabel('Differenced Rate')
#plt.legend()
#plt.show()

# Re-check stationarity with ADF
#result = adfuller(diff_data)
#print("ADF Statistic (after differencing):", result[0])
#print("p-value (after differencing):", result[1])

# Fit ARIMA model
model = ARIMA(unemployment_rate, order=(1, 1, 1))  # (p, d, q)
arima_result = model.fit()

# Print model summary
#print(arima_result.summary())

# Forecast the next 12 months
forecast = arima_result.forecast(steps=12)
#print("Forecasted Unemployment Rates:\n", forecast)

forecast_dates = pd.date_range(start=last_date + pd.offsets.MonthEnd(), 
                               periods=len(forecast), 
                               freq='ME')

# Create a forecast series
forecast_series = pd.Series(forecast, index=forecast_dates)

# Plot the actual vs forecasted values
#plt.figure(figsize=(10, 6))
#plt.plot(unemployment_rate, label='Actual Data')
#plt.plot(forecast, label='Forecast', color='red')
#plt.title('Unemployment Rate Forecast')
#plt.xlabel('Date')
#plt.ylabel('Unemployment Rate (%)')
#plt.legend()
#plt.grid()
#plt.show()