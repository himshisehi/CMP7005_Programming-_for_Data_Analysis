#!/usr/bin/env python
# coding: utf-8

# In[1]:


## Import the necessary Libraries to run the code

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn import preprocessing
import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')


# Selecting Wanshouxigong as Urban Site, Changping as Suburban Site, Huairou as Rural Site and Guanyuan and Aotizhongxin Industrial site

# ***Fundamental Data Understanding***

# In[2]:


## loading all four datasets here

urban_df = pd.read_csv("PRSA_Data_Wanshouxigong_20130301-20170228.csv")
suburban_df = pd.read_csv("PRSA_Data_Changping_20130301-20170228.csv")
rural_df = pd.read_csv("PRSA_Data_Huairou_20130301-20170228.csv")
industrial_df = pd.read_csv("PRSA_Data_Aotizhongxin_20130301-20170228.csv")


# In[3]:


## Adding the Category columns gives context to each record for later analysis

urban_df['Category'] = 'Urban'
suburban_df['Category'] = 'Suburban'
rural_df['Category'] = 'Rural'
industrial_df['Category'] = 'Industrial'


# In[4]:


### Read and combine all CSV files

air_quality_df = pd.concat([urban_df, suburban_df, rural_df, industrial_df], ignore_index=True)


# In[5]:


# Save the merged dataset to a CSV file

air_quality_df.to_csv("air_quality.csv", index=False)


# In[6]:


# Load the CSV file into a pandas DataFrame

df = pd.read_csv("air_quality.csv")
df.head()


# In[7]:


# Combine Year, Month, Day, and Hour into a new 'Datetime' column

air_quality_df['Datetime'] = pd.to_datetime(air_quality_df[['year', 'month', 'day', 'hour']].astype(str).agg('-'.join, axis=1),format='%Y-%m-%d-%H')


# In[8]:


# Check the first few rows to confirm the combined Datetime

print(air_quality_df[['year', 'month', 'day', 'hour', 'Datetime']].head())


# In[9]:


# Displaying basic information about the dataset

print(air_quality_df.info())


# In[10]:


# Get summary statistics of numerical columns

print(air_quality_df.describe())


# In[11]:


# Check for missing values in each column

print(air_quality_df.isnull().sum())


# In[12]:


pip install missingno


# In[13]:


## Use a heatmap to visualize missing data to understand which variables have missing values and their patterns.

import missingno as msno

msno.matrix(air_quality_df)
plt.show()


# In[14]:


# Check the data types of each column

print(air_quality_df.dtypes)


#  ***Data pre-processing***

# In[15]:


# Check for missing values in the dataset

missing_values = air_quality_df.isna().sum()
print("Missing values in each column:\n", missing_values)


# In[16]:


## Handle Missing Values
## Impute missing values in numerical columns with the mean

numerical_cols = air_quality_df.select_dtypes(include=['float64', 'int64']).columns
air_quality_df[numerical_cols] = air_quality_df[numerical_cols].fillna(air_quality_df[numerical_cols].mean())


# In[17]:


## For categorical columns, fill missing values with the mode - most frequent value 

categorical_cols = air_quality_df.select_dtypes(include=['object']).columns
air_quality_df[categorical_cols] = air_quality_df[categorical_cols].fillna(air_quality_df[categorical_cols].mode().iloc[0])


# In[18]:


# checking to see if missing values are handled

print("Missing values after imputation:\n", air_quality_df.isna().sum())


# In[19]:


# Check for duplicates and count them

duplicate_rows = air_quality_df.duplicated()
print(duplicate_rows.sum())


# In[20]:


# Display the duplicate rows

duplicates = air_quality_df[air_quality_df.duplicated()]
print(duplicates)


# Note : Feature engineering for combining year, month, day, and hour into a single Datetime column is done at the above steps. 
# ## Create Additional Time-Based Features: 

# In[21]:


# Create a binary 'Weekend' feature (1 for weekend, 0 for weekday)

air_quality_df['DayOfWeek'] = air_quality_df['Datetime'].dt.dayofweek
air_quality_df['Weekend'] = air_quality_df['DayOfWeek'].apply(lambda x: 1 if x >= 5 else 0)


# In[22]:


# Check the first few rows

print(air_quality_df[['Datetime', 'DayOfWeek', 'Weekend']].head())


# In[23]:


print(air_quality_df.head())


# In[24]:


# Create 'TimeOfDay' feature based on Hour

def get_time_of_day(hour):
    if 6 <= hour < 12:
        return 'Morning'
    elif 12 <= hour < 18:
        return 'Afternoon'
    elif 18 <= hour < 21:
        return 'Evening'
    else:
        return 'Night'

air_quality_df['TimeOfDay'] = air_quality_df['hour'].apply(get_time_of_day)

# Check the first few rows
print(air_quality_df[['hour', 'TimeOfDay']].head())


# In[25]:


# Save the new merged dataset to a CSV file 

air_quality_df.to_csv("updated_air_quality.csv", index=False)


# In[26]:


## Removing Unnecessary Columns:
## Drop unnecessary columns

air_quality_df.drop(['year', 'month', 'day', 'hour'], axis=1, inplace=True)


# In[27]:


# Verify if the columns are dropped

print(air_quality_df.head())


# ***Statistics/computation-based analysis and Visualisation***

# In[28]:


## generating general statistical summaries of the numerical and categorical variables
# Display overall dataset statistics

print(air_quality_df.describe(include='all'))


# ## Univariate analysis for the distribution of a all variables and Bar Chart for Categorical Data 

# In[29]:


# List of pollutant columns that you want to visualize
pollutants = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']

# Set up the matplotlib figure with an appropriate size
plt.figure(figsize=(16, 12))

# Loop through each pollutant and plot its histogram with a Kernel Density Estimate
for i, pollutant in enumerate(pollutants):
    plt.subplot(2, 3, i + 1)  # Create a 2x3 grid of subplots
    sns.histplot(air_quality_df[pollutant], kde=True, bins=30)
    plt.title(f'Distribution of {pollutant}')
    plt.xlabel(pollutant)
    plt.ylabel('Frequency')

plt.tight_layout()  # Adjust subplots for a clean layout
plt.show()


# In[30]:


## Bar charts for the distribution of records across different times of day

plt.figure(figsize=(8, 5))
sns.countplot(x='TimeOfDay', data=air_quality_df, order=['Morning', 'Afternoon', 'Evening', 'Night'])
plt.title('Record Count by Time of Day')
plt.xlabel('Time of Day')
plt.ylabel('Count')
plt.show()


# In[31]:


##  Bar charts of the records fall on weekends vs weekdays,

plt.figure(figsize=(6, 4))
sns.countplot(x='Weekend', data=air_quality_df, palette='viridis')
plt.title('Record Count by Weekend Indicator')
plt.xlabel('Weekend (0 = Weekday, 1 = Weekend)')
plt.ylabel('Count')
plt.show()


# In[32]:


# Plot a bar chart showing counts for each category
# note to myself: every category has 35064 records

plt.figure(figsize=(8, 5))
sns.countplot(x='Category', data=air_quality_df, palette='Set2')
plt.title('Record Count by Category')
plt.xlabel('Category')
plt.ylabel('Count')
plt.show()


# ## Bivariate Analysis for the relationship between two variables: Scatter Plot

# In[33]:


## Scatter Plot: PM2.5 vs PM10

plt.figure(figsize=(10, 6))
sns.scatterplot(x='PM2.5', y='PM10', data=air_quality_df)
plt.title('Scatter Plot of PM2.5 vs PM10')
plt.xlabel('PM2.5')
plt.ylabel('PM10')
plt.show()

## note to myself : higher PM2.5 values tend to coincide with higher PM10 values.


# In[34]:


# Scatter plot of PM2.5 vs NO2

plt.figure(figsize=(10, 6))
sns.scatterplot(x='PM2.5', y='NO2', data=air_quality_df, hue='Category', palette='Set2')
plt.title('PM2.5 vs NO2 by Category')
plt.xlabel('PM2.5')
plt.ylabel('NO2')
plt.show()


# In[35]:


## Boxplot for PM2.5 by Category

plt.figure(figsize=(10, 6))
sns.boxplot(x='Category', y='PM2.5', data=air_quality_df, palette='Set2')
plt.title('PM2.5 Distribution by Category')
plt.xlabel('Category')
plt.ylabel('PM2.5')
plt.show()


# In[42]:


# Boxplot for PM2.5 by Day of Week

plt.figure(figsize=(12, 6))
sns.boxplot(x='DayOfWeek', y='PM2.5', data=air_quality_df, palette='Set2')
plt.title('PM2.5 Distribution by Day of Week')
plt.xlabel('Day of Week')
plt.ylabel('PM2.5')
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.show()


# ## Multivariate Analysis for interactions between several variables at once          

# In[37]:


##  Correlation Heatmap of Pollutants
## used only the numeric pollutant columns

pollutants = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']
corr_matrix = air_quality_df[pollutants].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap of Pollutants')
plt.show()


# ## Time Series Visualization using Line Plot for Daily Average PM2.5

# In[38]:


# Ensure your Datetime is set as index for resampling
air_quality_df.set_index('Datetime', inplace=True)

# Resample to daily mean
daily_PM25 = air_quality_df['PM2.5'].resample('D').mean()

plt.figure(figsize=(14, 7))
daily_PM25.plot()
plt.title('Daily Average PM2.5 Concentration Over Time')
plt.xlabel('Date')
plt.ylabel('Average PM2.5')
plt.show()

# Reset index if needed for further analysis
air_quality_df.reset_index(inplace=True)


# In[40]:


##checking to see non linear features Himanshi

import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 5))
sns.scatterplot(x='TEMP', y='PM2.5', data=air_quality_df, alpha=0.3)
plt.title('Temperature vs PM2.5')
plt.xlabel('Temperature (°C)')
plt.ylabel('PM2.5 (µg/m³)')
plt.grid(True)
plt.show()


# In[41]:


##checking to see non linear features Himanshi

plt.figure(figsize=(8, 5))
sns.scatterplot(x='WSPM', y='PM10', data=air_quality_df, alpha=0.3, color='green')
plt.title('Wind Speed vs PM10')
plt.xlabel('Wind Speed (m/s)')
plt.ylabel('PM10 (µg/m³)')
plt.grid(True)
plt.show()


# Task 3:  building machine-learning model -  Random Forest 
# It handles non-linear relationships,

# In[61]:


X = pd.get_dummies(X, drop_first=True)


# In[62]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# In[63]:


#note: feature scaling -Dropped columns not useful for modeling(done in previous steps- 'Date', 'day', 'month', 'year', 'hour')
## Filled missing numerical values with median (done in previous steps)


from sklearn.model_selection import train_test_split

#selecting features
features = ['PM10', 'SO2', 'NO2', 'CO','O3' ,'TEMP', 'PRES', 'DEWP', 'RAIN', 'WSPM']
X = air_quality_df[features]
y = air_quality_df['PM2.5']

# Split into train and test (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[64]:


from sklearn.ensemble import RandomForestRegressor

# Create the model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

# Fit to training data
rf_model.fit(X_train, y_train)


# In[65]:


from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Predict on test data
y_predict = rf_model.predict(X_test)

# Evaluation metrics
mse = mean_squared_error(y_test, y_predict)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_predict)

print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R² Score: {r2:.2f}")
print(f"Mean Squared Error: {mae:.2f}")


# In[66]:


import pandas as pd
import matplotlib.pyplot as plt

# Create DataFrame of feature importances
feature_importance = pd.DataFrame({
    'Feature': features,
    'Importance': rf_model.feature_importances_
}).sort_values(by='Importance', ascending=False)

# Plot
plt.figure(figsize=(8, 5))
sns.barplot(data=feature_importance, x='Importance', y='Feature', palette='Blues_d')
plt.title('Feature Importance - Random Forest')
plt.xlabel('Importance')
plt.ylabel('Features')
plt.grid(True)
plt.show()


# In[ ]:




