import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import missingno as msno

# Set up layout
st.set_page_config(page_title="Air Quality App", layout="wide")

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("final_air_quality.csv")
    
air_quality_df = load_data()
df = air_quality_df.copy()

# Sidebar navigation
page = st.sidebar.radio("Navigation", ["Home", "Exploratory Data Analysis", "Modeling and Prediction"])
#st.sidebar.title("Navigation")
#home_button = st.sidebar.button("Home")
#eda_button = st.sidebar.button("Exploratory Data Analysis")
#modeling_button = st.sidebar.button("Modeling and Prediction")

st.markdown(
    """
    <style>
    .reportview-container {
        background: url("https://images.app.goo.gl/LFCobouKtT7oZ7Qv7")
    }
   st.sidebar{
        background: url("https://images.app.goo.gl/LFCobouKtT7oZ7Qv7")
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ----------------------------------------
# Home Page
# ----------------------------------------

if page == "Home":
    st.title("Air Quality Monitoring Dashboard")
    st.markdown("Welcome to the **Air Quality Analysis App**")
    st.markdown("This Streamlit-based web application provides a comprehensive analysis of air quality data helping users to explore pollution trends")

    st.markdown("- Explore China air pollution data (Urban, Suburban, Rural, Industrial)")
    st.markdown("- Understand pollutant trends over time")
    st.markdown("- Build predictive models using machine learning")

    st.metric("Total Records", len(air_quality_df))
    st.metric("Monitoring Sites", air_quality_df['station'].nunique())
 
    st.markdown("**Urban Site : Wanshouxigong**")
    st.markdown(" Located in central Beijing, this site represents typical urban air quality.Surrounded by dense traffic and residential/commercial zones")

    st.markdown("**Suburban Site : Changping**")
    st.markdown(" Located in the northern outskirts of Beijing. Considered a suburban area, less dense than the city center but still developed")

    st.markdown("**Rural Site : Huairou**")
    st.markdown(" Situated in the northern rural region of Beijing. Characterized by lower population density and more natural surroundings")
    
    st.markdown("**Industrial Site : Aotizhongxin**")
    st.markdown(" Known as a hotspot due to its proximity to Olympic venues and high development zones. This site is often monitored closely")

# ----------------------------------------
# EDA Page
# ----------------------------------------
elif page == "Exploratory Data Analysis":
    st.title(" Exploratory Data Analysis")

    st.subheader("Dataset Description")
    st.write(df.describe())

    st.subheader("Missing Values")
    st.write(df.isnull().sum())

    st.subheader("Distribution of Pollutants in a Bar Chart")
    pollutants = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']
    fig, axs = plt.subplots(2, 3, figsize=(16, 10))
    axs = axs.flatten()

    for i, pollutant in enumerate(pollutants):
        sns.histplot(df[pollutant], kde=True, bins=30, ax=axs[i])
        axs[i].set_title(f'Distribution of {pollutant}')
        axs[i].set_xlabel(pollutant)
        axs[i].set_ylabel('Frequency')
    plt.tight_layout()
    st.pyplot(fig)

    st.subheader("Record Count by Time of Day")
    plt.figure(figsize=(8, 5))
    sns.countplot(x='TimeOfDay', data=air_quality_df, order=['Morning', 'Afternoon', 'Evening', 'Night'])
    plt.title('Record Count by Time of Day')
    plt.xlabel('Time of Day')
    plt.ylabel('Count')
    st.pyplot(plt)

    st.subheader("Average Pollutant Levels on Weekdays vs Weekends")
    pollutants = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']
    # Group by 'Weekend' and calculate average
    avg_pollutants = air_quality_df.groupby('Weekend')[pollutants].mean().T
    avg_pollutants.index.name = 'Pollutant'
    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))
    avg_pollutants.plot(kind='bar', ax=ax, colormap='Set2')
    ax.set_title('Average Pollutant Levels: Weekdays vs Weekends')
    ax.set_xlabel('Pollutant')
    ax.set_ylabel('Average Concentration')
    ax.legend(title='Weekend', labels=['Weekday (0)', 'Weekend (1)'])
    plt.xticks(rotation=0)
    st.pyplot(fig)
    

    st.subheader("Record Count: Weekends vs Weekdays")
    plt.figure(figsize=(6, 4))
    sns.countplot(x='Weekend', data=air_quality_df, palette='viridis')
    plt.title('Record Count by Weekend Indicator')
    plt.xlabel('Weekend (0 = Weekday, 1 = Weekend)')
    plt.ylabel('Count')
    st.pyplot(plt)

    st.subheader("Scatter Plot Grid of Key Air Pollutants")
    pollutants = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']
    # Set up figure
    fig, axes = plt.subplots(nrows=5, ncols=3, figsize=(20, 25))
    axes = axes.flatten()
    plot_num = 0

    # Plot each unique pair
    for i in range(len(pollutants)):
        for j in range(i + 1, len(pollutants)):
            sns.scatterplot(
                x=air_quality_df[pollutants[i]],
                y=air_quality_df[pollutants[j]],
                ax=axes[plot_num],
                alpha=0.5
            )
            axes[plot_num].set_title(f'{pollutants[i]} vs {pollutants[j]}')
            axes[plot_num].set_xlabel(pollutants[i])
            axes[plot_num].set_ylabel(pollutants[j])
            plot_num += 1
    # Hide unused subplots
    for k in range(plot_num, len(axes)):
        fig.delaxes(axes[k])
    plt.tight_layout()
    st.pyplot(fig)
    st.markdown("""This scatter plot shows the relationship between all unique pollutant pair comparisons. Helps users visually assess correlation patterns.""")

    pollutants = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']
    sns.set(style="whitegrid")
    st.subheader("Distribution of Each Pollutant Across Areas")
    fig, axs = plt.subplots(2, 3, figsize=(18, 10))
    for i, pollutant in enumerate(pollutants):
        row, col = divmod(i, 3)
        sns.boxplot(x='Category', y=pollutant, data=air_quality_df, ax=axs[row][col], palette='Set2')
        axs[row][col].set_title(f'{pollutant} by Area')
        axs[row][col].set_xlabel('Area')
        axs[row][col].set_ylabel(pollutant)
    plt.tight_layout()
    st.pyplot(fig)
    st.markdown("""This section presents boxplots for each air pollutant to compare their concentration levels across different areas. This helps to understand **Median values** of pollutants. **Spread or variability** of data within each area.**Outliers** unusually high or low readings and **Comparative pollution levels** between different areas. """)

    st.subheader("Correlation Heatmap of Pollutants")
    pollutants = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']
    corr_matrix = air_quality_df[pollutants].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5, ax=ax)
    ax.set_title('Correlation Heatmap of Pollutants')
    st.pyplot(fig)
    st.markdown("This heatmap shows how strongly each pair of pollutants is related to one another, based on their Pearson correlation coefficient (values between -1 and 1)")
    st.markdown ("A high positive correlation between PM2.5 and PM10 means they often rise and fall together")
    st.markdown("A low or negative correlation between O3 and NO2 might indicate different sources or behaviors in the environment")

    st.subheader("Daily Average Pollutant Levels Over Time")
    air_quality_df['Datetime'] = pd.to_datetime(air_quality_df['Datetime'])
    air_quality_df.set_index('Datetime', inplace=True)
    st.markdown("This graph will help to observe trends, seasonality, or spikes in pollutant levels over time.Especially useful for identifying long-term pollution patterns or days with unusual spikes.")
    
    pollutants = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']
  # Set up 2x3 grid of subplots
    fig, axs = plt.subplots(2, 3, figsize=(18, 10))
    axs = axs.flatten()  # Flatten the 2D array of axes to iterate easily
    for i, pollutant in enumerate(pollutants):
        daily_avg = air_quality_df[pollutant].resample('D').mean()
        axs[i].plot(daily_avg, color='darkgreen')
        axs[i].set_title(f'{pollutant} Over Time', fontsize=12)
        axs[i].set_xlabel('Date')
        axs[i].set_ylabel(f'{pollutant} Level')
        axs[i].tick_params(axis='x', labelrotation=45)
        axs[i].grid(True)    
    plt.tight_layout()
    st.pyplot(fig)

    st.subheader("Temperature vs Pollutant Levels")
    st.markdown("These plots help to visually assess Temperature changes over time (e.g., summer vs winter). Some pollutants behave differently in different weather.Unusually high pollutant values at certain temperatures may stand out as anomalies")
    pollutants = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']
    for pollutant in pollutants:
        st.markdown(f"### Temperature vs {pollutant}")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.scatterplot(x='TEMP', y=pollutant, data=air_quality_df, alpha=0.3, ax=ax)
        ax.set_title(f'Temperature vs {pollutant}')
        ax.set_xlabel('Temperature (°C)')
        ax.set_ylabel(f'{pollutant} (µg/m³)')
        ax.grid(True)
        st.pyplot(fig)


    st.subheader("Scatter Plots: Wind Speed vs Pollutants")
    st.markdown("These plots helps to visually explore how wind speed influence the concentration levels of different pollutants. Correlation values gives a quantitative sense of how strongly wind speed is associated with changes in pollutant levels. A negative correlation may indicate that higher wind speed disperses pollutants.")
    pollutants = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']
    for pollutant in pollutants:
        correlation = air_quality_df['WSPM'].corr(air_quality_df[pollutant])
        st.markdown(f"### Wind Speed vs {pollutant} (Correlation = {correlation:.2f})")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.scatterplot(x='WSPM', y=pollutant, data=air_quality_df, alpha=0.3, color='green', ax=ax)
        ax.set_title(f'WSPM vs {pollutant}')
        ax.set_xlabel('Wind Speed (m/s)')
        ax.set_ylabel(f'{pollutant} (µg/m³)')
        ax.grid(True)
        st.pyplot(fig)
    
# ----------------------------------------
# Modeling Page
# ----------------------------------------
elif page == "Modeling and Prediction":
    st.title("Modeling and Prediction")

       
    # Feature selection
    features = ['PM10', 'SO2', 'NO2', 'CO','O3' ,'TEMP', 'PRES', 'DEWP', 'RAIN', 'WSPM']
    X = air_quality_df[features]
    y = air_quality_df['PM2.5']
    
    # Split into train and test (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create the model
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    
    # Fit the model
    rf_model.fit(X_train, y_train)

    # Predict on test data
    y_predict = rf_model.predict(X_test)

    # Evaluation metrics
    mse = mean_squared_error(y_test, y_predict)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_predict)

    # Display metrics in Streamlit
    st.subheader("Model Evaluation Metrics:")
    st.write(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    st.write(f"R² Score: {r2:.2f}")
    st.write(f"Mean Squared Error: {mse:.2f}")

    # Feature importance
    feature_importance = pd.DataFrame({
        'Feature': features,
        'Importance': rf_model.feature_importances_
    }).sort_values(by='Importance', ascending=False)

    # Plot Feature Importance
    st.subheader("Feature Importance (Random Forest)")
    plt.figure(figsize=(8, 5))
    sns.barplot(data=feature_importance, x='Importance', y='Feature', palette='Blues_d')
    plt.title('Feature Importance - Random Forest')
    plt.xlabel('Importance')
    plt.ylabel('Features')
    plt.grid(True)
    st.pyplot(plt)

    # Model prediction section
    st.subheader("Predictions for Test Data")
    predictions_df = pd.DataFrame({
        'True PM2.5': y_test,
        'Predicted PM2.5': y_predict
    })

    # Show a table of predictions
    st.write(predictions_df.head())