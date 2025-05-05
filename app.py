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
#df = air_quality_df.copy() 

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("final_air_quality.csv")
    
air_quality_df = load_data()
df = air_quality_df.copy()

# Sidebar navigation
page = st.sidebar.radio("Navigation", ["Home", "Exploratory Data Analysis", "Modeling and Prediction"])

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

    st.subheader("📌 Dataset Description")
    st.write(df.describe())

    st.subheader("📌 Missing Values")
    st.write(df.isnull().sum())

    st.subheader("📌 Boxplot: PM2.5 by Day of Week")
    fig, ax = plt.subplots()
    sns.boxplot(x='DayOfWeek', y='PM2.5', data=df, ax=ax)
    st.pyplot(fig)
        
    st.subheader("📌 Missing Data in a Heatmap")
    fig, ax = plt.subplots(figsize=(12, 6))
    msno.matrix(df, ax=ax)
    st.pyplot(fig)

    st.subheader("📌 Distribution of Pollutants in a Bar Chart")
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

    st.subheader("Record Count: Weekends vs Weekdays")
    plt.figure(figsize=(6, 4))
    sns.countplot(x='Weekend', data=air_quality_df, palette='viridis')
    plt.title('Record Count by Weekend Indicator')
    plt.xlabel('Weekend (0 = Weekday, 1 = Weekend)')
    plt.ylabel('Count')
    st.pyplot(plt)

    st.subheader("Correlation Between PM2.5 and PM10 Levels")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x='PM2.5', y='PM10', data=air_quality_df, ax=ax)
    ax.set_xlabel('PM2.5')
    ax.set_ylabel('PM10')
    st.pyplot(fig)
    st.markdown("""This scatter plot helps explore how concentrations of fine particles (PM2.5) relate to those of larger particles (PM10). Understanding this correlation can provide insights into pollution sources and help identify trends in air quality measurements.""")

    st.subheader("Relationship Between PM2.5 and NO2 by Air Quality Across Different Areas")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x='PM2.5', y='NO2', data=air_quality_df, hue='Category', palette='Set2', ax=ax)
    ax.set_xlabel('PM2.5')
    ax.set_ylabel('NO2')
    st.pyplot(fig)
    st.markdown("""This scatter plot shows the relationship between **PM2.5** and **NO2** levels, color-coded by station category (Urban, Suburban,Industrial, etc.).Different categories may show varying correlations, providing insights into how traffic, industry, or rural conditions impact pollution patterns.""")


# ----------------------------------------
# Modeling Page
# ----------------------------------------
elif page == "Modeling and Prediction":
    st.title("🤖 Modeling and Prediction")

    df = df.dropna()
    features = ['Temp', 'Dewp', 'Wspd', 'Rain', 'NO2', 'SO2', 'CO', 'O3']
    target = 'PM2.5'

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    st.subheader("📈 Model Performance")
    st.write(f"**Root Mean Squared Error (RMSE):** {rmse:.2f}")
    st.write(f"**R² Score:** {r2:.2f}")

    st.subheader("📌 Feature Importances")
    feature_importances = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
    st.bar_chart(feature_importances)
