import streamlit as st
import requests
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from datetime import datetime, timedelta
import pytz

API_KEY = 'your_api_key_here'
BASE_URL = 'https://api.openweathermap.org/data/2.5/'

# Weather Icons (Static)
weather_icons = {
    "Clear": "☀️",
    "Clouds": "☁️",
    "Rain": "🌧",
    "Snow": "❄️",
    "Thunderstorm": "⛈",
    "Fog": "🌫",
    "Mist": "🌫",
}

# Background Gradients
backgrounds = {
    "Clear": "linear-gradient(to right, #ff9a9e, #fad0c4)",
    "Clouds": "linear-gradient(to right, #4B79A1, #283E51)",
    "Rain": "linear-gradient(to right, #3a6186, #89253e)",
    "Snow": "linear-gradient(to right, #83a4d4, #b6fbff)",
    "Thunderstorm": "linear-gradient(to right, #3a7bd5, #3a6073)",
    "Fog": "linear-gradient(to right, #757F9A, #D7DDE8)",
}

# Function to get real-time weather data
def get_current_weather(city):
    url = f"{BASE_URL}weather?q={city}&appid={API_KEY}&units=metric"
    response = requests.get(url)
    data = response.json()
    
    if response.status_code != 200:
        return None  # Handle errors

    return {
        'city': data['name'],
        'current_temp': round(data['main']['temp']),
        'feels_like': round(data['main']['feels_like']),
        'temp_min': round(data['main']['temp_min']),
        'temp_max': round(data['main']['temp_max']),
        'humidity': round(data['main']['humidity']),
        'description': data['weather'][0]['description'],
        'country': data['sys']['country'],
        'wind_gust_dir': data['wind']['deg'],
        'wind_gust_speed': data['wind']['speed'],
        'pressure': data['main']['pressure']
    }

# Load and prepare historical data
def read_historical_data(filename):
    df = pd.read_csv(filename)
    df = df.dropna().drop_duplicates()
    return df

def prepare_data(data):
    le = LabelEncoder()
    data['WindGustDir'] = le.fit_transform(data['WindGustDir'])
    data['RainTomorrow'] = le.fit_transform(data['RainTomorrow'])
    X = data[['MinTemp', 'MaxTemp', 'WindGustDir', 'WindGustSpeed', 'Humidity', 'Pressure', 'Temp']]
    y = data['RainTomorrow']
    return X, y, le

def train_rain_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def prepare_regression_data(data, feature):
    X, y = [], []
    for i in range(len(data) - 1):
        X.append(data[feature].iloc[i])
        y.append(data[feature].iloc[i+1])
    return np.array(X).reshape(-1, 1), np.array(y)

def train_regression_model(X, y):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

def predict_future(model, current_value):
    predictions = [current_value]
    for _ in range(5):
        next_value = model.predict(np.array([[predictions[-1]]]))
        predictions.append(next_value[0])
    return predictions[1:]


# Streamlit Page Configuration
st.set_page_config(page_title="The Weather App", layout="wide")
# Sidebar Input
st.sidebar.title("🌤 Enter City")
city = st.sidebar.text_input("City Name", "New Delhi")
st.sidebar.write("🔄 **Live Weather Updates**")
st.sidebar.markdown("---")

if city:
    weather_data = get_current_weather(city)

    if weather_data:
        st.subheader(f"Current Weather in {weather_data['city']}, {weather_data['country']}")
        st.write(f"🌡️ Temperature: {weather_data['current_temp']}°C (Feels Like: {weather_data['feels_like']}°C)")
        st.write(f"📉 Min: {weather_data['temp_min']}°C | 📈 Max: {weather_data['temp_max']}°C")
        st.write(f"💧 Humidity: {weather_data['humidity']}%")
        st.write(f"🌬️ Wind Speed: {weather_data['wind_gust_speed']} m/s, Direction: {weather_data['wind_gust_dir']}°")
        st.write(f"🔍 Condition: {weather_data['description'].title()}")

        # Load historical data and train models
        historical_data = read_historical_data('weather.csv')
        X, y, le = prepare_data(historical_data)
        rain_model = train_rain_model(X, y)

        # Rain Prediction
        rain_prediction = rain_model.predict(pd.DataFrame({
            'MinTemp': [weather_data['temp_min']],
            'MaxTemp': [weather_data['temp_max']],
            'WindGustDir': [0],  # Placeholder for missing encoded values
            'WindGustSpeed': [weather_data['wind_gust_speed']],
            'Humidity': [weather_data['humidity']],
            'Pressure': [weather_data['pressure']],
            'Temp': [weather_data['current_temp']]
        }))[0]

        st.write(f"🌧️ Rain Prediction: {'Yes' if rain_prediction else 'No'}")
        

        # Future Predictions
        x_temp, y_temp = prepare_regression_data(historical_data, 'Temp')
        x_humidity, y_humidity = prepare_regression_data(historical_data, 'Humidity')
        x_pressure, y_pressure = prepare_regression_data(historical_data, 'Pressure')

        temp_model = train_regression_model(x_temp, y_temp)
        humidity_model = train_regression_model(x_humidity, y_humidity)
        pressure_model = train_regression_model(x_pressure, y_pressure)

        future_temp = predict_future(temp_model, weather_data['current_temp'])
        future_humidity = predict_future(humidity_model, weather_data['humidity'])
        future_pressure = predict_future(pressure_model, weather_data['pressure'])

        timezone = pytz.timezone('Asia/Kolkata')
        now = datetime.now(timezone)
        next_hour = now + timedelta(hours=1)
        next_hour = next_hour.replace(minute=0, second=0, microsecond=0)

        future_times = [(next_hour + timedelta(hours=i)).strftime('%H:00') for i in range(5)]

        # Display Predictions
        st.subheader("Future Weather Predictions")
        future_df = pd.DataFrame({
            'Time': future_times,
            'Temperature (°C)': [round(temp, 1) for temp in future_temp],
            'Humidity (%)': [round(hum, 1) for hum in future_humidity],
            'Pressure (hPa)': [round(press, 1) for press in future_pressure]
        })
        st.table(future_df)

    else:
        st.error("City not found! Please check the name and try again.")

