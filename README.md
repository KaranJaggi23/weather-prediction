# Weather Prediction App

A real-time weather prediction application built using Streamlit and Machine Learning. This app fetches real-time weather data, predicts the likelihood of rain, and forecasts future weather conditions based on historical data.

## Prerequisites

- Python 3.7 or higher
- pip for installing dependencies

## Features

- Real-time Weather Data: Fetches current weather conditions from OpenWeather API.
- Rain Prediction: Uses a Random Forest Classifier to predict if it will rain tomorrow.
- Temperature, Humidity & Pressure Forecast: Uses a Random Forest Regressor for short-term future predictions.
- Beautiful UI: Interactive, gradient-based UI with weather icons.

## Installation & Setup

1️⃣ Clone the Repository

`git clone https://github.com/KaranJaggi23/weather-prediction.git`

2️⃣ Create a Virtual Environment (Optional but Recommended)

```
python -m venv venv
source venv/bin/activate    # macOS/Linux
venv\Scripts\activate       # Windows
```

3️⃣ Install Dependencies

```
cd weather-prediction
pip install -r requirements.txt
```

4️⃣ Set Up Your OpenWeather API Key

This application requires an API key from OpenWeather to fetch real-time weather data. Follow these steps to get your API key:

- Go to OpenWeather and create a free account.
- Navigate to the API keys section in your account dashboard.
- Copy the generated API key.
- Open app.py and replace your_openweather_api_key_here with your actual API key:

5️⃣ Run the Application

`python -m streamlit run app.py`

Open your browser and navigate to:

`http://127.0.0.1:8501/`

## Usage

- Enter a city name to fetch real-time weather data.
- View current weather details, including temperature, humidity, and wind speed.
- Get rain prediction based on weather conditions.
- See future weather trends for the next few hours.

## Technologies Used

- Backend: Python, Streamlit
- Machine Learning: scikit-learn (Random Forest Classifier & Regressor)
- Data Processing: Pandas, NumPy
- API Integration: OpenWeather API
