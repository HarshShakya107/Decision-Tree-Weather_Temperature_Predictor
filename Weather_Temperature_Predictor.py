import streamlit as st
import pandas as pd
import joblib


st.set_page_config(
    page_title="🌤️ Weather Temperature Predictor",
    page_icon="🌦️",
    layout="wide"
)


st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #89f7fe 0%, #66a6ff 100%);
    color: #1e293b;
    font-family: 'Poppins', sans-serif;
}
h1, h2, h3 {
    color: #1e3a8a;
    text-align: center;
}
.sidebar .sidebar-content {
    background-color: #f1f5f9;
    border-radius: 10px;
}
.stButton>button {
    background: linear-gradient(90deg, #007aff, #00c6ff);
    color: white;
    border-radius: 10px;
    font-size: 18px;
    font-weight: 600;
    padding: 0.6em 1.2em;
}
</style>
""", unsafe_allow_html=True)


st.title("🌦️ Weather Temperature Predictor")
st.markdown("### Predict **Temperature (°C)** from real-time weather conditions using a trained Decision Tree Regression model.")


try:
    model = joblib.load("decision_tree_weather_model.joblib")
except:
    st.error("❌ Model file not found! Please make sure `decision_tree_weather_model.joblib` is in the same folder.")
    st.stop()


st.sidebar.header("🧮 Weather Inputs")


app_temp = st.sidebar.number_input("Apparent Temperature (°C)", -20.0, 60.0, 22.0)
humidity = st.sidebar.slider("Humidity", 0.0, 1.0, 0.6)
wind_speed = st.sidebar.number_input("Wind Speed (km/h)", 0.0, 120.0, 10.0)
wind_bearing = st.sidebar.slider("Wind Bearing (°)", 0, 360, 180)
visibility = st.sidebar.number_input("Visibility (km)", 0.0, 20.0, 10.0)
pressure = st.sidebar.number_input("Pressure (millibars)", 900.0, 1100.0, 1012.0)


summary_options = [
    'Breezy and Dry', 'Breezy and Foggy', 'Breezy and Mostly Cloudy',
    'Breezy and Overcast', 'Breezy and Partly Cloudy', 'Clear',
    'Dangerously Windy and Partly Cloudy', 'Drizzle', 'Dry',
    'Dry and Mostly Cloudy', 'Dry and Partly Cloudy', 'Foggy',
    'Humid and Mostly Cloudy', 'Humid and Overcast', 'Humid and Partly Cloudy',
    'Light Rain', 'Mostly Cloudy', 'Overcast', 'Partly Cloudy', 'Rain',
    'Windy', 'Windy and Dry', 'Windy and Foggy', 'Windy and Mostly Cloudy',
    'Windy and Overcast', 'Windy and Partly Cloudy'
]
summary = st.sidebar.selectbox("Weather Summary", summary_options)
precip_snow = st.sidebar.checkbox("❄️ Snowing?", value=False)

input_dict = {
    'Apparent Temperature (C)': [app_temp],
    'Humidity': [humidity],
    'Wind Speed (km/h)': [wind_speed],
    'Wind Bearing (degrees)': [wind_bearing],
    'Visibility (km)': [visibility],
    'Pressure (millibars)': [pressure],
    'Precip Type_snow': [1 if precip_snow else 0]
}


for s in summary_options:
    col_name = f'Summary_{s}'
    input_dict[col_name] = [1 if s == summary else 0]


input_df = pd.DataFrame(input_dict)


if hasattr(model, "feature_names_in_"):
    input_df = input_df.reindex(columns=model.feature_names_in_, fill_value=0)
else:
    st.warning("⚠️ Model does not have `feature_names_in_` attribute — ensure columns match training order manually.")


st.markdown("---")
st.subheader("🌡️ Prediction Result")

if st.button("🔮 Predict Temperature"):
    try:
        pred_temp = model.predict(input_df)[0]
        st.success(f"### 🌤️ Predicted Temperature: **{pred_temp:.2f} °C**")
        st.balloons()

        with st.expander("📋 View Input Details"):
            st.dataframe(input_df.T, use_container_width=True)

    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")


st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#1e3a8a;'>Made with ❤️ using Streamlit & Decision Tree Regression</p>",
    unsafe_allow_html=True
)

