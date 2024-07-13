import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error


class LSTMModelPredictor:
    def __init__(self, train_data_path, new_data_path, feature_name):
        self.train_data_path = train_data_path
        self.new_data_path = new_data_path
        self.model = None
        self.scaler = MinMaxScaler()
        self.is_scaler_fitted = False
        self.feature_name = feature_name

    def load_data(self, path):
        data = pd.read_csv(path)
        return data

    def preprocess_data(self, data):
        features = data[[self.feature_name]]
        target = data['temp']  # Assuming 'temp' is the target for all scenarios
        scaled_features = self.scaler.fit_transform(features)
        return scaled_features, target

    def create_sequences(self, data, seq_length):
        sequences = []
        targets = []
        for i in range(len(data) - seq_length):
            sequences.append(data[i:i + seq_length])
            targets.append(data[i + seq_length])
        return np.array(sequences), np.array(targets)

    def build_model(self, seq_length):
        self.model = Sequential()
        self.model.add(LSTM(50, return_sequences=True, input_shape=(seq_length, 1)))
        self.model.add(LSTM(50))
        self.model.add(Dense(1))
        self.model.compile(optimizer='adam', loss='mean_squared_error')

    def train_model(self, X_train, y_train, epochs=20, batch_size=32, validation_split=0.2):
        history = self.model.fit(X_train, y_train, epochs=epochs, validation_split=validation_split, batch_size=batch_size)
        return history

    def evaluate_model(self, y_true, y_pred):
        mse = mean_squared_error(y_true, y_pred)
        return mse

    def predict_new_data(self, new_data, seq_length):
        scaled_new_features = self.scaler.transform(new_data)
        X_new = self.create_sequences(scaled_new_features, seq_length)
        new_predictions = self.model.predict(X_new)
        new_predictions = self.scaler.inverse_transform(new_predictions)
        new_predictions = new_predictions.squeeze()
        return new_predictions

    def run(self, seq_length=10, epochs=20, batch_size=32, validation_split=0.2):
        # Load training data
        train_data = self.load_data(self.train_data_path)
        # Preprocess training data
        scaled_features, target = self.preprocess_data(train_data)
        # Create sequences for LSTM
        X_train, y_train = self.create_sequences(scaled_features, seq_length)
        # Build LSTM model
        self.build_model(seq_length)
        # Train LSTM model
        self.train_model(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=validation_split)
        # Example of predicting on new data
        new_data = self.load_data(self.new_data_path)
        new_features = new_data[[self.feature_name]]
        # Predict on new data
        new_predictions = self.predict_new_data(new_features, seq_length)
        return new_predictions[:60]  # Return the first 60 predictions


class RenewableEnergyRecommendation:
    def __init__(self):
        self.renewable_recommendations = {
            ('clearsky', 'low'): 'Solar Photovoltaic (PV) Panels',
            ('clearsky', 'medium'): 'Solar Thermal Power Plants',
            ('clearsky', 'high'): 'Concentrated Solar Power (CSP) Plants',
            ('cloudy', 'low'): 'Wind Turbines',
            ('cloudy', 'medium'): 'Wind Farms',
            ('cloudy', 'high'): 'Offshore Wind Farms',
            ('rainy', 'low'): 'Hydroelectric Power Plants',
            ('rainy', 'medium'): 'Tidal Power Generators',
            ('rainy', 'high'): 'Wave Energy Converters',
            ('hot', 'low'): 'Geothermal Power Plants',
            ('hot', 'medium'): 'Geothermal Heat Pumps',
            ('hot', 'high'): 'Enhanced Geothermal Systems',
            ('cold', 'low'): 'Biomass Power Plants',
            ('cold', 'medium'): 'Biogas Digesters',
            ('cold', 'high'): 'Biofuel Production Facilities'
            # Add more conditions and recommendations as needed
        }

    def recommend_plant(self, weather_condition, wind_condition):
        key = (weather_condition, wind_condition)
        if key in self.renewable_recommendations:
            return self.renewable_recommendations[key]
        else:
            return "No recommendation available for the given conditions"


class App:
    def __init__(self, root):
        self.root = root
        self.root.title("LSTM Model Predictor & Renewable Energy Recommendation")
        self.weather_options = ['clearsky', 'cloudy', 'rainy', 'hot', 'cold']
        self.wind_options = ['low', 'medium', 'high']

        # Initialize frames
        self.setup_frames()

    def setup_frames(self):
        # Data Prediction Frame
        self.data_prediction_frame = tk.Frame(self.root, padx=10, pady=10, borderwidth=2, relief="groove")
        self.data_prediction_frame.grid(row=0, column=0, padx=10, pady=10)

        self.label_train_path = tk.Label(self.data_prediction_frame, text="Train Data Path:")
        self.label_train_path.grid(row=0, column=0, padx=5, pady=5)
        self.entry_train_path = tk.Entry(self.data_prediction_frame, width=50)
        self.entry_train_path.grid(row=0, column=1, padx=5, pady=5)
        self.button_browse_train = tk.Button(self.data_prediction_frame, text="Browse", command=self.browse_train_file)
        self.button_browse_train.grid(row=0, column=2, padx=5, pady=5)

        self.label_new_path = tk.Label(self.data_prediction_frame, text="New Data Path:")
        self.label_new_path.grid(row=1, column=0, padx=5, pady=5)
        self.entry_new_path = tk.Entry(self.data_prediction_frame, width=50)
        self.entry_new_path.grid(row=1, column=1, padx=5, pady=5)
        self.button_browse_new = tk.Button(self.data_prediction_frame, text="Browse", command=self.browse_new_file)
        self.button_browse_new.grid(row=1, column=2, padx=5, pady=5)

        self.label_feature = tk.Label(self.data_prediction_frame, text="Feature:")
        self.label_feature.grid(row=2, column=0, padx=5, pady=5)
        self.entry_feature = tk.Entry(self.data_prediction_frame, width=50)
        self.entry_feature.grid(row=2, column=1, padx=5, pady=5)

        self.button_predict = tk.Button(self.data_prediction_frame, text="Predict", command=self.predict)
        self.button_predict.grid(row=3, column=0, columnspan=3, padx=5, pady=5)

        self.label_prediction_result = tk.Label(self.data_prediction_frame, text="Predictions:")
        self.label_prediction_result.grid(row=4, column=0, padx=5, pady=5)
        self.text_prediction_result = tk.Text(self.data_prediction_frame, height=10, width=60)
        self.text_prediction_result.grid(row=5, column=0, columnspan=3, padx=5, pady=5)

        # Renewable Energy Recommendation Frame
        self.recommendation_frame = tk.Frame(self.root, padx=10, pady=10, borderwidth=2, relief="groove")
        self.recommendation_frame.grid(row=1, column=0, padx=10, pady=10)

        self.label_weather_condition = tk.Label(self.recommendation_frame, text="Weather Condition:")
        self.label_weather_condition.grid(row=0, column=0, padx=5, pady=5)
        self.weather_condition_var = tk.StringVar(self.recommendation_frame)
        self.weather_condition_var.set(self.weather_options[0])
        self.option_menu_weather = tk.OptionMenu(self.recommendation_frame, self.weather_condition_var, *self.weather_options)
        self.option_menu_weather.grid(row=0, column=1, padx=5, pady=5)

        self.label_wind_condition = tk.Label(self.recommendation_frame, text="Wind Condition:")
        self.label_wind_condition.grid(row=1, column=0, padx=5, pady=5)
        self.wind_condition_var = tk.StringVar(self.recommendation_frame)
        self.wind_condition_var.set(self.wind_options[0])
        self.option_menu_wind = tk.OptionMenu(self.recommendation_frame, self.wind_condition_var, *self.wind_options)
        self.option_menu_wind.grid(row=1, column=1, padx=5, pady=5)

        self.button_recommend = tk.Button(self.recommendation_frame, text="Recommend", command=self.recommend)
        self.button_recommend.grid(row=2, column=0, columnspan=2, padx=5, pady=5)

        self.label_recommendation_result = tk.Label(self.recommendation_frame, text="Recommendation:")
        self.label_recommendation_result.grid(row=3, column=0, padx=5, pady=5)
        self.text_recommendation_result = tk.Text(self.recommendation_frame, height=5, width=40)
        self.text_recommendation_result.grid(row=4, column=0, columnspan=2, padx=5, pady=5)

    def browse_train_file(self):
        file_path = filedialog.askopenfilename()
        self.entry_train_path.insert(0, file_path)

    def browse_new_file(self):
        file_path = filedialog.askopenfilename()
        self.entry_new_path.insert(0, file_path)

    def predict(self):
        train_data_path = self.entry_train_path.get()
        new_data_path = self.entry_new_path.get()
        feature_name = self.entry_feature.get()

        if not train_data_path or not new_data_path or not feature_name:
            messagebox.showerror("Error", "Please provide all required inputs.")
            return

        lstm_predictor = LSTMModelPredictor(train_data_path, new_data_path, feature_name)
        predictions = lstm_predictor.run(seq_length=10, epochs=20, batch_size=32, validation_split=0.2)
        self.text_prediction_result.insert(tk.END, f"The next 60 days values predicted: {predictions}\n")

    def recommend(self):
        weather_condition = self.weather_condition_var.get()
        wind_condition = self.wind_condition_var.get()

        renewable_rec = RenewableEnergyRecommendation()
        recommendation = renewable_rec.recommend_plant(weather_condition, wind_condition)
        self.text_recommendation_result.insert(tk.END, f"Recommended renewable energy plant: {recommendation}\n")


if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
