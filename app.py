import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding
from tensorflow.keras.models import Model
import tkinter as tk
from tkinter import messagebox

# Example larger dataset
data = {
    "community_id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "sea_level_rise": [0.5, 1.0, 1.5, 0.8, 1.2, 0.6, 0.9, 1.3, 1.7, 1.0],
    "extreme_weather_events": [5, 10, 15, 8, 12, 6, 9, 11, 14, 7],
    "agricultural_changes": [0.2, 0.3, 0.4, 0.25, 0.35, 0.15, 0.28, 0.38, 0.42, 0.27],
    "current_strategies": [
        "Building sea walls and conducting mangrove restoration projects.",
        "Developing early warning systems and improving emergency response plans.",
        "Promoting drought-resistant crop varieties and efficient irrigation practices.",
        "Investing in green infrastructure to manage stormwater and reduce heat island effect.",
        "Updating building codes and enhancing structural resilience to extreme weather.",
        "Educating communities on disaster preparedness and fostering community resilience.",
        "Implementing public transportation improvements to reduce emissions and traffic.",
        "Supporting local adaptation efforts through community-driven initiatives.",
        "Enhancing coastal zone management and integrating climate adaptation into urban planning.",
        "Adopting policies to incentivize renewable energy adoption and reduce carbon footprint."
    ]
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Preprocessing for model training
# Define the input and output sequences
input_sequences = df[['sea_level_rise', 'extreme_weather_events', 'agricultural_changes']].values
output_sequences = df['current_strategies'].values

# Tokenize the output sequences
tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(output_sequences)
output_sequences = tokenizer.texts_to_sequences(output_sequences)

# Add a start token to the output sequences
start_token = 1  # Assuming 1 is the start token
output_sequences = [[start_token] + seq for seq in output_sequences]
output_sequences = tf.keras.preprocessing.sequence.pad_sequences(output_sequences, padding='post')

# Reshape input sequences to 3D
input_sequences = input_sequences.reshape((input_sequences.shape[0], 1, input_sequences.shape[1]))

# Define model parameters
embedding_dim = 100
lstm_units = 150
epochs = 2000  # Adjust epochs for further training

# Build the encoder
encoder_inputs = Input(shape=(input_sequences.shape[1], input_sequences.shape[2]))
encoder_lstm = LSTM(lstm_units, return_state=True)
encoder_outputs, state_h, state_c = encoder_lstm(encoder_inputs)
encoder_states = [state_h, state_c]

# Build the decoder
decoder_inputs = Input(shape=(None,))
decoder_embedding = Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=embedding_dim)(decoder_inputs)
decoder_lstm = LSTM(lstm_units, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)
decoder_dense = Dense(len(tokenizer.word_index) + 1, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

# Build the model
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

# Train the model
decoder_input_sequences = np.zeros_like(output_sequences)
for i in range(len(output_sequences)):
    decoder_input_sequences[i, 1:] = output_sequences[i, :-1]
    decoder_input_sequences[i, 0] = start_token

decoder_input_sequences = tf.keras.preprocessing.sequence.pad_sequences(decoder_input_sequences, padding='post')
decoder_target_sequences = np.expand_dims(output_sequences, -1)

model.fit([input_sequences, decoder_input_sequences], decoder_target_sequences, epochs=epochs, verbose=1)

# Define the inference models
encoder_model = Model(encoder_inputs, encoder_states)

decoder_state_input_h = Input(shape=(lstm_units,))
decoder_state_input_c = Input(shape=(lstm_units,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

decoder_embedding2 = Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=embedding_dim)
decoder_embedding_outputs = decoder_embedding2(decoder_inputs)

decoder_outputs, state_h, state_c = decoder_lstm(
    decoder_embedding_outputs, initial_state=decoder_states_inputs
)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)

decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs] + decoder_states
)

# Function to decode sequences
def decode_sequence(input_seq):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1.
    target_seq = np.array([start_token]).reshape(1, 1)

    # Sampling loop for a batch of sequences
    stop_condition = False
    decoded_sentence = []
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)

        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_word = tokenizer.index_word.get(sampled_token_index, '')

        if sampled_word:
            decoded_sentence.append(sampled_word)

        if sampled_word == '' or len(decoded_sentence) > 50:
            stop_condition = True

        target_seq = np.array([sampled_token_index]).reshape(1, 1)
        states_value = [h, c]

    return ' '.join(decoded_sentence)

# Tkinter GUI setup
def predict_strategy():
    try:
        sea_level_rise = float(sea_level_rise_entry.get())
        extreme_weather_events = int(extreme_weather_events_entry.get())
        agricultural_changes = float(agricultural_changes_entry.get())

        community_data = np.array([[sea_level_rise, extreme_weather_events, agricultural_changes]])
        community_data = community_data.reshape((1, 1, 3))

        strategy = decode_sequence(community_data)

        strategy_text.delete(1.0, tk.END)
        strategy_text.insert(tk.END, strategy)

    except ValueError:
        messagebox.showerror("Error", "Please enter valid numeric values.")
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {str(e)}")

# Tkinter window
root = tk.Tk()
root.title("Climate Change Adaptation Strategy Predictor")

# Frame for input fields
input_frame = tk.Frame(root, padx=10, pady=10)
input_frame.pack()

# Input fields
tk.Label(input_frame, text="Sea Level Rise").grid(row=0, column=0, padx=5, pady=5)
sea_level_rise_entry = tk.Entry(input_frame)
sea_level_rise_entry.grid(row=0, column=1, padx=5, pady=5)

tk.Label(input_frame, text="Extreme Weather Events").grid(row=1, column=0, padx=5, pady=5)
extreme_weather_events_entry = tk.Entry(input_frame)
extreme_weather_events_entry.grid(row=1, column=1, padx=5, pady=5)

tk.Label(input_frame, text="Agricultural Changes").grid(row=2, column=0, padx=5, pady=5)
agricultural_changes_entry = tk.Entry(input_frame)
agricultural_changes_entry.grid(row=2, column=1, padx=5, pady=5)

# Button to predict strategy
predict_button = tk.Button(root, text="Predict Strategy", command=predict_strategy)
predict_button.pack(pady=10)

# Frame for displaying strategy
strategy_frame = tk.Frame(root, padx=10, pady=10, bg='lightgray')
strategy_frame.pack()

# Text area for displaying strategy
strategy_text = tk.Text(strategy_frame, width=50, height=5)
strategy_text.pack()

root.mainloop()
