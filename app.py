import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from keras.models import load_model
from keras.layers import Input, Embedding, LSTM, Dense
from keras.models import Model
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from lime import lime_text
import numpy as np
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from lime.lime_text import LimeTextExplainer

# Disable the PyplotGlobalUseWarning
st.set_option('deprecation.showPyplotGlobalUse', False)
# Load the pre-trained RoBERTa model and tokenizer
model_name = "roberta-base"
tokenizer = RobertaTokenizer.from_pretrained(model_name)
num_labels = 3  # Replace with the actual number of labels in your task
model = RobertaForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

# Define a function to preprocess the input text for the model
def preprocess_text(text):
    inputs = tokenizer.encode_plus(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    return inputs

# Define a function to get the model's prediction
def get_prediction(text):
    inputs = preprocess_text(text)
    outputs = model(**inputs)[0]
    logits = outputs.detach().cpu().numpy()
    return logits

# Load the tokenizer
@st.cache_resource
def load_tokenizer():
    return Tokenizer()

tokenizer = load_tokenizer()

import tensorflow as tf

@st.cache_resource
def load_pretrained_model():
    return tf.keras.models.load_model('keras model.h5')

model = load_pretrained_model()

# Maximum sequence length
maxlen = 100

# Human-readable labels
labels = ['Commenting', 'Ogling/Facial Expressions/Staring', 'Touching /Groping']

# Streamlit app
st.title("Harassment Detection")
st.write("Enter a description to predict harassment types.")

# Sidebar navigation
selected_page = st.sidebar.radio("Navigation", ["Incidents", "Safety Tips", "Reports", "Explainable AI"])

# Input text area for entering description
description = st.text_area("Description", "")

# Predict button
if st.button("Predict"):
    if description:
        # Tokenize the input description
        sequence = tokenizer.texts_to_sequences([description])
        padded_sequence = pad_sequences(sequence, maxlen=maxlen)
        
        # Use the pre-trained model to predict
        predictions = model.predict(padded_sequence)
        
        # Convert predictions to human-readable labels
        predicted_labels = [labels[i] for i in range(len(labels)) if predictions[0][i] >= 0.5]
        
        # Display the predicted labels
        if predicted_labels:
            st.write("Predicted harassment types:")
            for label in predicted_labels:
                st.write(label)
        else:
            st.write("No harassment detected.")
    else:
        st.write("Please enter a description.")
        
dev_data = pd.read_csv("dev.csv")
labels = ['Commenting', 'Ogling/Facial Expressions/Staring', 'Touching /Groping']
# Page content based on selected navigation'

if selected_page == "Incidents":
    st.write("This is the Incidents page where you can input and share your real life experiences to help others and improvise the community for women's safety")
    st.write("It is anonymous sharing of data, so you dont have have to feel shy or guilty.")
    st.write("It is not your fault and we all are there with you.")
elif selected_page == "Safety Tips":
    st.write("### Safety Tips for Women")
    st.write("Here are some safety tips and tricks to help prevent harassment and stay safe:")
    
    st.write("- Trust your instincts. If a situation or person makes you feel uncomfortable, leave or seek help immediately.")
    st.write("- Be aware of your surroundings and avoid isolated or poorly lit areas, especially at night.")
    st.write("- Carry a personal safety device, such as a whistle, pepper spray, or a personal alarm.")
    st.write("- Learn self-defense techniques to protect yourself in case of an attack.")
    st.write("- Avoid walking alone, especially at night. Walk with a friend or in a group whenever possible.")
    st.write("- Keep your phone charged and easily accessible in case you need to call for help.")
    st.write("- Dress confidently and avoid revealing clothing that may attract unwanted attention.")
    st.write("- Report any incidents of harassment or assault to the authorities.")
elif selected_page == "Reports":
    st.write("### Reports Page")

    # Count the number of 1's for each label
    label_counts = []
    for label in labels:
        label_counts.append(dev_data[label].sum())

    # Create a DataFrame for visualization
    df_visualization = pd.DataFrame({"Label": labels, "Count": label_counts})

    # Create a figure and axis objects
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot the bar chart
    sns.barplot(x="Label", y="Count", data=df_visualization, ax=ax)
    ax.set_title("Count of Harassment Incidents")
    ax.set_xlabel("Harassment Type")
    ax.set_ylabel("Count")

    # Display the plot in Streamlit
    st.pyplot(fig)

elif selected_page == "Explainable AI":
    st.write("### Explainable AI")
    st.write("Use the AI explainer to understand the model's predictions.")
    # Get the input text from the user
    input_text = st.text_area("Enter a description", "")

    if st.button("Explain"):
        if input_text:
            # Create a LIME explainer
            class_names = ['Label 1', 'Label 2', 'Label 3']  # Replace with your actual class names
            explainer = LimeTextExplainer(class_names=class_names)

            # Get the model's prediction
            prediction = get_prediction(input_text)

            # Get the LIME explanation
            explanation = explainer.explain_instance(input_text, get_prediction, num_features=10)

            # Print the explanation
            st.write("LIME Explanation:")
            explanation_list = explanation.as_list()
            for exp in explanation_list:
                st.write(f"{exp[1]}: {exp[0]}")
        else:
            st.write("Please enter a description to explain.")