import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import shap
import numpy as np

# Disable the PyplotGlobalUseWarning
st.set_option('deprecation.showPyplotGlobalUse', False)

# Load the tokenizer
@st.cache_resource
def load_tokenizer():
    return Tokenizer()

tokenizer = load_tokenizer()

# Load the pre-trained model
@st.cache_resource
def load_pretrained_model():
    return load_model('keras model.h5')  

model = load_pretrained_model()

# Maximum sequence length
maxlen = 100

# Human-readable labels
labels = ['Commenting', 'Ogling/Facial Expressions/Staring', 'Touching /Groping']

# Streamlit app
st.title("Harassment Detection")
st.write("Enter a description to predict harassment types.")

# Sidebar navigation
selected_page = st.sidebar.radio("Navigation", ["Incidents", "Safety Tips", "Reports", "Community"])

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

# Page content based on selected navigation
if selected_page == "Incidents":
    st.write("This is the Incidents page.")
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

     # Load dev.csv dataset
    dev_data = pd.read_csv("dev.csv")

    # Human-readable labels
    labels = ['Commenting', 'Ogling/Facial Expressions/Staring', 'Touching /Groping']

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
    st.write("Use the SHAP explainer to understand the model's predictions.")

    # Get the input text from the user
    input_text = st.text_area("Enter a description", "")

    if st.button("Explain"):
        if input_text:
            # Tokenize the input text
            input_sequence = tokenizer.texts_to_sequences([input_text])
            padded_input = pad_sequences(input_sequence, maxlen=maxlen)

            # Create a background dataset for SHAP
            background_texts = dev_data['description'].tolist()  # Assuming 'description' is a column in your dev.csv
            background_sequences = tokenizer.texts_to_sequences(background_texts)
            padded_background = pad_sequences(background_sequences, maxlen=maxlen)

            # Create the SHAP explainer
            explainer = shap.DeepExplainer(model, data=padded_background)

            # Get the SHAP values
            shap_values = explainer.shap_values(padded_input)

            # Plot the SHAP force plot
            st.write("SHAP Force Plot:")
            shap.force_plot(explainer.expected_value[0], shap_values[0], padded_input[0])
            st.pyplot()
        else:
            st.write("Please enter a description to explain.")
            
elif selected_page == "Community":
    st.write("This is the Community page.")