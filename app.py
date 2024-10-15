import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data_loader import load_spiral_data
from neural_network import NeuralNetwork
from Visualizer import plot_decision_boundary

# New import for neural network visualization
from matplotlib.patches import Rectangle, Circle, ArrowStyle, FancyArrowPatch

# Load data
@st.cache_data
def load_data():
    return load_spiral_data('data/spiral.csv')

# Train model
@st.cache_resource
def train_model(X, y, n_hidden, learning_rate, reg, epochs):
    model = NeuralNetwork(n_features=2, n_hidden=n_hidden, n_classes=3)
    model.train(X, y, learning_rate=learning_rate, reg=reg, epochs=epochs, verbose=False)
    return model

# New function to visualize the neural network
def plot_neural_network(model):
    fig, ax = plt.subplots(figsize=(12, 8))
    left, right, bottom, top = 0.1, 0.9, 0.1, 0.9
    layer_sizes = [model.n_features, model.n_hidden, model.n_classes]
    
    v_spacing = (top - bottom)/float(max(layer_sizes))
    h_spacing = (right - left)/float(len(layer_sizes) - 1)
    
    # Input layer nodes
    layer_top = v_spacing*(layer_sizes[0] - 1)/2. + (top + bottom)/2.
    for m in range(layer_sizes[0]):
        circle = Circle((left, layer_top - m*v_spacing), v_spacing/4., color='r', ec='k', zorder=4)
        ax.add_artist(circle)
        
    # Hidden layer nodes
    layer_top = v_spacing*(layer_sizes[1] - 1)/2. + (top + bottom)/2.
    for m in range(layer_sizes[1]):
        circle = Circle((left + h_spacing, layer_top - m*v_spacing), v_spacing/4., color='g', ec='k', zorder=4)
        ax.add_artist(circle)
        
    # Output layer nodes
    layer_top = v_spacing*(layer_sizes[2] - 1)/2. + (top + bottom)/2.
    for m in range(layer_sizes[2]):
        circle = Circle((right, layer_top - m*v_spacing), v_spacing/4., color='b', ec='k', zorder=4)
        ax.add_artist(circle)
        
    # Edges
    for i in range(len(layer_sizes) - 1):
        layer_top_1 = v_spacing*(layer_sizes[i] - 1)/2. + (top + bottom)/2.
        layer_top_2 = v_spacing*(layer_sizes[i+1] - 1)/2. + (top + bottom)/2.
        for m in range(layer_sizes[i]):
            for n in range(layer_sizes[i+1]):
                line = FancyArrowPatch((left + i*h_spacing, layer_top_1 - m*v_spacing),
                                       (left + (i+1)*h_spacing, layer_top_2 - n*v_spacing),
                                       color='gray', arrowstyle='->', mutation_scale=10)
                ax.add_artist(line)
    
    ax.axis('off')
    plt.title('Neural Network Architecture')
    return fig

# Streamlit app
def main():
    st.title('SpiralNet Classifier')
    st.write('This app demonstrates a neural network classifier for spiral data.')

    # Load data
    X, y = load_data()
    st.write('Data loaded:', X.shape[0], 'samples')

    # Sidebar for hyperparameters
    st.sidebar.header('Hyperparameters')
    n_hidden = st.sidebar.slider('Number of hidden neurons', 10, 200, 100)
    learning_rate = st.sidebar.slider('Learning rate', 0.001, 1.0, 0.1, format="%.3f")
    reg = st.sidebar.slider('Regularization strength', 0.0001, 0.1, 0.001, format="%.4f")
    epochs = st.sidebar.slider('Number of epochs', 1000, 20000, 10000, step=1000)

    # Train model button
    if st.sidebar.button('Train Model'):
        with st.spinner('Training in progress...'):
            model = train_model(X, y, n_hidden, learning_rate, reg, epochs)
        st.success('Training complete!')

        # Evaluate model
        train_accuracy = np.mean(model.predict(X) == y)
        st.write(f"Training accuracy: {train_accuracy:.2f}")

        # Plot decision boundary
        st.subheader('Decision Boundary')
        fig, ax = plt.subplots(figsize=(10, 6))
        plot_decision_boundary(model, X, y)
        st.pyplot(fig)

        # Plot neural network visualization
        st.subheader('Neural Network Architecture')
        nn_fig = plot_neural_network(model)
        st.pyplot(nn_fig)

    # Add a section for custom predictions
    st.header('Make a custom prediction')
    x1 = st.number_input('Enter x1 value:', value=0.0, step=0.1)
    x2 = st.number_input('Enter x2 value:', value=0.0, step=0.1)

    if st.button('Predict'):
        model = train_model(X, y, n_hidden, learning_rate, reg, epochs)
        custom_X = np.array([[x1, x2]])
        prediction = model.predict(custom_X)
        st.write(f'Predicted class: {prediction[0]}')

if __name__ == "__main__":
    main()