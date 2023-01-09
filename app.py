import tensorflow as tf
import streamlit as st
import numpy as np

# Load the model
from matplotlib import pyplot as plt

model = tf.keras.models.load_model('model.h5')

# Map class labels to class names
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


# Function to plot the images and predictions
def plot_predictions(i, predictions_array, true_labels, images):
    predictions_array, true_label, img = predictions_array[i], true_labels[i], images[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    # Reshape the image data to (28, 28)
    img = img.reshape((28, 28))

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                         100 * np.max(predictions_array),
                                         class_names[true_label]),
               color=color)


# Function to plot the prediction array
def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array[i], true_label[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')


# Create the main app
def main():
    st.title("Fashion MNIST Image Classification")
    st.markdown(
        "This app allows you to test the accuracy of a trained model on the Fashion MNIST dataset. Select a test image by using the sidebar slider to see the model's prediction.")

    # Load the test dataset
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

    # Get the index of the image to display
    i = st.sidebar.slider("Select an image index:", 0, len(x_test) - 1, 0, 1)

    # Make the prediction
    predictions = model.predict(x_test[i:i + 1])

    # Plot the image and prediction
    st.subheader("Prediction")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 3))
    ax1.imshow(x_test[i].reshape((28, 28)), cmap=plt.cm.binary)
    ax2.bar(range(10), predictions[0])
    ax2.set_ylim([0, 1])
    ax2.set_xticks(range(10))
    ax2.set_xticklabels(class_names, rotation=90)
    st.pyplot(fig)

    # Display the image
    st.subheader("Image")
    st.image(x_test[i], caption=class_names[y_test[i]])


if __name__ == "__main__":
  main()

