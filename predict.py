import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt

image_to_predict = './cat_dog.jpg'
img_height = 200
img_width = 200
model_to_use = 'model_200_200.keras'
class_names = ['Cat', 'Dog']


# Load and test the model with a new image
def predict(image_path, model):
    img = Image.open(image_path).resize((img_height, img_width))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    predictions = model.predict(img_array)
    #print(predictions)

    predicted_class = class_names[np.argmax(predictions)]
    print(f'\n\nI am {max(predictions[0]) * 100:.2f}% sure the image is a {predicted_class}\n\n')


# Load the model
print("Loading model...")
trained_model = tf.keras.models.load_model(model_to_use)

trained_model.summary()
print("")

# Test with a new image
predict(image_to_predict, trained_model)

img = tf.keras.utils.load_img(image_to_predict, target_size=(img_width, img_height))
img_array = tf.keras.utils.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

# Normalize the image
img_array = img_array / 255.0

# Get the feature maps for the input image
_ = trained_model.predict(img_array)

# Create a list of layer outputs
layer_outputs = [layer.output for layer in trained_model.layers]
layer_inputs = [layer.input for layer in trained_model.layers]
print(layer_outputs)
print(layer_inputs)

# Create a model that will return these outputs given the model input
intermediate_model = tf.keras.Model(inputs=layer_inputs[0], outputs=layer_outputs)

# Get the feature maps for the input image
feature_maps = intermediate_model.predict(img_array)


# Visualize the feature maps
def plot_feature_maps(feature_maps, layer_index):
    feature_map = feature_maps[layer_index]
    num_filters = feature_map.shape[-1]

    size = feature_map.shape[1]
    display_grid = np.zeros((size, size * num_filters))

    for i in range(num_filters):
        # Post-process the feature to make it visually interpretable
        feature_image = feature_map[0, :, :, i]
        feature_image -= feature_image.mean()
        feature_image /= feature_image.std()
        feature_image *= 64
        feature_image += 128
        feature_image = np.clip(feature_image, 0, 255).astype('uint8')
        display_grid[:, i * size: (i + 1) * size] = feature_image

    scale = 20. / num_filters
    plt.figure(figsize=(scale * num_filters, scale))
    plt.title(f"Layer {layer_index}: {trained_model.layers[layer_index].name}")
    plt.grid(False)
    plt.imshow(display_grid, aspect='auto', cmap='viridis')


# Plot the feature maps for each convolutional layer
# for i in range(len(trained_model.layers)):
#     if 'conv' in trained_model.layers[i].name:
#         plot_feature_maps(feature_maps, i)


# Run the image through the network, thus obtaining all
# intermediate representations for this image.
successive_feature_maps = intermediate_model.predict(img_array)

# These are the names of the layers, so you can have them as part of the plot
layer_names = [layer.name for layer in trained_model.layers[1:]]

# Display the representations
for layer_name, feature_map in zip(layer_names, successive_feature_maps):
    if len(feature_map.shape) == 4:

        # Just do this for the conv / maxpool layers, not the fully-connected layers
        n_features = feature_map.shape[-1]  # number of features in feature map

        # The feature map has shape (1, size, size, n_features)
        size = feature_map.shape[1]

        # Tile the images in this matrix
        display_grid = np.zeros((size, size * n_features))
        for i in range(n_features):
            x = feature_map[0, :, :, i]

            # Normalize the feature map
            if x.std() != 0:  # Avoid division by zero
                x -= x.mean()
                x /= x.std()
            else:
                x -= x.mean()  # Just zero-center it if std is 0

            # Scale the feature map for better visualization
            x *= 64
            x += 128
            x = np.clip(x, 0, 255).astype('uint8')

            # Tile each filter into this big horizontal grid
            display_grid[:, i * size: (i + 1) * size] = x

        # Display the grid
        scale = 120. / n_features
        plt.figure(figsize=(scale * n_features, scale))
        plt.title(layer_name)
        plt.grid(False)
        plt.imshow(display_grid, aspect='auto', cmap='viridis')

plt.show()
