import tensorflow as tf

# Load CIFAR-100 dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()

# Save some images from the test set for testing your Flask app
num_images_to_download = 10  # Change this number as needed
for i in range(num_images_to_download):
    image = x_test[i]
    label = y_test[i][0]  # Assuming the label is stored as a single integer
    
    # Save the image with its corresponding label
    filename = f'image_{i}_label_{label}.png'
    tf.keras.preprocessing.image.save_img(filename, image)

print(f"{num_images_to_download} images downloaded and saved for testing.")