import tensorflow as tf
import os
from PIL import Image

bad_paths = []

path = "./dataset/validation/Dog"


def fix_corrupt_image(image_path):
    try:
        img = Image.open(image_path)
        img.load()  # Load the image data
        img.save(image_path)  # Save it back to fix potential corruption
        print(f"Fixed image: {image_path}")
    except Exception as e:
        print(f"Could not fix image: {image_path}. Error: {e}")


def fix_images(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".jpg") or file.endswith(".jpeg"):
                image_path = os.path.join(root, file)
                fix_corrupt_image(image_path)


# Use the function on your dataset directory
fix_images(path)

files = os.listdir(path)
for image_path in files:
    file_path = os.path.join(path, image_path)
    print("Checking image {}".format(file_path))
    try:
        img_bytes = tf.io.read_file(file_path)
        decoded_img = tf.io.decode_image(img_bytes)
    except tf.errors.InvalidArgumentError as e:
        print(f"Found bad path {file_path}...{e}")
        bad_paths.append(file_path)

print("BAD PATHS:")
for bad_path in bad_paths:
    print(f"{bad_path}")
    os.remove(bad_path)
