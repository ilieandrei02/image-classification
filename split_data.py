import os
import shutil


# Function to split data
def split_data(dataset_dir, category, validation_percentage):
    # Create new directories
    validation_dir = os.path.join(dataset_dir, 'validation', category)
    test_dir = os.path.join(dataset_dir, 'test', category)

    os.makedirs(validation_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    category_dir = os.path.join(dataset_dir, category)
    if os.path.exists(category_dir):
        files = os.listdir(str(category_dir))

        num_files = len(files)
        num_validation = int(num_files * validation_percentage)

        validation_files = files[:num_validation]
        test_files = files[num_validation:]

        for file in validation_files:
            shutil.move(os.path.join(str(category_dir), file), os.path.join(str(validation_dir), file))

        for file in test_files:
            shutil.move(os.path.join(str(category_dir), file), os.path.join(str(test_dir), file))

        shutil.rmtree(category_dir)


print('Data has been split into validation and test sets.')
