# image-classification
Dogs and cats classification

# Dataset used
https://www.kaggle.com/datasets/shaunthesheep/microsoft-catsvsdogs-dataset

# Instructions
1. Download the dataset into your project work directory `./dataset`
2. The structure of dataset has to be:
   - `./dataset/Dog`
   - `./dataset/Cat`
3. Run `./new_model.py`
4. If you encounter errors then you may have to remove the corrupted files from dataset
   - Run ./remove_corrupted_dataset.py for each subdirectory of ./dataset
   - To choose another directore, update the `path = "./dataset/validation/Dog"` in order to point to the directory you want to clean
