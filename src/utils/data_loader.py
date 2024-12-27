import os
import numpy as np
from PIL import Image
from typing import Tuple, Iterator
import glob
from sklearn.model_selection import train_test_split

class DataLoader:
    def __init__(
            self,
            data_dir,
            image_size: Tuple[int, int] = (32, 32),
            batch_size: int = 32,
            val_split: float = 0.2,
            shuffle: bool = True
    ):
        self.data_dir = data_dir
        self.image_size = image_size
        self.batch_size = batch_size
        self.val_split = val_split
        self.shuffle = shuffle

        self.load_data()

    def load_data(self):
        """Load data and create train and validation splits"""
        images = []
        labels = []
        class_to_idx = {}

        class_dirs = [d for d in os.listdir(self.data_dir)
                      if os.path.isdir(os.path.join(self.data_dir, d))]
        
        print(f"Found class directories: {class_dirs}")

        # Create class mapping
        for idx, class_name in enumerate(sorted(class_dirs)):
            class_to_idx[class_name] = idx

        print(f"Class to index mapping: {class_to_idx}")

        # Load images from each class
        for class_name in class_dirs:
            class_path = os.path.join(self.data_dir, class_name)
            class_idx = class_to_idx[class_name]

            image_files = glob.glob(os.path.join(class_path, '*.jpeg'))
            print(f"Found {len(image_files)} images in {class_name}")

            for img_path in image_files:
                try:
                    img_data = self.preprocess_image(img_path)
                    images.append(img_data)
                    labels.append(class_idx)
                except Exception as e:
                    print(f"Error loading {img_path}: {str(e)}")
        
        X = np.array(images)
        y = np.array(labels)

        # Create train/validation split
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X, y, test_size=self.val_split, shuffle=self.shuffle
        )

        print(f"Training samples: {len(self.X_train)}")
        print(f"Validation samples: {len(self.X_val)}")

    def preprocess_image(self, image_path):
        """Load and preprocess a single image
        
        returns: preprocessed images array in channel-first format (C, H, W)
        """

        with Image.open(image_path) as img:
            if img.mode != 'RGB':
                img = img.convert('RGB')

            img = img.resize(self.image_size, Image.BILINEAR)  

            img_array = np.array(img)

            img_array = img_array.astype(np.float32) / 255.0

            # Convert to channel-first format (C, H, W)
            img_array = np.transpose(img_array, (2, 0, 1))

            return img_array
        
    def get_batches(self, train):
        X = self.X_train if train else self.X_val
        y = self.y_train if train else self.y_val

        indices = np.arange(len(X))
        if self.shuffle and train:
            np.random.shuffle(indices)

        for start_idx in range(0, len(X), self.batch_size):
            batch_indices = indices[start_idx:start_idx + self.batch_size]
            batch_X = X[batch_indices]
            batch_y = y[batch_indices]

            yield batch_X, batch_y
    
    def get_validation_data(self):
        return self.X_val, self.y_val
    

if __name__ == "__main__":
    data_loader = DataLoader(
        data_dir="/home/mouli/Desktop/codes/cnn/data/train",
        image_size=(32, 32),
        batch_size=32
    )

    for batch_X, batch_y in data_loader.get_batches(train=True):
        print(f"Batch shape: {batch_X.shape}")
        print(f"Labels shape: {batch_y.shape}")
        break

    X_val, y_val = data_loader.get_validation_data()
    print(f"Validation data shape: {X_val.shape}")
    print(f"Validation labels shape: {y_val.shape}")
