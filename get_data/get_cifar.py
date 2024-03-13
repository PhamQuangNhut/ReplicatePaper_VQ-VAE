import os
import shutil
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToPILImage
from torchvision import transforms

# Define paths for train and test folders
train_dir = 'cifar10_data/train'
test_dir = 'cifar10_data/test'

# Function to create required directories and avoid FileExistsError
def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

# Create train and test directories
create_dir(train_dir)
create_dir(test_dir)

# Function to save images to respective folders
def save_images(dataset, directory):
    transform = ToPILImage()  # Transform to convert tensor images to PIL images

    for i, (image, label) in enumerate(dataset):
        # Create label directories
        label_dir = os.path.join(directory, str(label))
        create_dir(label_dir)

        # Convert tensor image to PIL image
        pil_image = transform(image)

        # Save image
        pil_image.save(os.path.join(label_dir, f'{i}.jpg'))

# Download and load the CIFAR-10 training and test datasets
train_dataset = CIFAR10(root='cifar10_data/', train=True, download=True, transform=transforms.ToTensor())
test_dataset = CIFAR10(root='cifar10_data/', train=False, download=True, transform=transforms.ToTensor())

# Save training and test images
save_images(train_dataset, train_dir)
save_images(test_dataset, test_dir)

print('CIFAR-10 images have been saved to train and test folders.')
