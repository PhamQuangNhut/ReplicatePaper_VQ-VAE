from pathlib import Path
from torchvision import transforms
batch_size = 128
num_workers = 4
input_size = 64
hidden_units = 256
latent_size = 16
emb_size = 256
beta = 0.2
lr = 2e-4

train_dir = Path('cifar10_data/train')
test_dir = Path('cifar10_data/test')
accelerator = 'gpu'
device = 1
transform = transforms.Compose([transforms.Resize((32, 32)),
    # transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    ])
min_epochs = 1
max_epochs = 600