
def download_file(url, filename):
    """Download a file from a URL to the given filename."""
    response = requests.get(url)
    with open(filename, 'wb') as f:
        f.write(response.content)

def read_idx(filename):
    """Read an IDX file and return it as a numpy array."""
    with open(filename, 'rb') as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(shape)

def save_images(images, labels, directory):
    """Save images to the specified directory with labels as filenames."""
    os.makedirs(directory, exist_ok=True)
    for i, (image, label) in enumerate(zip(images, labels)):
        image_path = os.path.join(directory, f"{i}_{label}.png")
        Image.fromarray(image).save(image_path)

# URLs for the MNIST dataset
urls = {
    "train_images": "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz",
    "train_labels": "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz",
    "test_images": "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz",
    "test_labels": "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz"
}

# Download and unzip MNIST dataset
for name, url in urls.items():
    filename = url.split("/")[-1]
    filepath = os.path.join("mnist_data", filename)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    if not os.path.exists(filepath[:-3]):  # Check if the unzipped file exists
        print(f"Downloading {filename}...")
        download_file(url, filepath)
        print(f"Extracting {filename}...")
        os.system(f"gunzip -k {filepath}")

# Read IDX files and convert to numpy arrays
train_images = read_idx("mnist_data/train-images-idx3-ubyte")
train_labels = read_idx("mnist_data/train-labels-idx1-ubyte")
test_images = read_idx("mnist_data/t10k-images-idx3-ubyte")
test_labels = read_idx("mnist_data/t10k-labels-idx1-ubyte")

# Save images to folders
save_images(train_images, train_labels, "mnist_data/train_images")
save_images(test_images, test_labels, "mnist_data/test_images")

print("MNIST dataset has been processed and saved as images.")
