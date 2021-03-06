# Load, split and scale the maps dataset ready for training

from os import listdir
from numpy import asarray
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from numpy import savez_compressed


# Load all images in a directory into memory
def load_images(path, size=(256,512)):

    # Create lists
    src_list, tar_list = list(), list()

    # Enumerate filenames in directory
    for filename in listdir(path):

        # Load and resize the image
        pixels = load_img(path + filename, target_size=size)

        # Convert to numpy array
        pixels = img_to_array(pixels)

        # Split into satellite and map
        sat_img, map_img = pixels[:, :256], pixels[:, 256:]
        src_list.append(sat_img)
        tar_list.append(map_img)

    return [asarray(src_list), asarray(tar_list)]

# Dataset path
path = "maps/train/"

# load dataset
[src_images, tar_images] = load_images(path)
print("Loaded: ", src_images.shape, tar_images.shape)

# Save as compressed numpy array
filename = "maps_256.npz"
savez_compressed(filename, src_images, tar_images)
print("Saved dataset: ", filename)