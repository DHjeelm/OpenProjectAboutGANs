
# Code to check if everything looks alright
from numpy import load
from matplotlib import pyplot

# Load the dataset
data = load("maps_256.npz")
src_images, tar_images = data["arr_0"], data["arr_1"]
print("Loaded: ", src_images.shape, tar_images.shape)

n_samples = 3

# Plot source images
for i in range(n_samples):
    pyplot.subplot(2, n_samples, 1 + i)
    pyplot.axis("off")
    pyplot.imshow(src_images[i].astype("uint8"))

# Plot target image
for i in range(n_samples):
    pyplot.subplot(2, n_samples, 1 + n_samples + i)
    pyplot.axis("off")
    pyplot.imshow(tar_images[i].astype("uint8"))
pyplot.show()