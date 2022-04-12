# Load and plot the prepared dataset

# Imports
from numpy import load
from matplotlib import pyplot

# Load the dataset
data = load("horse2zebra_256.npz")
dataA, dataB = data["arr_0"], data["arr_1"]
print("Loaded: ", dataA.shape, dataB.shape)

# Plot source images
n_samples = 5

for i in range(n_samples):
    pyplot.subplot(2, n_samples, 1 + i)
    pyplot.axis("off")
    pyplot.imshow(dataA[i].astype("uint8"))

# Plot target image
for i in range(n_samples):
    pyplot.subplot(2, n_samples, 1 + n_samples + i)
    pyplot.axis("off")
    pyplot.imshow(dataB[i].astype("uint8"))

pyplot.show()