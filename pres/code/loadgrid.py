# (1)
self.labels = mnist_loader.load_idx_data(mnist_labels_path)

# (2)
num_examples = np.prod(size)
i = 0
labels_i = []
while len(labels_i) < num_examples:
    if self.labels[i] in digits:
        labels_i.append(i)
    i += 1

# (3)
self.images = mnist_loader.load_idx_data(mnist_images_path, pos=labels_i)

# (4)
self.labels = self.labels[labels_i].reshape(size[::-1] + self.labels.shape[1:])
self.images = self.images.reshape(size[::-1] + self.images.shape[1:])
