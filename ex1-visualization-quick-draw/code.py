import numpy as np
import matplotlib.pyplot as plt

test_image = np.load('../all-data/full_numpy_bitmap_book.npy').astype(np.float32)[0]

list_item = ['airplane', 'apple', 'book']
scores = []
weights = []

for item in list_item:
    images = np.load(f'../all-data/full_numpy_bitmap_{item}.npy').astype(np.float32)
    avg_image = np.mean(images, axis=0)
    weights.append(avg_image)
    scores.append(avg_image @ test_image)

print(scores)
print(f'the test_image is most likely {list_item[np.argmax(scores)]}')

plt.figure(figsize=(10, 5))
for i in range(len(weights)):
    plt.subplot(1, len(weights), i + 1)
    plt.imshow(weights[i].reshape(28, 28))
    plt.axis('off')
    plt.title(list_item[i])
plt.show()