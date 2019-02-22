from engine.segment import CHEFEngine
import numpy as np
import matplotlib.pyplot as plt

data = np.load('fish_ear.npy')

chef = CHEFEngine(blur=1.6, number_threshold=100)

labels = chef.run(data)

plt.imshow(labels.max(-1))
plt.show()
