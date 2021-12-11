import data
import matplotlib.pyplot as plt
import os
fig, axes = plt.subplots()
for i in range(3):
    axes.boxplot(data.train_data, sym='rd', positions=[i])
if not os.path.exists('image'):
    os.makedirs('image')
plt.savefig(os.path.join('image', str(2)))


