import matplotlib.pyplot as plt

plt.plot([1, 2, 3, 4], [1, 4, 9, 16], 'ro')
plt.axis((0, 5, 0, 20))
plt.xlabel('numbers')
plt.ylabel('squre of numbers')
plt.show()