import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# number of data points
N = 1000

# uniformly distributed X=(x_1, x_2) data from [-3,-3] to [3,3]
X = np.random.random((N, 2)) * 6 - 3

# example of target function: y = cos(2 x_1) + cos(3 x_2)
y = np.cos(2*X[:,0]) + np.cos(3*X[:,1])

# plot it (we can do it on a separate python script in order to play with the plot)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:,0], X[:,1], y)
plt.show()

# build model
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1)
])

# compile model with custom optimizer (check documentation)
opt = tf.keras.optimizers.Adam(0.01)
model.compile(optimizer=opt, loss='mse')

# fit model
r = model.fit(X, y, epochs=100)

# plot training
plt.plot(r.history['loss'])
plt.title('Training loss')

# plot prediction surface
points = np.linspace(-3, 3, 50)
xx, yy = np.meshgrid(points, points)
X_grid = np.vstack((xx.flatten(), yy.flatten())).T
y_hat = model.predict(X_grid).flatten()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:,0], X[:,1], y)
ax.plot_trisurf(X_grid[:,0], X_grid[:,1], y_hat, linewidth=0.2, antialiased=True)
plt.show()