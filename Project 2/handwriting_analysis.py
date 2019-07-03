#import tensorflow
import tensorflow as tf

#import dataset mnist
mnist = tf.keras.datasets.mnist

#remember load_data splits into two tuples
(training_images, training_labels), (testing_images, testing_labels) = mnist.load_data()

training_images = training_images / 255.0
testing_images = testing_images / 255.0

# define callback
class myCallback(tf.keras.callbacks.Callback):
	def on_epoch_end(self, epochs, logs = {}):
		if logs.get('acc') >= 0.99:
			print('\nReached 99% accuracy so cancelling training!\n')
			self.model.stop_training = True

#define the model
callbacks = myCallback()

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

#model.compile
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['acc'])

# model.fit
model.fit(training_images, training_labels, epochs=10, callbacks=[callbacks])
