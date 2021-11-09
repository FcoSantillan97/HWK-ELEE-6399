#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from d2l import tensorflow as d2l
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
import numpy as np
from keras.datasets import mnist

def net():
    return tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(filters=16, kernel_size=3, activation='relu',
                               padding='same'),
        tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu',
                               padding='same'),
        tf.keras.layers.MaxPool2D(pool_size=2, strides=2),
        tf.keras.layers.Conv2D(filters=16, kernel_size=3,
                               activation='relu'),
        tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu',
                               padding='same'),
        tf.keras.layers.MaxPool2D(pool_size=2, strides=2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(100, activation='sigmoid'),
        tf.keras.layers.Dense(50, activation='sigmoid'),
        tf.keras.layers.Dense(25, activation='sigmoid'),
        tf.keras.layers.Dense(10)])


# In[ ]:


X = tf.random.uniform((1, 28, 28, 1))
for layer in net().layers:
    X = layer(X)
    print(layer.__class__.__name__, 'output shape: \t', X.shape)
    


# In[ ]:


(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = np.reshape(x_train, [60000,28,28,1])
x_test = np.reshape(x_test, [10000,28,28,1])
x_train = x_train/255.0
x_test = x_test/255.0

batch_size = 10
train_iter = tf.data.Dataset.from_tensor_slices((x_train,y_train)).batch(batch_size)
test_iter = tf.data.Dataset.from_tensor_slices((x_test,y_test)).batch(batch_size)


# In[ ]:


class TrainCallback(tf.keras.callbacks.Callback):  #@save
    """A callback to visiualize the training progress."""
    def __init__(self, net, train_iter, test_iter, num_epochs, device_name):
        self.timer = d2l.Timer()
        self.animator = d2l.Animator(
            xlabel='epoch', xlim=[1, num_epochs],
            legend=['train loss', 'train acc', 'test acc'])
        self.net = net
        self.train_iter = train_iter
        self.test_iter = test_iter
        self.num_epochs = num_epochs
        self.device_name = device_name

    def on_epoch_begin(self, epoch, logs=None):
        self.timer.start()

    def on_epoch_end(self, epoch, logs):
        self.timer.stop()
        test_acc = self.net.evaluate(self.test_iter, verbose=0,
                                     return_dict=True)['accuracy']
        metrics = (logs['loss'], logs['accuracy'], test_acc)
        self.animator.add(epoch + 1, metrics)
        if epoch == self.num_epochs - 1:
            batch_size = next(iter(self.train_iter))[0].shape[0]
            num_examples = batch_size * tf.data.experimental.cardinality(
                self.train_iter).numpy()
            print(f'loss {metrics[0]:.3f}, train acc {metrics[1]:.3f}, '
                  f'test acc {metrics[2]:.3f}')
            print(f'{num_examples / self.timer.avg():.1f} examples/sec on '
                  f'{str(self.device_name)}')

#@save
def train_ch6(net_fn, train_iter, test_iter, num_epochs, lr, device):
    """Train a model with a GPU (defined in Chapter 6)."""
    device_name = device._device_name
    strategy = tf.distribute.OneDeviceStrategy(device_name)
    with strategy.scope():
        optimizer = tf.keras.optimizers.SGD(learning_rate=lr)
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        net = net_fn()
        net.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    callback = TrainCallback(net, train_iter, test_iter, num_epochs,
                             device_name)
    net.fit(train_iter, epochs=num_epochs, verbose=0, callbacks=[callback])
    return net


# In[ ]:


lr, num_epochs = 0.05, 10
train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())

