import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # desabilitando mensagem de debug tensorflow

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler #escalar as features para os pesos serem comparaveis
from sklearn.datasets import load_breast_cancer

import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy as np

url = 'https://raw.githubusercontent.com/lazyprogrammer/machine_learning_examples/master/tf2.0/moore.csv'
data = pd.read_csv(url, header=None).to_numpy()
X = data[:,0].reshape(-1, 1) #deixando no formato N x D (n_linhas x n_caracteristicas)
y = data[:,1]
print(f'X shape: {X.shape}, y shape: {y.shape}')

plt.scatter(X,y) #função é exponencial
plt.show()

y = np.log(y) # transformando a função em linear

plt.scatter(X,y) #função é exponencial y_original = 10 ^ np.log(y_original)
plt.show()

X = X - X.mean() #escalando a coluna de anos

print(type(X), X)

N, D = X.shape

model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(D,)),
    tf.keras.layers.Dense(1, activation=None)
])

model.compile(
    #escolhido no lugar do adam pois esse performou melhor nesse caso
    optimizer = tf.keras.optimizers.SGD(0.001, 0.9),
    #como a saida não é binária (mse = mean square erro)
    loss = 'mse' 
)

def schedule(epoch, lr):
    if epoch >= 50:
        return 0.0001
    return 0.001

scheduler = tf.keras.callbacks.LearningRateScheduler(schedule)

r = model.fit(X, y, epochs=60, callbacks=[scheduler])

#plot the loss
plt.plot(r.history['loss'], label='loss')
plt.legend()
plt.show()

weights = model.layers[0].get_weights()
a = weights[0][0][0]
b = weights[1][0]
print(f"eq: y = {a} * x + {b}")