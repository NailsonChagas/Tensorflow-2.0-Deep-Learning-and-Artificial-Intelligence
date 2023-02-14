import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # desabilitando mensagem de debug tensorflow

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler #escalar as features para os pesos serem comparaveis
from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt
import tensorflow as tf

print(tf.__version__)

data = load_breast_cancer()
print(data.keys())
print(f"features shape: {data.data.shape}, target shape: {data.target.shape}")

X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.33)
print(data.keys())
print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train) #faz fit e transform
X_test = scaler.transform(X_test) #como o fit já foi feito em cima n precisa repetir

N, D = X_train.shape

model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(D,)),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam', 
    loss='binary_crossentropy', 
    metrics=['accuracy']
)

r = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100)

print(f"Train Score: {model.evaluate(X_train, y_train)}")
print(f"Test Score: {model.evaluate(X_test, y_test)}")

plt.plot(r.history['loss'], label='loss')
plt.plot(r.history['val_loss'], label='val_loss')
plt.legend()
plt.show()

plt.plot(r.history['accuracy'], label='acc')
plt.plot(r.history['val_accuracy'], label='val_acc')
plt.legend()
plt.show()