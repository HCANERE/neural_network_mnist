import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from  neural_network import neural_networksel


(x_train,y_train),(x_test,y_test)=mnist.load_data()

x_train = x_train.reshape(-1, 784).astype("float32") / 255
x_test = x_test.reshape(-1, 784).astype("float32") / 255   # 0 1 değerlerine getir
y_train = to_categorical(y_train, 10)  
y_test = to_categorical(y_test, 10)

model = neural_networksel(784, 128, 10)

def accuaracy(y_predict,y_true):

    max_predict=np.argmax(y_predict,axis=1)

    real_data=np.argmax(y_true,axis=1)

    return np.sum(max_predict == real_data) / len(real_data)


epoch=50
batch_size = 64
learning_rate=0.1

for epoch in range(epoch):

    permutation = np.random.permutation(x_train.shape[0])
    x_train_shuffled = x_train[permutation]
    y_train_shuffled = y_train[permutation]

    for i in range(0, x_train.shape[0], batch_size):

        x_batch = x_train_shuffled[i : i + batch_size]
        y_batch = y_train_shuffled[i : i + batch_size]

        output = model.forward_pass(x_batch)
        dw1, dw2, db1, db2 = model.backward_pass(x_batch, y_batch)
        model.update_weight(dw1, dw2, db1, db2, learning_rate)
    

    final_output = model.forward_pass(x_train)
    loss = model.compute_loss(y_train, final_output)
    acc = accuaracy(y_train, final_output)
    print(f"Epoch {epoch} | Loss: {loss:.4f} | Accuracy: {acc:.4f}")



