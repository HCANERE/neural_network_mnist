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
learning_rate=0.1

for epoch in range(epoch):

    output=model.forward_pass(x_train) #ileri besleme ile alınan ilk çıkış

    loss=model.compute_loss(y_train,output)

    dw1,dw2,db1,db2=model.backward_pass(x_train,y_train)

    model.update_weight(dw1, dw2, db1, db2, learning_rate)

    if epoch % 5 == 0:
        acc = accuaracy(y_train, output)
        print(f"Epoch {epoch} | Loss: {loss:.4f} | Accuracy: {acc:.4f}")



