import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

(x_train,y_train) , (x_test,y_test)=mnist.load_data()  #Verileri çek
x_train = x_train.astype("float32") / 255 # 0 1 arasına sıkıştır
x_test = x_test.astype("float32") / 255
x_train = x_train.reshape(-1, 784) # 28*28 = 784
x_test = x_test.reshape(-1, 784)
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

class neural_network:
    
    def __init__(self,input_size,hidden_size,output_size):
        
        self.w1=np.random.randn(input_size,hidden_size)*0.01 #AĞIRLIK 1
        self.w2=np.random.randn(hidden_size,output_size)*0.01 #AĞIRLIK 2
        self.b1=np.zeros((1,hidden_size)) #Bias 1 
        self.b2=np.zeros((1,output_size)) # bias 2
        pass

    def relU(self,x):       #ReLU fonksiyonum
        return np.maximum(0,x)
    
    def sigmoid(self,z):    #Sigmoid fonk
        return 1 / (1+np.exp(-z))
    
    def sigmoid_derivative(self,z): #Sigmoidin türevi

        s= self.sigmoid(z)
        return s *(1-s)
    

    def forward_pass(self,input_data):  #FORWARD PASS İŞLEMLERİ 
         
        self.linear_procces1=np.dot(input_data,self.w1)+self.b1 # LİNEER İŞLEM  ara katman

        self.act_procces=self.relU(self.linear_procces1) #Aktivasyon fonksiyonu kullanımı ara katman 

        self.linear_procces2=np.dot(self.act_procces,self.w2)+self.b2 #Çıkış katmanı linear 2

        self.output=self.sigmoid(self.linear_procces2) #Çıkış katmanı

        return self.output
    
    def compute_loss(self,y_real,y_predict):

        batch_size=y_real.shape[0] #Tüm örnek veri seti boyutu

        y_predict=np.clip(y_predict,1e-15, 1 - 1e-15) #Tahminleri sayısal kararlılık için 2 sayı arasına sıkıştır

        loss=-1/batch_size*np.sum(y_real*np.log(y_predict)+(1-y_real)*np.log(1-y_predict)) #Binary cross entropy ile loss hesaplamasal

        return loss
    
    def backward_pass(self, input, y_real):

        batch_size=y_real.shape[0]
        dZ2=self.output-y_real       #Çıkış katmanındaki hata
        dW2=(1/batch_size)*np.dot(self.act_procces.T,dZ2) 
        dB2=(1/batch_size)*np.sum(dZ2,axis=0,keepdims=True)
        
        dZ1=np.dot(dZ2, self.w2.T)*(self.linear_procces1>0)

        dW1=(1/batch_size)*np.dot(input.T,dZ1)
        dB1=(1/batch_size)*np.sum(dZ1,axis=0,keepdims=True)

        return dW1,dW2,dB1,dB2

    def update_weight(self,dW1,dW2,dB1,dB2,learning_rate):

            #Katman 2
        self.w2-=learning_rate*dW2
        self.w1-=learning_rate*dW1
            #katman 1 güncellemesi
        self.b2-=learning_rate*dB2
        self.b1-=learning_rate*dB1

        return self.w2,self.w1,self.b1,self.b2

