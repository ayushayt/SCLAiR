import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.utils import plot_model, to_categorical
from tensorflow.keras.layers import Dense, Activation, Conv1D, MaxPooling1D, GlobalAveragePooling1D, Flatten, Dropout, BatchNormalization, Input,UpSampling1D,LSTM
from tensorflow.keras.layers import Concatenate, Lambda,Average
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler,StandardScaler 
from sklearn.svm import SVC
import numpy as np
import random
import scipy.io
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from scipy.interpolate import interp1d
import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"]="1"

datafolder = 'Subject_wise_data3'
accuracy_vec = []
index_vec = []
test_arr = np.zeros((1,1))
pred_arr = np.zeros((1,1))


for i in range(0,55):
    X_train = np.zeros((1,155,6))
    Y_train = np.zeros((1,1))

    for fi in os.listdir(datafolder):
        if fi.endswith(".npy") == True and fi[0]=='X':
            xy,nn =  fi.split('_')
            #print(xy)
            n,_  = nn.split('.')
            #print(n)
            #print(os.path.join(datafolder,fi))

            if n != str(i):
                X_train = np.vstack((X_train,np.load(os.path.join(datafolder,'X_'+nn))[:,0:155,:]))
                Y_train = np.vstack((Y_train,np.load(os.path.join(datafolder,'Y_'+nn))))
            else:
                X_test = np.load(os.path.join(datafolder,'X_'+nn))[:,0:155,:]
                Y_test = np.load(os.path.join(datafolder,'Y_'+nn))
                Y_test = np.reshape(Y_test,(390,))
                print(os.path.join(datafolder,fi))

    X_train = X_train[1:,:,:]
    Y_train = Y_train[1:,:]
    Y_train = np.reshape(Y_train,(len(Y_train),))
    
    
    
    
    i1 = Input(shape=(X_train.shape[1],X_train.shape[2]))
    x1 = LSTM(256,activation='tanh')(i1)
    x1 = Dropout(0.5)(x1)
    output = Dense(26, activation='softmax')(x1)

    
    model = Model(inputs=i1, outputs=output)# summarize layers

    #print(model.summary())
    #plot_model(model, to_file='forked_rnn.png')
    
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    es = EarlyStopping(monitor='val_accuracy', verbose=0, patience=5)
    model.fit(X_train, y=to_categorical(Y_train),validation_split=0.2,epochs=50, batch_size=32,verbose=0,callbacks=[es])
    
    predictions = model.predict(X_test)
    Y_pred = np.argmax(np.asarray(predictions),axis=1)
    
    acc = accuracy_score(Y_test,Y_pred)
    print(acc)
    accuracy_vec.append(acc)
    index_vec.append(i)
    tensorflow.keras.backend.clear_session()
    test_arr = np.vstack((test_arr,np.reshape(Y_test,(390,1))))
    pred_arr = np.vstack((pred_arr,np.reshape(Y_pred,(390,1))))

accuracy_vec = np.asarray(accuracy_vec)
index_vec = np.asarray(index_vec)

print(np.mean(accuracy_vec))


df = pd.DataFrame()
df['Subject'] = index_vec
df['Accuracy'] = accuracy_vec
df.to_csv('results/final_results/Accuracies_lstm.csv',index=False)
np.save('results/final_results/pred_arr_lstm.npy',pred_arr)
np.save('results/final_results/test_arr_lstm.npy',test_arr)

