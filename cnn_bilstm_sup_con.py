import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.metrics import classification_report,confusion_matrix
import seaborn as sns
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pandas as pd
from sklearn.metrics import accuracy_score
os.environ["CUDA_VISIBLE_DEVICES"]="1"

class SupervisedContrastiveLoss(keras.losses.Loss):
    def __init__(self, temperature=1, name=None):
        super(SupervisedContrastiveLoss, self).__init__(name=name)
        self.temperature = temperature

    def __call__(self, labels, feature_vectors, sample_weight=None):
        # Normalize feature vectors
        feature_vectors_normalized = tf.math.l2_normalize(feature_vectors, axis=1)
        # Compute logits
        logits = tf.divide(tf.matmul(feature_vectors_normalized, tf.transpose(feature_vectors_normalized)),self.temperature)
        return tfa.losses.npairs_loss(tf.squeeze(labels), logits)


def create_encoder():
    
    inputs = keras.Input(shape=(155,6))
    x1 = layers.Conv1D(100, kernel_size=10,strides=1,activation='relu')(inputs)
    x1 = layers.Conv1D(100, kernel_size=10,strides=1,activation='relu')(x1)
    x1 = layers.MaxPooling1D(pool_size=3, strides=3, padding="valid")(x1)
    outputs = layers.Bidirectional(layers.LSTM(256,activation='tanh'))(x1)
    model = keras.Model(inputs=inputs, outputs=outputs, name="encoder")
    return model

def add_projection_head(encoder):
    inputs = keras.Input(shape=(155,6))
    features = encoder(inputs)
    outputs = layers.Dense(128,activation="relu")(features)
    model = keras.Model(inputs=inputs, outputs=outputs, name="encoder_with_projection-head")
    return model
    
    
def create_classifier(encoder, trainable=True):

    for layer in encoder.layers:
        layer.trainable = trainable

    inputs = keras.Input(shape=(155,6))
    features = encoder(inputs)
    features = layers.Dropout(0.5)(features)
    outputs = layers.Dense(26, activation="softmax")(features)

    model = keras.Model(inputs=inputs, outputs=outputs, name="classifier")
    model.compile(optimizer=keras.optimizers.Adam(learning_rate),loss=keras.losses.SparseCategoricalCrossentropy(),metrics=[keras.metrics.SparseCategoricalAccuracy()])
    return model


datafolder = 'Subject_wise_data3'
accuracy_vec = []
index_vec = []
test_arr = np.zeros((1,1))
pred_arr = np.zeros((1,1))

learning_rate = 0.001
batch_size = 32
num_epochs = 50
temperature = 0.05

X_tsne = np.zeros((1,2))
Y_tsne = np.zeros((1,))


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
    #print(X_train.shape)
    
    X_train_final, X_val, Y_train_final, Y_val = train_test_split(X_train, Y_train, test_size=0.2, random_state=1)


    encoder = create_encoder()
    #print(encoder.summary())
    encoder_with_projection_head = add_projection_head(encoder)
    encoder_with_projection_head.compile(optimizer=keras.optimizers.Adam(learning_rate),loss=SupervisedContrastiveLoss(temperature))
    #print(encoder_with_projection_head.summary())
    es_encoder = keras.callbacks.EarlyStopping(monitor='val_loss', verbose=0, patience=5)
    history_encoder = encoder_with_projection_head.fit(x=X_train_final, y=Y_train_final,validation_data=(X_val,Y_val), batch_size=batch_size, epochs=num_epochs,callbacks=[es_encoder],verbose=0)
    
      
    
    classifier = create_classifier(encoder, trainable=False)
    #print(classifier.summary())
    es_classifier = keras.callbacks.EarlyStopping(monitor='val_sparse_categorical_accuracy', verbose=0, patience=5)
    history_classifier = classifier.fit(x=X_train_final, y=Y_train_final,validation_data=(X_val,Y_val), batch_size=batch_size, epochs=num_epochs,callbacks=[es_classifier],verbose=0)
    
    
    
    accuracy = classifier.evaluate(X_test, Y_test)[1]
    Y_pred = classifier.predict(X_test)
    index_vec.append(i)
    accuracy_vec.append(accuracy)
    test_arr = np.vstack((test_arr,np.reshape(Y_test,(390,1))))
    pred_arr = np.vstack((pred_arr,np.reshape(np.argmax(Y_pred,axis=1),(390,1))))
    tf.keras.backend.clear_session()


test_arr = test_arr[1:,0]
pred_arr = pred_arr[1:,0]

print(accuracy_score(test_arr,pred_arr))

accuracy_vec = np.asarray(accuracy_vec)

print(np.mean(accuracy_vec))



df = pd.DataFrame()
df['Subject'] = index_vec
df['Accuracy'] = accuracy_vec

df.to_csv('results/final_results/Accuracies_cnnbilstm_sup_con.csv',index=False)
np.save('results/final_results/pred_arr_cnnbilstm_sup_con.npy',pred_arr)
np.save('results/final_results/test_arr_cnnbilstm_sup_con.npy',test_arr)


