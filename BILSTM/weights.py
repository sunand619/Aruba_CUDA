import multiprocessing
import numpy as np
import csv
import matplotlib as pyplot
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, Input
from keras.layers import Bidirectional
from keras.layers import GRU
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.utils import to_categorical
model=Sequential()
from keras.models import load_model
model =  load_model('BILSTM_model_aruba_70.h5')
print(model.summary())
for layer in model.layers:
    weights = layer.get_weights()
print(weights)
print(len(model.layers))
for _ in weights:
  list1=_
  break
for _ in weights:
  list2=_  
print(np.shape(list1))
print(np.shape(list2))
#for e in zip(model.layers[0].trainable_weights, model.layers[0].get_weights()):
#    print('Param %s:\n%s' % (e[0],e[1]))

units = int(int(model.layers[0].trainable_weights[0].shape[1])/4)
print("No units: ", units)

W = model.layers[0].get_weights()[0]
U = model.layers[0].get_weights()[1]
b = model.layers[0].get_weights()[2]

W_i = W[:, :units]
W_f = W[:, units: units * 2]
W_c = W[:, units * 2: units * 3]
W_o = W[:, units * 3:]

W_i=np.transpose(W_i)
W_f=np.transpose(W_f)
W_c=np.transpose(W_c)
W_o=np.transpose(W_o)

U_i = U[:, :units]
U_f = U[:, units: units * 2]
U_c = U[:, units * 2: units * 3]
U_o = U[:, units * 3:]

b_i = b[:units]
b_f = b[units: units * 2]
b_c = b[units * 2: units * 3]
b_o = b[units * 3:]
print("W:i,f,c,o")
print(np.shape(W_i))
print(np.shape(W_f))
print(np.shape(W_c))
print(np.shape(W_o))
print("U:i,f,c,o")
print(np.shape(U_i))
print(np.shape(U_f))
print(np.shape(U_c))
print(np.shape(U_o))
print("b:i,f,c,o")
print(np.shape(b_i))
print(np.shape(b_f))
print(np.shape(b_c))
print(np.shape(b_o))
np.savetxt('weights/W_i_bilstm_70.txt', W_i, delimiter=',')
np.savetxt('weights/W_f_bilstm_70.txt', W_f, delimiter=',')
np.savetxt('weights/W_c_bilstm_70.txt', W_c, delimiter=',')
np.savetxt('weights/W_o_bilstm_70.txt', W_o, delimiter=',')
np.savetxt('weights/U_i_bilstm_70.txt', U_i, delimiter=',')
np.savetxt('weights/U_f_bilstm_70.txt', U_f, delimiter=',')
np.savetxt('weights/U_c_bilstm_70.txt', U_c, delimiter=',')
np.savetxt('weights/U_o_bilstm_70.txt', U_o, delimiter=',')
np.savetxt('weights/b_i_bilstm_70.txt', b_i, delimiter=',')
np.savetxt('weights/b_f_bilstm_70.txt', b_f, delimiter=',')
np.savetxt('weights/b_c_bilstm_70.txt', b_c, delimiter=',')
np.savetxt('weights/b_o_bilstm_70.txt', b_o, delimiter=',')
with open('weights/dense_bilstm_70.txt', 'w') as file:
    np.savetxt(file,list2,delimiter=',')	
    file.writelines(','.join(str(j) for j in i) + '\n' for i in list1)
