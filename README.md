# Aruba_CUDA

1)final_aruba1.txt : Test Dataset

Main Task: To predict activity change points in Aruba Dataset.

Window size: While predicting a datapoint whether or nor it is an activity change point, in the test dataset, we have considered a fixed number of points (Window size) above and below the point which is to be predicted and compute a dot product of the probabilities of its activities which are obtained after going through Deep Learning network. The dot product is between window size set above the datapoint and the window size set below the datapoint. Our idea is that the value of the dot product if found to low increases the datapoint to be a change point of activity.

2)BILSTM : Contains the CUDA code for data parallelization for various BiLstm models.
    
   BILSTM_model_aruba_10.h5 : Contains the extracted weights and parameters from a trained bilstm network of window size 10.
    
   BILSTM_model_aruba_30.h5 : Contains the extracted weights and parameters from a trained bilstm network of window size 30.
    
   BILSTM_model_aruba_50.h5 : Contains the extracted weights and parameters from a trained bilstm network of window size 50.
    
   BILSTM_model_aruba_70.h5 : Contains the extracted weights and parameters from a trained bilstm network of window size 70.
    
   bilstm_10.cu: Cuda code which performs parallel predictions on the test data ,corresponding to a trained bilstm model of window size 10.(Reads the weights from       BILSTM_model_aruba_10.h5) 
   
   bilstm_30.cu: Cuda code which performs parallel predictions on the test data ,corresponding to a trained bilstm model of window size 30. 
   
   bilstm_50.cu: Cuda code which performs parallel predictions on the test data ,corresponding to a trained bilstm model of window size 50. 
   
   bilstm_70.cu: Cuda code which performs parallel predictions on the test data ,corresponding to a trained bilstm model of window size 70. 
   
   bilstm_10_multigpu.cu: Cuda code which performs parallel predictions on the test data ,corresponding to a trained bilstm model of window size 10, along with         distributing the threads across 4 gpus.
   
   bilstm_30_multigpu.cu: Cuda code which performs parallel predictions on the test data ,corresponding to a trained bilstm model of window size 30, along with         distributing the threads across 4 gpus.
   
   bilstm_50_multigpu.cu: Cuda code which performs parallel predictions on the test data ,corresponding to a trained bilstm model of window size 50, along with         distributing the threads across 4 gpus.
   
   bilstm_70_multigpu.cu: Cuda code which performs parallel predictions on the test data ,corresponding to a trained bilstm model of window size 70, along with         distributing the threads across 4 gpus.
  
 "TO compile the CUDA code ( say bilstm_10.cu), use the command : nvcc bilstm_10.cu -std=c++11  -Xcompiler -fopenmp"
  

3)LSTM : Contains the cuda code for data parallelization of lstm model.
   
   LSTM_model_aruba_50.h5: Contains the extracted weights and parameters from a trained lstm network of window size 50.
   
   lstm_50.cu : Cuda code which performs parallel predictions on the test data ,corresponding to a trained lstm model of window size 50.
   
   "TO compile the CUDA code ( lstm_50.cu), use the command : nvcc lstm_50.cu -std=c++11  -Xcompiler -fopenmp"
  
4)cpd1.py : Python prediction code for the test data, which takes 'aruba_50_bidirectional_lstm_model.h5' ,to read the pre-trained weigths and parameters.

5)aruba_50_bidirectional_lstm_model.h5 : Trained BiLstm network of window size 50.(the h5 files are used to read the weights and various parameters of the trained model.)
