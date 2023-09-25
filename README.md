# Mathematical_simulation

This is the first question's answer accounting to F of the 20th 'huawei' cup Mathematical simulation.

Use U-net to process the rainfall data and related micro_physical value.

According to the first 10 frames of data, we output later 10 frames of data. The shape is (batch_size, 30, 256, 256).
Here 30 means 10 prediction frames of 3 different heights, such as 1km, 3km, 7km.

And the Evaluation method is the loss_function and accuracy. Among two of them, accuracy is defined as the number of 
'pixel' whose difference value with val_data is smaller than the custom variable.

The deep learning network is defined in LSTM.py. In the 'train.py', we can change the value of 'load' to decide whether
load the former .pth file or train a new one. 'utils.py' includes the data source, such as train_data and test_data, 
you can see it in the function 'TrainDataset' and you need to change the path according to your own environment.

Run train.py, and train it!
