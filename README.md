Stumble Upon Evergreen Classification Problem

Starting with reading data, we take the whole data using the pandas library to explore the data and gain useful insight from it. The actual data in the given dataset is contained in the “boilerplate” column which would be used for the nlp task. 

Data Cleaning: 
The data in the “boilerplate” column consists of a lot of garbage data which needs to be cleaned before we feed it into our model. 
The def clean(data) functions, takes the data frame and cleans the data. 
This function removes some redundant texts such as “title” and “url” which only increase the bias during the model training. It also converts the whole text into lowercase. 
This is to generate a large corpus of textual data which can be fed into the model.


Data separation:
The “boilerplate” data from the df_train dataframe is distributed into train data and validation data. The output label of train data is present in the “.label” column of the data. 

The validation data is used to provide an unbiased evaluation of a model for on the training dataset while tuning model hyperparameters.

Model:

We cannot feed raw text into the neural network model. Hence we need to convert our text data into numeric data. 
So we tokenize our text using the “nnlm-em-dim50” available on Tensorflow-hub.It is a text embedding model based on feed-forward Neural-Net Language Models with pre-built OOV. It maps from text to 50-dimensional embedding vectors.
This module takes a batch of sentences in a 1-D tensor of strings as input and preprocesses it by splitting on spaces.
This model also cleans the data on stop words and punctuation marks.
More details about the model on https://tfhub.dev/google/nnlm-en-dim50/2

The model used in the training is a Keras Sequential model with the text embedding layer as the first layer. This layer preprocesses the data and feeds it forward to the next layer.
The next layer is Batch normalization layer which is used for training deep neural networks that standardizes the inputs to a layer for each mini-batch. This layer stabilizes the learning process and enables faster training. 

The next layer is a 128 neuron dense layer which has “relu” as an activation function. 
The next layer is a Dropout layer which prevents a model from overfitting. Dropout works by randomly setting outgoing edges of hidden units to 0 at each update of the training phase. This model has a dropout of 0.2.

The next layer is again a dense layer with 64 neurons and “relu” as activation function.
The last layer in the model is a 1 neuron layer with “sigmoid” as the architecture. This layer is the output layer, which predicts the output based on the data fed by the previous layers. 




Model Compilation: 

We compile the model with loss function as “binary cross entropy” and optimizer as “adam”. 
The evaluation metric is AUC().

The loss function evaluates how well the algorithm models the dataset.
The optimizer changes the attributes of the neural network such as weights and learning rate in order to reduce the losses. 
The evaluation metric computes the approximate AUC (Area under the curve) via a Riemann sum.


Score Achieved: 0.76813 (as per Kaggle)
GitHub link : https://github.com/rohan1907/StumbleUpon-Evergreen-Classification-Challenge



