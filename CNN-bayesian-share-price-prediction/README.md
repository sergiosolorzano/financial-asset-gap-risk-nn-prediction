## Convolutional Neural Network with Bayesian hyperparameter optimization to predict next day share price from a stock price time series

## Preface
This is my enhanced end project for a [6-month Professional Certification at Imperial Business School on Machine Learning and Artificial Intelligence](https://execed-online.imperial.ac.uk/professional-certificate-ml-ai) that I completed in June 2024.

It is work in progress and focuses on using CNNs to predict the next-day share price of financial assets. Bayesian optimization helps narrowing the search space to evaluate hyperparameters. Unfortunately, GADF-encoded images as inputs has resulted in low prediction accuracy. These results suggest the temporal correlation between each pair of prices in the series in the form of GADF-encoded inputs is not sufficiently robust to capture the temporal structure of prices. 

<p align="center">
<img width="150" alt="star" src="https://github.com/sergiosolorzano/ai_gallery/assets/24430655/3c0b02ea-9b11-401a-b6f5-c61b69ad651b">
</p>

---------------------------------------------

## Description
I train and optimize the hyperparameters for a LeNet5-design based Convolutional Neural Network to predict the next-day share price.
I test the model with the share price time series of the Sylicon Valley Bank for the period before and after bankruptcy.
The repo runs on python and leverages available pytorch libraries.

The share prices' day Low, High, Close, Open, Adjusted Close time series are encoded into 32x32 images using [pyts summation Gramian angular field (GAF)](https://pyts.readthedocs.io/en/stable/auto_examples/image/plot_single_gaf.html) to obtain a temporal correlation between each pair of prices in the series.
Render of a GAF 32-day share price time series window for each feature:
<img width="1045" alt="image" src="https://github.com/sergiosolorzano/CNN-bayesian-share-price-prediction/assets/24430655/985af796-f2d1-43c2-98e9-86e9610262dc">

Render average of the above GAF images:

<img width="225" alt="image" src="https://github.com/sergiosolorzano/CNN-bayesian-share-price-prediction/assets/24430655/27cb4600-58c8-42ca-8968-d0a1b6d99586">

I generate a stack of 32x32 images with shape (5, 491, 32, 32) which represents each of the 5 share price features' time series. Each image represents a time series window of 32 days. I slide each window by 1 day from Ti to T(i+32) hence obtaining 491 time series windows or GAF images for each feature.

The actual share price for each window is its the next day share price. 

The image dataset is split 80/20 into training/testing datasets.
The CNN is trained in mini-batches of 10 windows for each of the 5 features.

## DATA
I use [Yahoo Finance](https://pypi.org/project/yfinance/) python package and historical daily share price database for the period 2021-10-01 to 2023-12-01.

### Data stack to train the model:
For each of the 5 features (Close, High, etc), I generate sliding windows of prices for 491 days.
Original Time Series for Each Feature
-------------------------------------
![alt text](images/features_total.png)

I slide each window by 1 day from Ti to T(i+32) hence obtaining 491 time series windows or GAF images for each feature.

Sliding Window Process For Each Feature
---------------------------------------
![alt text](images/features_windows_sliding.png)

The actual price for each window is the price of the relevant feature at time end_of_window_day+1.

Generated Windows for Each Feature
----------------------------------
![alt text](images/features_windows.png)

Effectively, I generate a stack of 32x32 images with shape (5, 491, 32, 32) which represents each of the 5 share price features' time series. Each image represents a time series window of 32 days. 32 because GAF obtain a temporal correlation between each pair of prices in the series, like a grid of each day price.

Tensors of torch.Size([5, 1, 32, 32]) up to 491 are used to train the model.

Stack of Images (Shape: 5, 491, 32, 32)
---------------------------------------
![alt text](images/features_stacked.png)

The actual share price for each window is its next day share price.

## MODEL 
A LeNet5-design based Convolutional Neural Network which includes:
+ 1 Convolution Layer 1: It's output is processed through a Rectified Linear Unit ReLU activation function and Max Pool kernel.
+ 1 Convolution Layer 2: It's output is processed through a ReLU activation function and Max Pool kernel.
+ 1 Fully Connected Layer 1: It's output is processed through a ReLU activation function.
+ 1 Fully Connected Layer 2: It's output is processed through a ReLU activation function.
+ 1 Fully Connected Layer 3: It's output is processed through a ReLU activation function.
+ filter_size_1=(2, 2) applied to Convo 1
+ filter_size_2=(2, 3) applied to Max Pool
+ filter_size_3=(2, 3) applied to Convo 2
+ stride=2 for convo layers

The model incorporates drop out regularization on the fully connected layers.

The choice of model used leverages prior work and there is no other particular reason but to test the concept.

## HYPERPARAMETER OPTIMSATION
The repo optimizes the model's hyperparameters leveraging [BayesianOptimization library s_opt module](https://github.com/bayesian-optimization/BayesianOptimization).

I run up to 10,000 epochs and optimize the number of outputs for the Convolution Layer 1 and 2, learning rate and Dropout probability for 10 steps of bayesian optimization and steps of random exploration. See optimizer_results.txt for these results. The models are saved in /bayesian_optimization_saved_models:

    pbounds = {'output_conv_1': (40, 80), 'output_conv_2': (8, 16), 'learning_rate': (0.00001, 0.0001), 'dropout_probab': (0.0, 0.5), 'momentum': (0.8, 1.0)}

This is only an initial choice in the search space.

## RESULTS
The model predicts at low accuracy and mostly fails to converge to near zero loss when backpropagating, though the notebook's hyperparameters lead to convergence. Literature indicates a LetNet design is not optimal to fit the time series data as the model fails to capture temporal dependencies in time series data. Provided this results and the cost to run bayesian optimization it is not worth running further scenarios but explore alternatives.

Bayesian optimization results helped to manually explore higher accuracy hyper-parameter and model parameters.

Different model designs, in particular a Long short-term memory model (LTSM) may be more suited for this prediction task.

Best Bayesian optimization test dataset highest accuracy performance achieves a score of 3.125% for the output shown below. Bayesian simulations were run on an Azure Virtual Machine NC4as T4 v3 instance over four days.

    'params': {'dropout_probab': 0.49443054445324736, 'learning_rate': 7.733490889418554e-05, 'momentum': 0.8560887984128811, 'output_conv_1': 71.57117313805955, 'output_conv_2': 8.825808052621136}

Bayesian optimization results helps us to manually explore hyper-parameters and model parameter optimal results: The model achieves percentage of predictions within 2 decimal places: 16.46%, 1 decimal places: 39.79%, 0 decimal places: 68.75%, and mean % Diff:111.76%

    accuracy = (correct price compared at [x] decimal places / total) * 100

    'params': {'dropout_probab': 0, 'learning_rate': 0.0001, 'momentum': 0.9, 'output_conv_1': 40, 'output_conv_2': 12}

The mean sum of predicted-to-actual predict price difference to 2.dp as a percentage of the actual price is 112%. This gives us a relative mesaure of the mean percentage difference. This metric provides lower explainability than the accuracy metric mentioned above. In particular there are significant outliers that bias this mean sum result and the analysis would benefit from removing outliers which I have not done.

    batch_absolute_diff = torch.abs(predicted_rounded - actual_rounded)
    batch_percentage_diff = (batch_absolute_diff / actual_rounded) * 100
    #accumulate sum of diffs
    sum_diff += torch.sum(batch_percentage_diff).item()
    mean_percentage_diff = (abs(sum_diff) / total)

Another source for improvement may derive from running the network with log return inputs instead of outright prices.

Finally, since training is carried out 5 different price features, I may be adding unnecessary noise on the testing phase as the market may re-adjust post closing for an over-reaction and would be reflected on the Open the next day. These metrics between features has not been checked.

## ACKNOWLEDGEMENTS
I thank [Yahoo Finance](https://pypi.org/project/yfinance/) for the time series data provided. I also thank for the inspiration [repo](https://github.com/ShubhamG2311/Financial-Time-Series-Forecasting), the [BayesianOptimization library s_opt module](https://github.com/bayesian-optimization/BayesianOptimization), and the clarity on RNNs advantages found [in this research paper](https://arxiv.org/pdf/1506.00019).
