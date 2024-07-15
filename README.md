## Convolutional Neural Network with Bayesian hyperparameter optimization to predict next day share price from a stock price time series

## Preface
This is my end project for a [6-month Professional Certification at Imperial Business School on Machine Learning and Artificial Intelligence](https://execed-online.imperial.ac.uk/professional-certificate-ml-ai) that I completed in June 2024.

I initially intended to focus on CNNs to predict the next-day share price of financial assets. Unfortunately, GADF-encoded images as inputs has resulted in low prediction accuracy. These results suggest the temporal correlation between each pair of prices in the series in the form of GADF-encoded inputs is not sufficiently robust to capture the temporal structure of prices. This has led me to enhance this analysis in this repo which I will do in named git-branches.

I have started by preparing a vanilla Recurrent Neural Network's (RNN), which I expect to be more powerful than fixed networks where the entire state of the network is lost after each data point is processed, something detrimental to the prediction power of the model where data points are related in time. By contrast, RNNs sequencing processing method and its design to influence outputs by the inputs fed in, in my case sequenced GADF-encoded images, and the history of inputs fed in the past which provides the ability to selectively embed information across sequence steps similar to an autoregressive approach, may capture these temporal relationships. This may lead me to test Long-Short-Term-Memory RNNs to help overcome vanishing gradients and capture long and short term memory that provides temporal memory for the time series. I may then add transformers to the design based on recent successes. Stay tunned !

## Description
I train and optimize the hyperparameters for a LeNet5-design based Convolutional Neural Network to predict the next-day share price.
I test the model with the share price time series of the Sylicon Valley Bank for the period before and after bankruptcy.
The repo runs on python and leverages available pytorch libraries.

The share prices' day Low, High, Close, Open, Adjusted Close time series are encoded into 32x32 images using [pyts summation Gramian angular field (GAF)](https://pyts.readthedocs.io/en/stable/auto_examples/image/plot_single_gaf.html) to obtain a temporal correlation between each pair of prices in the series.
Render of a GAF 32-day share price time series window for each feature:
<img width="1045" alt="image" src="https://github.com/sergiosolorzano/CNN-bayesian-share-price-prediction/assets/24430655/985af796-f2d1-43c2-98e9-86e9610262dc">

Render average of the above GAF images:

<img width="225" alt="image" src="https://github.com/sergiosolorzano/CNN-bayesian-share-price-prediction/assets/24430655/27cb4600-58c8-42ca-8968-d0a1b6d99586">

I generate a stack of 32x32 images with shape (5, 491, 32, 32) which represents each of the 5 share price features' time series.
Each image represents a time series window of 32 days. I slide each window by 1 day from Ti to T(i+32) hence obtaining 491 time series windows or GAF images for each feature.

The image dataset is split 80/20 into training/testing datasets. The actual share price for each window is its the next day share price.
The CNN is trained in mini-batches of 10 windows for each of the 5 features.

This repo is my choice for the end of course project at [Professional Certificate in Machine Learning and Artificial Intelligence](https://execed-online.imperial.ac.uk/professional-certificate-ml-ai)

## DATA
I use [Yahoo Finance](https://pypi.org/project/yfinance/) python package and historical daily share price database for the period 2021-10-01 to 2023-12-01.

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
The model predicts at low accuracy and fails to converge to near zero loss when backpropagating. Literature indicates a LetNet design is not optimal to fit the time series data as the model fails to capture temporal dependencies in time series data. Provided this results and the cost to run bayesian optimization it is not worth running further scenarios but explore alternatives.

Bayesian optimization results helped to manually explore higher accuracy hyper-parameter and model parameters.

Different model designs, in particular a Long short-term memory model (LTSM) may be more suited for this prediction task.

Best Bayesian optimization test dataset highest accuracy performance achieves a score of 3.125% for:

    'params': {'dropout_probab': 0.49443054445324736, 'learning_rate': 7.733490889418554e-05, 'momentum': 0.8560887984128811, 'output_conv_1': 71.57117313805955, 'output_conv_2': 8.825808052621136}

Bayesian optimization results helps us to manually explore hyper-parameters and model parameter optimal results, achieving 17.91% accuracy as shown below. Bayesian simulations were run on an Azure Virtual Machine NC4as T4 v3 instance over four days:

    accuracy = (correct price compared at 2 d.p / total) * 100

    'params': {'dropout_probab': 0, 'learning_rate': 0.0001, 'momentum': 0.9, 'output_conv_1': 40, 'output_conv_2': 12}

The mean sum of predicted-to-actual predict price difference to 2.dp as a percentage of the actual price is 95%. This gives us a relative mesaure of the mean percentage difference. This metric provides lower explainability than the accuracy metric mentioned above. In particular there are significant outliers that bias the result and the analysis would benefit from removing these which I have not done. I also acknowledge 2 d.p. may be an unnecessarily too high a threhold to determine this difference:

    batch_absolute_diff = torch.abs(predicted_rounded - actual_rounded)
    batch_percentage_diff = (batch_absolute_diff / actual_rounded) * 100
    #accumulate sum of diffs
    sum_diff += torch.sum(batch_percentage_diff).item()
    mean_percentage_diff = (abs(sum_diff) / total)

## ACKNOWLEDGEMENTS
I thank [Yahoo Finance](https://pypi.org/project/yfinance/) for the time series data provided. I also thank for the inspiration [repo](https://github.com/ShubhamG2311/Financial-Time-Series-Forecasting), the [BayesianOptimization library s_opt module](https://github.com/bayesian-optimization/BayesianOptimization), and the clarity on RNNs advantages found [in this research paper](https://arxiv.org/pdf/1506.00019).
