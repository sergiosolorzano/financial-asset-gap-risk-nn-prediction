## Analysis of Financial Asset Gap Risk Using Neural Networks and Machine Learning Techniques

## Description
The inspiration from this project stems from a [6-month Professional Certification at Imperial Business School on Machine Learning and Artificial Intelligence](https://execed-online.imperial.ac.uk/professional-certificate-ml-ai) that I completed in June 2024. 

The project uses different techniques to estimate gap risk on financial assets, and each is held in a subdirectory in this repo. The projects in this repo may slightly deviate from this objective as I explore related predictions that help me build towards the end goal.

I am continuously developing this, and it remains a work in progress.

## Project Directory Structure:
### CNN-bayesian-share-price-prediction: Convolutional Neural Network to predict next day share price from a stock price time series, with Bayesian hyperparameter optimization
  The project includes a Convolutional Neural Network to predict next day share price from a stock price time series. Bayesian optimization helps narrowing the search space of hyperparameters to also manually optimize for higher accuracy. GADF-encoded images used as inputs for this approach has resulted in low prediction accuracy. These results suggest the temporal correlation between each pair of prices in the series in the form of GADF-encoded inputs is not sufficiently robust to capture the temporal structure of prices.
### simple-lstm: I am building my understanding how LSTM models function by building a simple LSTM project
I have started by preparing a vanilla Recurrent Neural Network's (RNN), which I expect to be more powerful than fixed networks where the entire state of the network is lost after each data point is processed, something detrimental to the prediction power of the model where data points are related in time. By contrast, RNNs sequencing processing method and its design to influence outputs by the inputs fed in, in my case sequenced GADF-encoded images, and the history of inputs fed in the past which provides the ability to selectively embed information across sequence steps similar to an autoregressive approach, may capture these temporal relationships. This may lead me to test Long-Short-Term-Memory RNNs to help overcome vanishing gradients and capture long and short term memory that provides temporal memory for the time series. I may then add transformers to the design based on recent successes. Stay tuned !

## REFERENCES
The report includes relevant references used in the project, such as research papers and online resources. I thank [Yahoo Finance](https://pypi.org/project/yfinance/) for the time series data provided. I also thank for the inspiration [repo](https://github.com/ShubhamG2311/Financial-Time-Series-Forecasting), the [BayesianOptimization library s_opt module](https://github.com/bayesian-optimization/BayesianOptimization), and the clarity on RNNs advantages found [in this research paper](https://arxiv.org/pdf/1506.00019).
