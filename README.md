## Analysis of Financial Asset Gap Risk Using Neural Networks and Machine Learning Techniques

<p align="center">
  <img src="https://img.shields.io/badge/Status-Work%20In%20Progress-red" alt="Work In Progress">
</p>

## Description
The inspiration for this project stems from the interest to continue learning after completing in June 2024 a [6-month Professional Certification at Imperial Business School on Machine Learning and Artificial Intelligence](https://execed-online.imperial.ac.uk/professional-certificate-ml-ai).

I continue experimenting with neural networks on this project. I am privileged to be guided by [Ali Muhammad](https://www.linkedin.com/in/muhammad-ali-76551016/), who lectured me during my certification at Imperial.

In this project we use different neural network approaches to estimate gap risk on the price of financial assets, and each approach is held in a subdirectory in this repo. The projects in this repo may slightly deviate from this objective as I explore and research associated predictions that help me build towards the end goal.

The project is work in progress. You can follow the journey in this [blog](https://tapgaze.com/blog/neural-networks-and-gap-risk-in-finance).

## Project Directory Structure:
### CNN-bayesian-share-price-prediction: Convolutional Neural Network to predict next day share price from a stock price time series, with Bayesian hyperparameter optimization
  The project includes a LeNet-5-based design, which is a Convolutional Neural Network (CNN) with both convolutional and linear (fully connected) layers, to predict next day share price from a stock price time series. Stock financial time series are encoded in Gramian Angular Field images and used as inputs.

### simple-lstm: I am building my understanding how LSTM models function by building a simple LSTM project
I have studying and preparing a vanilla Recurrent Neural Network's (RNN), which I expect to be more powerful than fixed networks where the entire state of the network is lost after each data point is processed, something detrimental to the prediction power of the model where data points are related in time. By contrast, RNNs sequencing processing method and its design to influence outputs by the inputs fed in, in my case sequenced GAF-encoded images, and the history of inputs fed in the past which provides the ability to selectively embed information across sequence steps similar to an autoregressive approach, may capture these temporal relationships. This may lead me to test Long-Short-Term-Memory RNNs to help overcome vanishing gradients and capture long and short term memory that provides temporal memory for the time series. I may then add transformers to the design based on recent successes.

##
If you find this helpful you can buy me a coffee :)
   
<a href="https://www.buymeacoffee.com/sergiosolorzano" target="_blank"><img src="https://www.buymeacoffee.com/assets/img/custom_images/orange_img.png" alt="Buy Me A Coffee" style="height: 41px !important;width: 174px !important;box-shadow: 0px 3px 2px 0px rgba(190, 190, 190, 0.5) !important;-webkit-box-shadow: 0px 3px 2px 0px rgba(190, 190, 190, 0.5) !important;" ></a>
      
