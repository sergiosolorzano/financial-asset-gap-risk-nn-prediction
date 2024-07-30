# Datasheet Template

Comments on the data used.

## Motivation

- The share price time series is provided for free by Yahoo Finance. Yahoo Finance offers private and public availability for financial data. Yahoo Finance is owned by [Yahoo inc](https://en.wikipedia.org/wiki/Yahoo!_Finance).

 
## Composition

- The dataset comprises Low, High, Open, Close, Adjusted Close share prices and volume for Sylicon Valley Bank.
- The data is public and non-confidential

## Collection process

- The data was downloaded via the [Yahoo Finance](https://pypi.org/project/yfinance/) package.
- The sampling for the data captures ~520 days for the period 2021-10-01 to 2023-12-01, representing the period before and after the collapse of Silicon Valley Bank.

## Preprocessing/cleaning/labelling

- Days with missing data were excluded from the dataset.
- The data is not coded to be persistent in the jupyter notebook. 
 
## Uses

- Beyond the use of the data for this repo, volume can be used to train the model. Yahoo Finance may offer additional company specific data that can aid training.

## Distribution

- The data is distributed via [Yahoo Finance](https://uk.finance.yahoo.com/) and [pip package](https://pypi.org/project/yfinance/).
- The data is public and free for personal projects. Commercial usage of the API requires a paid subscription which is not the case for this project.

## Maintenance

- The dataset is maintained by [Yahoo Finance](https://uk.finance.yahoo.com/) and pip's package for access.

