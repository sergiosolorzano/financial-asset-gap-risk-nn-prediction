from sklearn.preprocessing import StandardScaler,MinMaxScaler
import sys
import os

#import scripts
import importlib as importlib
sys.path.append(os.path.abspath('./helper_functions'))
import helper_functions.generate_images as generate_images

#init parameters
class Parameters:
    def __init__(self):
        self.scenario = 0

        # Stock tickers
        self.train_stock_ticker = 'SIVBQ'
        self.external_test_stock_ticker = 'SICP'
        #self.test_stock_ticker = 'MSFT'
        self.index_ticker = '^SP500-40'
        
        # Close price time period
        self.start_date = '2021-12-05'
        self.end_date = '2023-01-25'

        #cols used
        self.training_cols_used = ["Open", "High", "Low", "Close"]
        self.external_test_cols_used = ["Open", "High"]

        # Time series to image transformation algorithm: GRAMIAN 1; MARKOV 2
        self.transform_algo_type = 1
        self.transform_algo = generate_images.TransformAlgo.from_value(self.transform_algo_type)
        self.image_resolution_x = 32
        self.image_resolution_y = 32
        
        # GAF image inputs
        self.gaf_method = "summation"#["summation", "difference"]
        self.transformed_img_sz = 32
        self.sample_range = (0, 1)#[(-1, 0), (0, 0.5), (0.5, 1), (1, 1)]
        
        # GRAMIAN/MARKOV: image transformation scale
        # self.scaler = MinMaxScaler(feature_range=(-1, 1))
        self.scaler = StandardScaler()#[StandardScaler(), MinMaxScaler()]

        # Training's test size
        self.training_test_size = 0.5
        self.external_test_size = 1

        self.model_name ='LeNet-5 Based Net'
        # Default hyperparameters
        self.filter_size_1 = (2, 3)
        self.filter_size_2 = (2, 2)
        self.filter_size_3 = (2, 3)

        self.stride_1 = 1
        self.stride_2 = 2

        self.output_conv_1 = 40
        self.output_conv_2 = 12
        self.output_FC_1 = 100
        self.output_FC_2 = 70
        self.final_FCLayer_outputs = 1

        self.learning_rate = 0.00001#[0.00001, 0.0001, 0.001]
        self.momentum = 0.9#[0.7, 0.8, 0.9]

        self.dropout_probab = 0#[0, 0.5, 0.8]

        self.batch_size = 16#[4, 16, 32]

        self.num_epochs_input = 10000

        self.loss_threshold = 0.0001

        self.epoch_running_loss_check = 500
        
        self.epoch_running_gradients_check = 4000
