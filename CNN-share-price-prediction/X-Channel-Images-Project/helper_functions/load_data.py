import os
import torch
print(torch.__version__)
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, SequentialSampler, Subset

import numpy as np
import yfinance as yf

import matplotlib.pyplot as plt
import seaborn as sns

import helper_functions.generate_images as generate_images

class DataPrep(Dataset):
    def __init__(self, inputs, labels):
        self.inputs = inputs #all features in one large array
        self.labels = labels
        self.transform = generate_images.SetTransform()

    def __len__(self):
      return len(self.inputs)

    def __getitem__(self, index):
        X = self.inputs[index]
        Y = self.labels[index]
        return X, Y
  
    def prepare_ordered_dataset(self):
        x = []
        y = []
        #print("len inputs", len(self.inputs), "shape", self.inputs.shape, self.inputs.shape[0])
        #print("len images 0",len(self.inputs), "len images 0:",len(self.inputs[0]))
        #print("images 0:",self.inputs[0])
        #print("labels",self.labels)
        #print("len labels", len(self.labels), self.labels.shape, self.labels.shape[0])

        for image_num in range(self.inputs.shape[0]):
            #print("img num",image_num,"image",self.inputs[image_num])
            #print("len image data 0",len(self.inputs[data_window][0]),"shape",self.inputs[data_window].shape)
            #print("label data",self.labels[image_num][0])
            #print("imag num:",image_num)
            #print("image data at index image_num len:",len(self.inputs[image_num]))
            
            self.inputs[image_num] = self.transform(self.inputs[image_num])
            
            x.append(np.expand_dims(self.inputs[image_num], axis=0))
            y.append(self.labels[image_num])
            #print("img num",image_num,"label",self.labels[image_num])
            #print("img num",image_num,"img",self.inputs[image_num])
            #print("img num",image_num,"img len",len(self.inputs[image_num]))
            
        #cnn requests labels size (4,1) instead of (4)
        y = np.expand_dims(y, axis=1) 
        #print("size self",self.inputs.shape,self.labels.shape)
        #print("size self",len(x),len(y))
        dataset = [(img, label) for img, label in zip(x, y)]
        #print("type dataset returned",type(dataset), len(dataset), len(dataset[0]), len(dataset[1]))
        #print("len dataset[0][0]",len(dataset[0][0][0][0]))
        #print("len dataset[1][1]",len(dataset[1][1]))
        #print("dataset[0]",dataset[1])
        return dataset
        
        #return np.array(x),np.array(y)
    
    def split_data(self,dataset, batch_size, test_size, train_shuffle=False):
        print("split data test size",test_size)
        num_samples = len(dataset)
        #print("numsamples",num_samples)
        num_test_samples = int(test_size * num_samples)
        num_train_samples = num_samples - num_test_samples
        num_train_samples = num_samples - num_test_samples
        #print("num_train_samples",num_train_samples)
        #indices = np.random.permutation(num_samples)
        indices = np.arange(num_samples)
        train_indices = indices[:num_train_samples]
        test_indices = indices[num_train_samples:]
        print("len train",len(train_indices),"len test",len(test_indices))

        #random
        # train_sampler = SubsetRandomSampler(train_indices)
        # test_sampler = SubsetRandomSampler(test_indices)
        ## sequential
        train_subset = Subset(dataset, train_indices)
        test_subset = Subset(dataset, test_indices)
        train_sampler = SequentialSampler(train_subset)
        test_sampler = SequentialSampler(test_subset)

        train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler,shuffle=train_shuffle)
        test_loader = DataLoader(dataset, batch_size=batch_size, sampler=test_sampler)

        #for e in train_loader:
            #print("train loader ele",e)

        # sample_batch = next(iter(train_loader))
        # input_shape = sample_batch[0].shape
        # label_shape = sample_batch[1].shape
        # print("input len",len(input_shape),"input shape",input_shape,"label len",len(label_shape))

        return train_loader, test_loader
    
def Generate_Train_And_Test_Loaders(feature_image_dataset_list_f32,labels_scaled_list_f32, 
                                test_size, batch_size, train_shuffle=False):

    #print("feature_image_dataset_list_f32[0][0].shape",feature_image_dataset_list_f32[0][0].shape, "feature_image_dataset_list_f32[0][0].shape[0]", feature_image_dataset_list_f32[0][0].shape[0])

    #reshape for cnn
    #reshaped_feature_image_dataset_list_f32 = np.expand_dims(feature_image_dataset_list_f32[0][0].reshape(-1, *feature_image_dataset_list_f32[0][0].shape[2:]), axis=1)
    #print("feature_image_dataset_list_f32 shape",feature_image_dataset_list_f32.shape)
    #print("res",reshaped_feature_image_dataset_list_f32.shape)
    #print("labels list",labels_scaled_list_f32)

    #generate a list for images and labels
    data_prep_class = DataPrep(feature_image_dataset_list_f32, labels_scaled_list_f32)
    

    #print("feature_image_dataset_list_f32",feature_image_dataset_list_f32[0][0].shape)
    #print("labels_scaled_list_f32",labels_scaled_list_f32.shape)
    #returns list size all observations of all features of size 2:
    #(image32x32,label) i.e. shape (4*480,32,32) and (4*480,1)
    dataset = data_prep_class.prepare_ordered_dataset()

    for c in range(len(dataset[0])):
        print(f"size labels {c}",dataset[1][c].size)
        print(f"size image {c}",dataset[0][c].shape)

    train_loader, test_loader = data_prep_class.split_data(dataset, 
                                                            batch_size, test_size,
                                                            train_shuffle)

    # for c,e in enumerate(train_loader):
    #     print("count",c)
        # print("type",type(e))
        # print("imga",e[0].shape)
        # print("label",e[1].shape)
    #returns 191 train_loaders that contain batch of 10 images32x32 and 10 labels
    #=191*10=1910 i.e. 80% of 2400 total
    return train_loader, test_loader

def Generate_Loaders(feature_image_dataset_list_f32,labels_scaled_list_f32, test_size, batch_size, train_shuffle=False):
    train_loader,test_loader = Generate_Train_And_Test_Loaders(feature_image_dataset_list_f32,labels_scaled_list_f32, test_size, batch_size=batch_size, train_shuffle=False)

    return train_loader,test_loader

def import_dataset(ticker, start_date, end_date):

    dataset = yf.download(ticker, start=start_date, end=end_date, interval='1d')

    dataset = dataset.dropna().dropna()
    dataset.dropna(how='any', inplace=True)
    print("Num rows for df Close col",len(dataset['Close']))
    print(dataset.columns)
    dataset = dataset.reset_index()
    #reorder to split the data to train and test
    desired_order = ['Date','Open', 'Close', 'High', 'Low']
    if 'Date' in dataset.columns:
        dataset = dataset[desired_order]
    else:
        print("Column 'Date' is missing.")
    dataset = dataset.set_index('Date')
    #print(dataset.head())
    print("day count",dataset.index.max()-dataset.index.min())

    return dataset

def import_stock_data(stock_ticker, start_date, end_date):
    #import stock dataset
    stock_dataset_df = import_dataset(stock_ticker, start_date, end_date)
    stock_dataset_df.head()

    return stock_dataset_df