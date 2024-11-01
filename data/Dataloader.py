from data.BindingDB_dataset import *
from utils.util import *
from torch.utils.data.dataloader import default_collate
import numpy as np
from torch.utils.data import Subset, DataLoader
import torch
import numpy as np
from tqdm import tqdm
import math

def mpnn_collate_func(x):
    # for normal
    mpnn_feature = [i[0] for i in x] 
    x_remain = [[i[1], i[2]] for i in x]
    x_remain_collated = default_collate(x_remain)

    return [mpnn_feature] + x_remain_collated


def mpnn_collate_func2(batch):
    mpnn_feature = [i[0] for i in batch]  # Assuming the first element is the feature
    x_remain = [[i[1], i[2], i[3]] for i in batch]  # Collect the remaining elements
    x_remain_collated = default_collate(x_remain)

    return [mpnn_feature] + x_remain_collated  # Return the collated features and remaining data


class BD_dataloader:
    def __init__(self, opt):
        self.opt = opt
        self.dataset = Binding_DB_dataset(opt)
        # 공통 데이터로더 설정
        self.dataloader_args = {
            'batch_size': opt.batchsize,
            'num_workers': int(opt.num_workers),
            'pin_memory': True,
            'drop_last': True,
            'collate_fn': mpnn_collate_func
        }
        
        if self.opt.isTrain:
            if opt.split_size:
                dataset_size = len(self.dataset)
                train_size = int(dataset_size * opt.split_size) 
                val_size = dataset_size - train_size
                # 전체 데이터셋의 인덱스를 생성
                indices = list(range(len(self.dataset)))
                np.random.shuffle(indices)
                split = int(np.floor(opt.split_size * dataset_size))

                # 섞인 인덱스를 사용하여 훈련 및 검증 데이터셋 생성
                train_indices, val_indices = indices[:split], indices[split:]
                train_dataset = Subset(self.dataset, train_indices)
                val_dataset = Subset(self.dataset, val_indices)
                print("Dataset Split Ratio Train/ Val is {}".format(self.opt.split_size))
                print("Train dataset size {}".format(train_size))
                print("Validation dataset size {}".format(val_size))
                
                self.train_dataloader = DataLoader(train_dataset, shuffle=True, **self.dataloader_args)
                self.val_dataloader = DataLoader(val_dataset, shuffle=False, **self.dataloader_args)
            else:
                print("Train the model with the entire dataset")
                self.train_dataloader = DataLoader(self.dataset, shuffle=True, **self.dataloader_args)
                
        else:
            print("Test the model") # test 진행시 1만개로 sampling 코드 추가
            self.test_dataloader = DataLoader(self.dataset, shuffle=False, **self.dataloader_args)    
    def get_train_loader(self):
        self.dataloader = self.train_dataloader
        return self.dataloader
    
    def get_val_loader(self):
        self.dataloader = self.val_dataloader
        return self.dataloader
    
    def get_test_loader(self):
        self.dataloader = self.test_dataloader
        return self.dataloader

    def __len__(self):
        return len(self.dataloader)

    def __iter__(self):
        for data in self.dataloader:
            yield data
        
        

class BD_inference_dataloader:
    def __init__(self, opt, inputs, database_csv, filtered_data_csv):
        
        
        self.opt = opt
        self.inputs = inputs
        self.input_type = inputs[0]
        self.max_batch_size = opt.batchsize  # 최대 배치 크기 설정 512        
        self.dataloader_args = {'shuffle': False,
            'num_workers': int(self.opt.num_workers),
            'pin_memory': True,
            'drop_last': False,
        }
        self.database_csv = database_csv
        self.filtered_data_csv = filtered_data_csv
        
        if len(self.database_csv) > 0:
            DB_inference_database_dataset = DB_inference_database(opt, self.inputs, self.database_csv)
            if len(self.database_csv) < self.max_batch_size:
                self.max_batch_size = len(self.database_csv)
           
            self.DB_database_loader = DataLoader(DB_inference_database_dataset, batch_size=self.max_batch_size, collate_fn = mpnn_collate_func2, **self.dataloader_args)
       
        else:
            self.DB_database_loader = None
            
        if len(filtered_data_csv) < self.max_batch_size:
            self.max_batch_size = len(filtered_data_csv)
        else:
            self.max_batch_size = opt.batchsize
            
        DB_inference_filtered_dataset = DB_inference_filtered(opt, self.inputs, filtered_data_csv)
       
        self.DB_filtered_loader = DataLoader(DB_inference_filtered_dataset, batch_size=self.max_batch_size, collate_fn= mpnn_collate_func, **self.dataloader_args)
        
    def is_database(self):
        return True if len(self.database_csv) > 0 else False
    
  
    def get_database_loader(self):
        self.db_loader = self.DB_database_loader
        return self.db_loader

    def get_filtered_loader(self):
        self.db_loader = self.DB_filtered_loader
        return self.db_loader

    def __len__(self):
        return len(self.db_loader)

    def __iter__(self):
        for data in self.db_loader:
            yield data
    