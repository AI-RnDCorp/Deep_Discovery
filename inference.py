import os
import re
import json
import time
import logging
from datetime import datetime
from typing import Optional, Dict, Any

# Importing third-party libraries
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import (accuracy_score, precision_score, 
                            recall_score, f1_score, roc_auc_score)
from rdkit import Chem
from rdkit.Chem import inchi, AllChem
import torch
from torch import nn

# Importing custom modules
from options.base_options import BaseOptions
from data.BindingDB_dataset import *
from data.Dataloader import *
from utils.optimized_feature_extractor import *
from utils.util import *
from model.drug_discovery import *



class DeepDiscovery:
    def __init__(self, opt):
        self.opt = opt
        if opt.gpu_ids:
            device = opt.gpu_ids[0]
        else:
            device = 'cpu'
        print("device:", device) 
        self.device = device
        
        self.data_csv = pd.read_csv(opt.train_datapath)

        print('---------- Networks initialized -------------')    
    
        print(f'모델을 불러옵니다:{self.opt.weight_pt}')
        docking_model, optimizer = define_deepdiscovery_initializer(opt=self.opt, device=device, model_path=self.opt.weight_pt)
        self.docking_model = docking_model
        
        
        
    # valid input check
    def input_wrapper(self, input_type: str, input_data: str, input_category: list, Temp: list, pH: list):
        """
        input_type = SMILES, AAseq
        input_data = smiles or AAseq
        input_category = ['all'], ['US Patent' 'Curated from the literature by BindingDB' 'PubChem' 'CSAR' 'D3R' 'WIPO' 'PDSP Ki' 'ChEMBL']  
        """
        amino_char = 'ACBEDGFIHKMLONQPSRUTWVYXZ'
        self.input_type = input_type
        # Invalid SMIELS input    
        if input_type == 'SMILES':
            mol = Chem.MolFromSmiles(input_data)
            if mol == None:
                print('Invalid SMILES input')
                return None
        # Invalid AAseq input
        else:
            sequence = input_data.upper()
            pattern = f"^[{amino_char}]+$"
            if not bool(re.match(pattern, sequence)):
                print('Invalid AAseq input')
                return None
        return [input_type, input_data, input_category, Temp, pH] 
        
    # filtering the csv
    def filtered_csv(self, inference_inputs):
        """
        database is for accuracy and referring
        """
        input_type, input_data, input_category, Temp, pH = inference_inputs
        database_csv = self.data_csv[self.data_csv[input_type] == input_data]
        # pH filtering
        condition_pH = (database_csv['pH'] >= pH[0]) & (database_csv['pH'] <= pH[1])
        database_csv = database_csv[condition_pH]

        condition_Temp = (database_csv['Temp (C)'] >= Temp[0]) & (database_csv['Temp (C)'] <= Temp[1])
        database_csv = database_csv[condition_Temp]
        if len(database_csv) > 0:
            print(f'Number of corresponding data in database is: {len(database_csv)}')
        else:
            print(f"Number of corresponding data in database is none")
        # catagory filtering
        if 'all' in input_category:

            filtered_data_csv = self.data_csv
        else:
            filtered_data_csv = self.data_csv[self.data_csv['Curation/DataSource'].isin(input_category)]

        if input_type == "SMILES":
            filtered_data_csv = filtered_data_csv.drop_duplicates(subset='AAseq').reset_index(drop=True)
            pass
        else:
            filtered_data_csv = filtered_data_csv.drop_duplicates(subset='SMILES').reset_index(drop=True)
        
        print(f"Number of filtered data is: {len(filtered_data_csv)}")
        
        
        
        return database_csv, filtered_data_csv

    
    def inference_run(self, inference_inputs, database_csv, filtered_data_csv):
        
        print('---------- Dataset initialized -------------')
        
        self.filtered_data_csv = filtered_data_csv
        
        data_loader = BD_inference_dataloader(self.opt, inference_inputs, database_csv, self.filtered_data_csv)
        
        filtered_results_df, elapsed_time, plp, batch_min, accuracy = self.inference(self.docking_model, data_loader, self.device)
        
        # Sort in ascending order
        sorted_filtered_data = self.filtered_data_csv.sort_values(by='BindingDB Reactant_set_id', ascending=True)
        sorted_result = filtered_results_df.sort_values(by='setID', ascending=True)[['Probabilities', 'Predictions']] 
        
        final_result_csv = sorted_filtered_data.assign(**sorted_result)
        
        final_result_csv = final_result_csv.drop(columns=['IC50_label', 'pH', 'Temp (C)', 'BindingDB Reactant_set_id'])
        
        if self.input_type == 'SMILES':
            category_name = 'Target'
            columns  = ['AAseq', 'Target Name', 'PDB ID(s) of Target Chain', 'Curation/DataSource', 'Target Source Organism According to Curator or DataSource', 'Probabilities', 'Predictions']
            final_result_csv = final_result_csv[columns]
            final_result_csv = final_result_csv.drop_duplicates(subset='Target Name').reset_index(drop=True)
            sorted_final_result = final_result_csv.sort_values(by='Probabilities', ascending=False)
            
            columns_sort = ['Target Name', 'AAseq', 'Probabilities']

            sorted_result_csv = sorted_final_result[:10]
            sorted_result_csv = sorted_result_csv[columns_sort]
                    
        elif self.input_type =='AAseq':
            category_name = 'Ligand'
            columns = ['SMILES', 'InChI Key', 'Curation/DataSource', 'Target Source Organism According to Curator or DataSource', 'Probabilities', 'Predictions']
            final_result_csv = final_result_csv[columns]
            sorted_final_result = final_result_csv.sort_values(by='Probabilities', ascending=False)
            columns_sort = ['InChI Key', 'SMILES', 'Probabilities']
            sorted_result_csv = sorted_final_result[:10]
            sorted_result_csv = sorted_result_csv[columns_sort]

        formatted_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            # 월을 문자열로 변환
#        month_str = time.strftime("%B", time.localtime())  # %B는 전체 월 이름을 반환합니다.
        #formatted_time_with_month_str = formatted_time.replace(time.strftime("%m", time.localtime()), month_str)

        

        return final_result_csv, sorted_result_csv ,formatted_time, category_name, elapsed_time, plp, batch_min, accuracy
    
    def inference(self, model, data_loader, device):
        
        model.eval()
        
        # result save path
        save_folder = './results'
        
        mkdir(save_folder)
        
        filtered_loader = data_loader.get_filtered_loader()    
        filtered_pbar = tqdm(filtered_loader, desc='filtered data inference processing...', unit='batch')
        
        filtered_predictions, filtered_probabilities, filtered_setid = [], [], []
        
        print(f"# of batch in filtered_loader: {len(filtered_loader)}")
        if data_loader.is_database():
            database_loader = data_loader.get_database_loader()
            database_pbar = tqdm(database_loader, desc='database data inference processing...', unit='batch')
            print(f"# of batch in database_loader: {len(database_loader)}")
            database_predictions, database_probabilities, database_labels, database_setid = [], [], [], []
        
        inference_start_time = time.time()
                
        with torch.no_grad():
            for i, data in enumerate(filtered_pbar):
                df, pf, set_id = data
                df = [[d.to(device) if isinstance(d, torch.Tensor) else d for d in sublist] for sublist in df]
                pf = pf.to(device)
                
                output = model(df, pf)            
                probabilities = torch.sigmoid(output)
                predictions = (probabilities > 0.5).float()
                
                filtered_predictions.extend(predictions.cpu().numpy())
                filtered_probabilities.extend(probabilities.cpu().numpy())
                filtered_setid.extend(set_id.cpu().numpy())
            # 결과를 DataFrame으로 저장
            
            filtered_predictions = [element[0] for element in filtered_predictions]
            filtered_probabilities = [element[0] for element in filtered_probabilities]

            filtered_results_df = pd.DataFrame({
                'setID' : filtered_setid,   
                    'Probabilities': filtered_probabilities,
                    'Predictions': filtered_predictions      
            })
    
            elapsed_time = time.time() - inference_start_time   
            elapsed_min = elapsed_time/60
            plp = (len(filtered_setid)/elapsed_time)
            batch_min = (elapsed_min/len(filtered_loader))
                
            if data_loader.is_database():
                for i, data in enumerate(database_pbar):
                    df, pf, gt, set_id = data
                    df = [[d.to(device) if isinstance(d, torch.Tensor) else d for d in sublist] for sublist in df]
                    pf, gt = pf.to(device), gt.to(device)
                    
                    output = model(df, pf)
                    
                    probabilities = torch.sigmoid(output)
                    predictions = (probabilities > 0.5).float()
                
                    database_predictions.extend(predictions.cpu().numpy())
                    #database_probabilities.extend(probabilities.cpu().numpy())
                    database_labels.extend(gt.cpu().numpy())
                # database_setid.extend(set_id)
        
                accuracy = accuracy_score(database_labels, database_predictions)
                #precision = precision_score(database_labels, database_predictions)
                #recall = recall_score(database_labels, database_predictions)
                #f1 = f1_score(database_labels, database_predictions)

                # print(f"Database 결과:")
                # print(f"Accuracy: {accuracy:.4f}")
                # print(f"Precision: {precision:.4f}")
                # print(f"Recall: {recall:.4f}")
                # print(f"F1 Score: {f1:.4f}")

            else:
                accuracy = 0

            return filtered_results_df, elapsed_time, plp, batch_min, accuracy
