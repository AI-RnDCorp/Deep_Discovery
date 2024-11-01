from torch.utils.data import Dataset
import pandas as pd
from utils.optimized_feature_extractor import *
import numpy as np



def extract_features(smiles):
    
    Bond_block_input = Bond_index2Bond_Atom_feature(smiles)
    Atom_block_input = Atom_index2Atom_feature(smiles)
    bond_relation = Bond_relation(smiles)
    bgraph = making_bgraph(Bond_block_input, bond_relation)
    abgraph = making_abgraph(Atom_block_input, bond_relation)
    agraph = making_agraph(abgraph)
    
    return [Bond_block_input, Atom_block_input, bgraph, abgraph, agraph]

class Binding_DB_dataset(Dataset):    

    def __init__(self, opt):
        self.opt = opt
        if opt.isTrain: 
            self.data_csv = pd.read_csv(opt.train_datapath)  
        else:
            self.data_csv = pd.read_csv(opt.test_datapath)

        self.smiles = self.data_csv['SMILES'].values
        self.aaseq = self.data_csv['AAseq'].values
        self.ic50_label = self.data_csv['IC50_label'].values  
        
    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        
        drug_feature = extract_features(self.smiles[idx])
        protein_feature = profile_generation(self.aaseq[idx], max_len=1000)
        
        
        return (drug_feature,
                protein_feature,
                torch.tensor(self.ic50_label[idx], dtype=torch.float16))
        
class DB_inference_database(Dataset):

    def __init__(self, opt, inputs, database_csv):
        self.opt = opt
        self.input_type = inputs[0]
        self.input_data = inputs[1]
        self.database_csv = database_csv
        # database 데이터 처리
        self.ic50_label_database = database_csv['IC50_label'].values  
        self.set_id_database = database_csv['BindingDB Reactant_set_id'].values
        if self.input_type == 'SMILES':
            self.smiles = self.input_data
            self.drug_feature = extract_features(self.smiles)
            self.aaseq_database = database_csv['AAseq'].values
        else:
            self.aaseq = self.input_data
            self.protein_feature = profile_generation(self.aaseq, max_len=1000)
            self.smiles_database = database_csv['SMILES'].values
                

    def __len__(self):
        """
        returns the number of data from filtered dataset.
        """
        return len(self.set_id_database)

    def __getitem__(self, idx):
        
        lablel_database = torch.tensor(self.ic50_label_database[idx], dtype=torch.float16)
        set_id_database = self.set_id_database[idx]
        if self.input_type == 'SMILES':
            drug_feature_database = self.drug_feature
            protein_feature_database = profile_generation(self.aaseq_database[idx], max_len=1000) 
            
        else:
            protein_feature_database = self.protein_feature
            drug_feature_database = extract_features(self.smiles_database[idx])
            
        database_item = (drug_feature_database,
                    protein_feature_database, lablel_database, set_id_database)
        
        return database_item

class DB_inference_filtered(Dataset):

    def __init__(self, opt, inputs, filtered_data_csv):
        self.opt = opt
        self.input_type = inputs[0]
        self.input_data = inputs[1]
        self.filtered_data_csv = filtered_data_csv 
        # filtered data 처리
        # smiles 한개만 할당
        self.set_id = filtered_data_csv['BindingDB Reactant_set_id'].values

        if self.input_type == 'SMILES':
            self.smiles = self.input_data
            self.drug_feature = extract_features(self.smiles)
            self.aaseq = filtered_data_csv['AAseq'].values   

        # aaseq 한개만 할당  
        else:
            self.aaseq = self.input_data
            self.protein_feature = profile_generation(self.aaseq, max_len=1000)
            self.smiles = filtered_data_csv['SMILES'].values
            
                
    def __len__(self):
        """
        returns the number of data from filtered dataset.
        """
        return len(self.set_id)

    def __getitem__(self, idx):
        
        # filtered
        set_id = self.set_id[idx]

        if self.input_type == 'SMILES':
            drug_feature = self.drug_feature
            protein_feature = profile_generation(self.aaseq[idx], max_len=1000)
        else:
            protein_feature = self.protein_feature
            drug_feature = extract_features(self.smiles[idx])
            
        filtered_item = (drug_feature,
                        protein_feature, set_id)
        
        return filtered_item

