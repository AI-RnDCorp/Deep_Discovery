import os
import torch
import numpy as np
import random
import json
import pandas as pd
import re
import rdkit.Chem as Chem


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def seed_everything(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    np.random.seed(random_seed)
    random.seed(random_seed)
    
def rm_files(directory):
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        try:
            if os.path.isfile(file_path):  # 파일인지 확인
                os.remove(file_path)  # 파일 삭제
                print(f"{file_path} 파일이 삭제되었습니다.")
        except Exception as e:
            print(f"파일 삭제 중 오류 발생: {e}")

    
def data_to_json(message, path):
    with open(path, 'w', encoding='utf-8') as json_file:
        json.dump(message,json_file)
    print(f"{path} has created")

# json dictionary for source/ organism category

def meta_parser(csv_path):
    original_csv = pd.read_csv(csv_path)
    unique_sources = original_csv['Curation/DataSource'].unique().tolist()
    all = original_csv['Target Source Organism According to Curator or DataSource'].unique().tolist()

    disease_related = [
    'Human immunodeficiency virus 1', 'Hepatitis C virus', 'Severe acute respiratory syndrome coronavirus 2',
    'Hepatitis C virus genotype 1a (isolate H)', 'Human SARS coronavirus', 'Influenza A virus', 'Influenza B virus (strain B/Lee/1940)',
    'Human immunodeficiency virus', 'Dengue virus 2', 'West Nile virus', 'Simian virus 40', 'Human rhinovirus B',
    'Human cytomegalovirus (strain AD169)', 'Epstein-Barr virus (strain GD1)', 'Hepatitis C virus genotype 1b (isolate BK)',
    'Hepatitis GB virus B', 'Hepatitis C virus genotype 3a (isolate NZL1)', 'Human herpesvirus 1 (strain 17)',
    'Human herpesvirus 2', 'Human herpesvirus 6A (strain Uganda-1102)', 'Human T-lymphotropic virus 1',
    'Human papillomavirus type 11', 'Escherichia coli', 'Staphylococcus aureus', 'Mycobacterium tuberculosis',
    'Streptococcus pneumoniae serotype 2 (strain D39 / NCTC 7466)', 'Bacillus anthracis', 'Pseudomonas aeruginosa', 'Mycobacterium tuberculosis H37Rv',
    'Klebsiella pneumoniae', 'Staphylococcus aureus (strain MRSA252)', 'Clostridium botulinum', 'Plasmodium falciparum',
    'Trypanosoma cruzi', 'Plasmodium falciparum (isolate FcB1 / Columbia)', 'Leishmania major', 'Leishmania donovani',
    'Schistosoma mansoni', 'Trypanosoma brucei', 'Toxoplasma gondii', 'Candida albicans', 'Cryptococcus neoformans',
    'Candida glabrata', 'Aspergillus niger', 'Neosartorya fumigata', 'Pneumocystis carinii', 'Human herpesvirus 4 type 2',
    'Legionella pneumophila'
    ]

    homosapiens = ['Homo sapiens']

    others = [organism for organism in all if organism not in disease_related and organism not in homosapiens]

    data = {'all': all, 
            'homosapiens': homosapiens, 
            'disease_related': disease_related, 
            'others': others,
            'source': unique_sources}

    with open('Target Source Organism Categories', 'w') as f:
        json.dump(data, f)




if __name__ == '__main__':
    original_path = '/Users/bioai/Documents/Datasets/BindingDB_dataset/classification_binding_DB_training.csv'

    meta_parser(original_path)