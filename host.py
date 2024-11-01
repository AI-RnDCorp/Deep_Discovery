
import os
from torch import nn
import pandas as pd
import numpy as np
from data.BindingDB_dataset import *
from options.base_options import BaseOptions
from data.Dataloader import *
from utils.optimized_feature_extractor import *
from utils.util import *
from model.drug_discovery import *
from utils.structure_parser import *
import torch
from tqdm import tqdm
import os
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from inference import *
import json




"""
Initialization
"""
#opt = BaseOptions().parse()
# 모델 불러오기
#deepdiscovery = DeepDiscovery(opt)

def run(deepdiscovery):
    file_cache = []
    
    while True:
        # 현재 디렉토리의 파일 목록 가져오기
        files = os.listdir(input_directory)
        files = [file for file in files if 'req' in file]
        time.sleep(1)
        print(f"현재 input requests 수: {len(files)}")
        if len(files) >=1: # 파일이 한개 이상일 경우
        # 새로운 파일이 있는지 확인
            new_files = set(files) - set(file_cache)
            
            if new_files:  # 새로운 파일이 있을 경우
                new_file = new_files.pop()  # 새로운 파일 하나 가져오기
                file_cache.append(new_file)
                
                new_file_path = os.path.join(input_directory, new_file)
                # 새로운 파일을 캐시에 추가
                
                # 새로운 파일 들어올떄 오픈
                with open(new_file_path, 'r') as file:
                    data = json.load(file)

                inference_inputs = deepdiscovery.input_wrapper(data['input_type'], data['input_string'], data['category'], data['temp'], data['pH'])

                # inference_inputs가 None인지 확인
                if inference_inputs is None:
                    error_message = {'error_type':'error 1'} # invalid input data
                    error_path = os.path.join(error_directory, new_file+'_error')
                    data_to_json(error_message, error_path ) 
        
                else:
                    """
                    h100 전달용
                    """
                    h100_input_path = os.path.join(input_h100, new_file)
                    data_to_json(data, h100_input_path) 
                    database_csv, filtered_data_csv = deepdiscovery.filtered_csv(inference_inputs)
                    if len(filtered_data_csv) == 0:
                        """
                        실제론 error 2는 발생하지 않음.
                        """
                        error_message = {'error_type':'error 2'} # number of filtered data is 0
                        error_path = os.path.join(error_directory, new_file + '_error2')
                        data_to_json(error_message, error_path)
                    else: 
                        number_of_filtered_data = len(filtered_data_csv)
                        
                        # 프로그레스바 | 결과 |
                        if number_of_filtered_data//512 < 2:
                            expected_time = 90
                        else:
                            expected_time = number_of_filtered_data/512 * 2.5 + 90 # for progress input올리는시간 + inference 시간 + result, report, result  가져오는 시간 계산. 배치당 평균 1.4 s
                        message = {'number of files':number_of_filtered_data, 'expected_time': expected_time}
                        number_of_data_path = os.path.join(filtering_files, new_file + '_files')
                        data_to_json(message, number_of_data_path)
                        
                        """
                        run input
                        Model Inference
                        """
                        
                        final_result_csv, sorted_result_csv, time_with_month, category_name, elapsed_time, plp, batch_min, accuracy = deepdiscovery.inference_run(inference_inputs, database_csv, filtered_data_csv)

                        print('Inference is completed!')
                        report_result = {'time': time_with_month, 'category': category_name,
                        'processed_data': len(filtered_data_csv), 'elapsed_time': elapsed_time, 
                        'plp': plp, 'batch_min': batch_min, 'accuracy': accuracy}

                        report_result_save = os.path.join(report_result_path, new_file+ '_report')
                        data_to_json(report_result, report_result_save)
                        """
                        """
                        # 10개 항목 저장
                        report_sorted_csv_path = os.path.join(report_sorted_csv, new_file + '_sorted.csv')
                        sorted_result_csv.to_csv(report_sorted_csv_path, index=False)
                        
                        # 저장 csv # aaseq일 경우
                        results_csv_path = os.path.join(results_csv, new_file+'_result.csv')
                        final_result_csv.to_csv(results_csv_path, index=False)
                

                        go_files = os.listdir(input_go)
                        
                        input_go_cache = []

                        check = True
                        while check:
                            check_new = os.listdir(input_directory)
                            check_new_files = [file for file in check_new if 'req' in file]
                            check_new = set(check_new_files) - set(file_cache)
                            
                            if check_new:
                                check = False
                                rm_files(input_go)
                                rm_files(go_structure)
                            
                            go_files = os.listdir(input_go)
                            go_files = [file for file in go_files if 'req' in file]
                            go_new = set(go_files) - set(input_go_cache)                               
                            """
                            3D structure
                            """
                            if go_new:
                                go_file = go_new.pop()
                                input_go_cache.append(go_file)
                                go_file_path = os.path.join(input_go, go_file)
                                
                                with open(go_file_path, 'r') as file:
                                    go_data = json.load(file)
                                go_string = go_data['input_string']
                                if inference_inputs[0] == 'SMILES':
                                    # AAseq 정보 로드
                                    location = deepdiscovery.data_csv.index[deepdiscovery.data_csv['AAseq'] == go_string].tolist()[0]
                                    pdb = deepdiscovery.data_csv['PDB ID(s) of Target Chain'].iloc[location]
                                    pdb = pdb.split(',')
                                    structure_result = process_aaseq_data(pdb, top_n=5)
                                    
                                else:
                                    # SMILES 정보 로드
                                    Inchikey = go_string
                                    structure_result = process_inchikey_data(Inchikey, rank=1)
                                structure_result_path = os.path.join(go_structure ,go_file+'_'+len(input_go_cache)+'_structure')
                                data_to_json(structure_result, structure_result_path)
                                print(f"json has saved from {structure_result_path}")


if __name__ == '__main__': # adain update
    # initialization #
        
    home = '/home/elicer/data/req'
    # inputs 
    input_directory = os.path.join(home, 'input')
    input_go = os.path.join(home, 'input_go')

    error_directory = os.path.join(home, 'error')
    filtering_files = os.path.join(home, 'filtering_files')
    # results
    results_csv = os.path.join(home, 'results_csv')
    report_result_path = os.path.join(home, 'report_results') # 용량이 큼.
    report_sorted_csv = os.path.join(home, 'report_sorted_csv')
    go_structure = os.path.join(home, 'go_structure')

    ### h100 path
    input_h100 = os.path.join(home, 'input_h100')

    mkdir(input_directory)
    mkdir(input_go)
    mkdir(error_directory)
    mkdir(filtering_files)
    mkdir(results_csv)
    mkdir(report_result_path)
    mkdir(report_sorted_csv)
    mkdir(input_h100)
    mkdir(go_structure)



    opt  = BaseOptions().parse()
    # 모델 불러오기
    deepdiscovery = DeepDiscovery(opt)
    
    run(deepdiscovery)