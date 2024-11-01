import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import os
from tqdm import tqdm
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import pandas as pd
from data.Dataloader import *
from utils.util import *

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm1d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
        
class Bond_Message_Block(nn.Sequential):
    def __init__(self, hidden_dim=256, hidden_dim2=512, graph_depth=3):
        super(Bond_Message_Block, self).__init__()

        self.dense_input = nn.Linear(12, hidden_dim)
        self.depth = graph_depth
        self.dense_hidden = nn.Linear(hidden_dim2, hidden_dim2)
        self.dense_output = nn.Linear(hidden_dim2, hidden_dim)


    def forward(self, bond_input, bgraph):
        
        residual = self.dense_input(bond_input)
        bond_feature = F.relu(residual)
        for i in range(self.depth):
            
            x = new_mapping_bond2nei_bond(bgraph, bond_feature)
            x = self.dense_hidden(x)
            x = x.sum(dim = 1)
            x = self.dense_output(x)
            x = x + residual
            bond_feature = F.relu(x)
        return bond_feature
    
def new_mapping_bond2nei_bond(bgraph, bond_feature):
    device = bond_feature.device
    # 같은 디바이스에 x_bond_zero 생성
    x_bond_zero = torch.zeros(1, bond_feature.shape[1], device=device)
    # bond_feature에 x_bond_zero 추가
    bond_feature = torch.cat([bond_feature, x_bond_zero], dim=0)
    # bgraph를 long 타입으로 변환하고 flatten
    idx = bgraph.flatten().to(torch.long)  # int32 대신 long 사용
    # 인덱스 범위를 bond_feature의 크기로 제한 (clamp 사용)
    idx = torch.clamp(idx, 0, bond_feature.shape[0] - 1)
    # index_select 사용
    bg_result = torch.index_select(bond_feature, 0, idx)
    # 결과 텐서의 shape 변경
    bg_result = bg_result.view(bgraph.shape[0], 6, -1)
    
    return bg_result

def new_mapping_atom2nei_atom(agraph, atom_feature):
    device = atom_feature.device
    x_atom_zero = torch.zeros(1, atom_feature.shape[1], device=device)
    atom_feature = torch.cat([atom_feature, x_atom_zero], dim=0)
    # agraph를 long 타입으로 변환하고 flatten
    ag_idx = agraph.flatten().to(torch.long)
    # 인덱스 범위를 atom_feature의 크기로 제한 (clamp 사용)
    ag_idx = torch.clamp(ag_idx, 0, atom_feature.shape[0] - 1)
    # index_select 사용
    ag_result = torch.index_select(atom_feature, 0, ag_idx)
    # 결과 텐서의 shape 변경
    ag_result = ag_result.view(agraph.shape[0], 6, -1)
    
    return ag_result

def new_mapping_atom2nei_atom_bond2(abgraph_t, x_atom_feature_hidden, x_bond_feature_hidden):
    device = abgraph_t.device

    ag = abgraph_t[:,:6]
    bg = abgraph_t[:,6:]
        
    x_bond_zero = torch.zeros(1, x_bond_feature_hidden.shape[1], device=device)
    x_bond_feature_hidden = torch.cat([x_bond_feature_hidden, x_bond_zero], dim=0)
    
    x_atom_zero = torch.zeros(1, x_atom_feature_hidden.shape[1], device=device)
    x_atom_feature_hidden = torch.cat([x_atom_feature_hidden, x_atom_zero], dim=0)
    
    ag_idx = ag.flatten().to(torch.long)
    bg_idx = bg.flatten().to(torch.long)

    # 안전한 인덱싱을 위한 클램핑
    ag_idx = torch.clamp(ag_idx, 0, x_atom_feature_hidden.shape[0] - 1)
    bg_idx = torch.clamp(bg_idx, 0, x_bond_feature_hidden.shape[0] - 1)
    
    try:
        ag_result = torch.index_select(x_atom_feature_hidden, 0, ag_idx)
        ag_result = ag_result.view(abgraph_t.shape[0], 6, -1)
        
        bg_result = torch.index_select(x_bond_feature_hidden, 0, bg_idx)
        bg_result = bg_result.view(abgraph_t.shape[0], 6, -1)
        
        result = torch.cat([ag_result, bg_result], dim=2)
        
        return result
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        print(f"ag_idx unique values: {ag_idx.unique()}")
        print(f"bg_idx unique values: {bg_idx.unique()}")
        raise


class Atom_Message_Block(nn.Sequential):
    def __init__(self, hidden_dim=256, graph_depth=3):
        super(Atom_Message_Block, self).__init__()

        self.dense_input = nn.Linear(47, hidden_dim)
        self.depth = graph_depth
        self.dense_hidden = nn.Linear(hidden_dim, hidden_dim)
        self.dense_output = nn.Linear(hidden_dim, hidden_dim)



    def forward(self, atom_input, agraph):
        
        residual = self.dense_input(atom_input)
        atom_feature = F.relu(residual)

        for i in range(self.depth):
            x = new_mapping_atom2nei_atom(agraph, atom_feature)
            x = self.dense_hidden(x)
            x = x.sum(dim = 1)
            x = self.dense_output(x)
            x = x + residual
            atom_feature = F.relu(x)    
        return atom_feature
    
class Concat_Message_Block(nn.Sequential):
    def __init__(self, hidden_dim=512):
        super(Concat_Message_Block, self).__init__()
        
        self.dense_hidden = nn.Linear(hidden_dim, hidden_dim)
        self.dense_output = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, abgraph, bond_message_feature, atom_message_feature):
        
        x = new_mapping_atom2nei_atom_bond2(abgraph, atom_message_feature, bond_message_feature)
        
        x = self.dense_hidden(x)
        x = x.sum(dim = 1)
        x = self.dense_output(x)
        concat_message_feature = F.relu(x)
        
        return concat_message_feature


class ABCMPN(nn.Sequential):
    
    def __init__(self, Atom_Message_Block, Bond_Message_Block, Concat_Message_Block):
        super(ABCMPN, self).__init__()
        
        self.AM_block = Atom_Message_Block
        self.BM_block = Bond_Message_Block
        self.CM_block = Concat_Message_Block    
    def to(self, device):
        self.AM_block = self.AM_block.to(device)
        self.BM_block = self.BM_block.to(device)
        self.CM_block = self.CM_block.to(device)
        return super(ABCMPN, self).to(device)

    def forward(self, feature_list):
        """ 
        feature_list = [Bond_block_input, Atom_block_input, bgraph, abgraph, agraph])
        
        """        
        # device = feature_list[1].device
        device = next(self.parameters()).device
        batch_total_y = torch.Tensor([]).to(device)
        
        for item in feature_list:
            bond_input, atom_input, bgraph, abgraph, agraph = item
            bond_input = bond_input[:, :12]
            Atom_message = self.AM_block(atom_input, agraph)           
            Bond_message = self.BM_block(bond_input, bgraph)
            
            # 1. output y : atom_num x mpnn_feature_dim
            Concat_message = self.CM_block(abgraph, Bond_message, Atom_message)
            G = Concat_message
            
            W_att = F.softmax(torch.matmul(G, G.T), dim=0)
            attention_value = torch.matmul(W_att, G)
            Concat_message = G + attention_value
            
            y = torch.mean(Concat_message, 0).view(1,-1)
            # 4. batch 개수 대로 concat해서 output 형태로 만들기. (GPU 서버 생각하면서 알고리즘 제작)
            
            batch_total_y = torch.cat([batch_total_y, y], dim=0)
        
        return batch_total_y
    
class Protein_model_CNN(nn.Module):
    def __init__(self):
        super(Protein_model_CNN, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels=26, out_channels=64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool1d(kernel_size=2)
        
        self.fc = nn.Linear(256, 256)
    def to(self, device):
        return super(Protein_model_CNN, self).to(device)


    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        
        x = F.relu(self.conv3(x))
        x = self.pool3(x)
        
        x = F.adaptive_max_pool1d(x, output_size=1)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x
    
    

#drug_model = ABCMPN(Atom_Message_Block(), Bond_Message_Block(),Concat_Message_Block())


class Docking_model_MLP(nn.Module):
    def __init__(self, drug_model, ptn_model, use_dropout = False, Ddim=512, Pdim=256):
        super(Docking_model_MLP, self).__init__()

        self.drug_model = drug_model
        self.ptn_model = ptn_model
        
        self.Ddim = Ddim
        self.Pdim = Pdim
        
        docking_model = []
        docking_model += [nn.Linear(self.Ddim + self.Pdim, 256), nn.BatchNorm1d(256), nn.ReLU(),
                        nn.Linear(256,128), nn.BatchNorm1d(128), nn.ReLU()]
        
        if use_dropout:
            docking_model += [nn.Dropout(0.5)]
        
        docking_model += [nn.Linear(128, 1)]
        self.docking_model = nn.Sequential(*docking_model)
    
    def to(self, device):
        self.drug_model = self.drug_model.to(device)
        self.ptn_model = self.ptn_model.to(device)
        self.docking_model = self.docking_model.to(device)
        return super(Docking_model_MLP, self).to(device)
        
    def forward(self, drug_input, ptn_input):
        drug_vec = self.drug_model(drug_input)
        ptn_vec = self.ptn_model(ptn_input)
        
        feature_vector = torch.cat([drug_vec, ptn_vec], dim=1)
        
        output = self.docking_model(feature_vector)
        
        return output  # 시그모이드 함수 적용

def define_drug_model():
    
    drug_model =  ABCMPN(Atom_Message_Block(), Bond_Message_Block(),Concat_Message_Block())
    drug_model.apply(weights_init)
            
    return drug_model

def define_protein_model():
    
    ptn_model =  Protein_model_CNN()
    ptn_model.apply(weights_init)
            
    return ptn_model


def define_Docking_model(drug_model, ptn_model, use_dropout=False):
    
    Docking_model = Docking_model_MLP(drug_model= drug_model, ptn_model = ptn_model, use_dropout = use_dropout)    
    Docking_model.apply(weights_init)
            
    return Docking_model


def optimizer_setup(docking_model, opt):  # 
    optimizer = torch.optim.Adam(docking_model.parameters(), lr =opt.lr, weight_decay=opt.weight_decay)
    return optimizer

def define_deepdiscovery_model(opt):
    drug_model, protein_model = define_drug_model(), define_protein_model()
    docking_model = define_Docking_model(drug_model, protein_model)
    docking_model = docking_model
    return docking_model


def define_deepdiscovery_initializer(opt, device, model_path=None):
    drug_model, protein_model = define_drug_model(), define_protein_model()
    docking_model = define_Docking_model(drug_model, protein_model)
    docking_model = docking_model.to(device)
    optimizer = optimizer_setup(docking_model, opt)
    if opt.gpu_ids[0] >=0:
        device_str = 'cuda'
    else:
        device_str = 'cpu'
    
    if opt.isTrain:
        if opt.start_epoch != 0:
            print('모델을 체크포인트에서 로드합니다: {}'.format(model_path))
            #device_str = 'cuda' if torch.cuda.is_available() else 'cpu'

            checkpoint = torch.load(model_path, map_location=device_str)
            docking_model.load_state_dict(checkpoint['docking_model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        else:
            pass
    else: # test or inference
        print('모델을 체크포인트에서 로드합니다: {}'.format(model_path), 'device:', device)
        checkpoint = torch.load(model_path, map_location=device_str)
        docking_model.load_state_dict(checkpoint['docking_model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return docking_model, optimizer

# 모델 저장
def model_save(save_path, docking_model, optimizer, epoch):
    model_path = os.path.join(save_path, f"weight_{epoch}.pt")
    
    print('모델을 저장합니다: {}'.format(model_path))
    torch.save({
        'docking_model_state_dict': docking_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, model_path)
    

def train_epoch(opt, model, data_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    total_accuracy = 0.0
    
    pbar = tqdm(data_loader, desc='Training', unit="batch")
    
    epoch_start_time = time.time()

    for i, data in enumerate(pbar):
        df, pf, gt = data
        df = [[d.to(device) if isinstance(d, torch.Tensor) else d for d in sublist] for sublist in df]
        pf, gt = pf.to(device), gt.to(device)
        
        optimizer.zero_grad()
        output = model(df, pf)
        loss = criterion(output, gt.float().view(-1, 1))
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        predictions = (torch.sigmoid(output) > 0.5).float()
        total_accuracy += accuracy_score(gt.cpu().numpy(), predictions.cpu().numpy())
        
        pbar.set_postfix({'Loss': f'{loss.item():.4f}'})

    avg_loss = total_loss / len(data_loader)
    avg_accuracy = total_accuracy / len(data_loader)
    elapsed_time = time.time() - epoch_start_time

    print(f"Loss: {avg_loss:.4f}")
    print(f"Accuracy: {avg_accuracy:.4f}")
    print(f"Time taken: {elapsed_time :.2f} seconds")
    print("------------------------")
    return avg_loss, avg_accuracy, elapsed_time


def evaluate(opt, model, data_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_predictions = []
    all_labels = []
    
    pbar = tqdm(data_loader, desc='Validation', unit='batch')
    
    epoch_start_time = time.time()

    with torch.no_grad():
        for i, data in enumerate(pbar):
            df, pf, gt = data
            df = [[d.to(device) if isinstance(d, torch.Tensor) else d for d in sublist] for sublist in df]
            pf, gt = pf.to(device), gt.to(device)
            
            output = model(df, pf)
            loss = criterion(output, gt.float().view(-1, 1))
            
            total_loss += loss.item()
            predictions = (torch.sigmoid(output) > 0.5).float()
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(gt.cpu().numpy())
    
    avg_loss = total_loss / len(data_loader)
    accuracy = accuracy_score(all_labels, all_predictions)
    elapsed_time = time.time() - epoch_start_time

    print(f"Validation Loss: {avg_loss:.4f}")
    print(f"ValidationAccuracy: {accuracy:.4f}")
    print(f"Time taken: {elapsed_time:.2f} seconds")
    print("------------------------")

    return avg_loss, accuracy, elapsed_time

def test(opt, model, data_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_predictions = []
    all_labels = []
    all_probabilities = []
    
    pbar = tqdm(data_loader, desc='테스트', unit='batch')
    
    test_start_time = time.time()

    with torch.no_grad():
        for i, data in enumerate(pbar):
            df, pf, gt = data
            df = [[d.to(device) if isinstance(d, torch.Tensor) else d for d in sublist] for sublist in df]
            pf, gt = pf.to(device), gt.to(device)
            
            output = model(df, pf)
            loss = criterion(output, gt.float().view(-1, 1))
            
            total_loss += loss.item()
            probabilities = torch.sigmoid(output)
            predictions = (probabilities > 0.5).float()
            
            all_predictions.extend(predictions.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
            all_labels.extend(gt.cpu().numpy())
    
    avg_loss = total_loss / len(data_loader)
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions)
    recall = recall_score(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions)
    auc = roc_auc_score(all_labels, all_probabilities)

    print(f"테스트 결과:")
    print(f"Loss: {avg_loss:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"AUC: {auc:.4f}")
    print(f"소요 시간: {time.time() - test_start_time:.2f} 초")

    # 결과를 DataFrame으로 저장
    results_df = pd.DataFrame({
        'True Labels': all_labels,
        'Predictions': all_predictions,
        'Probabilities': all_probabilities
    })
    results_df.to_csv('{}_test_results.csv'.format(opt.experiment_name), index=False)

    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc
    }


