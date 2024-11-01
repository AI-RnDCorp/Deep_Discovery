import torch
import numpy as np
import rdkit
import rdkit.Chem as Chem
from tqdm import tqdm
import time
import pickle
import pandas as pd
import os
from numba import jit, prange, njit



# atom & bond feature extractor
def mol_generate(smiles):
    """
    convert smiles into mol
    """
    mol = Chem.MolFromSmiles(smiles)
    Chem.Kekulize(mol)
    return mol


def Bond_feature(bond): 
    """
    bond feature one-hot vector generation
    
    input: bond ex) make.GetBonds()
    output: torch.Size([12]) tensor
    """
    bond_types = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]
    stereos = [0, 1, 2, 3, 4, 5]
    rings = [0, 1]
    
    bond_features = [bond.GetBondType(), int(bond.GetStereo()), bond.IsInRing()]
    
    vectors = []
    for i, feature in enumerate([bond_types, stereos, rings]):
        one_hot = torch.zeros(len(feature))
        if bond_features[i] in feature:
            index = feature.index(bond_features[i])
        else: # for the case other characteristics
            index = 0
        one_hot[index] = 1
        vectors.append(one_hot)
    
    return torch.cat(vectors, dim = 0).to(torch.float32)


def Atom_feature(atom):
    """
    atom feature one-hot vector generation
    
    input: atm ex) mol.GetBonds().GetBeginAtom(), mol.GetBonds().GetEndAtom(), mol.GetAtoms()
    output: torch.Size([47]) tensor
    """
    
    atom_char = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'Al', 'I', 'B', 'K', 'Se', 'Zn', 'H', 'Cu', 'Mn', 'unknown']
    hybrids = [rdkit.Chem.rdchem.HybridizationType.S, rdkit.Chem.rdchem.HybridizationType.SP, rdkit.Chem.rdchem.HybridizationType.SP2, rdkit.Chem.rdchem.HybridizationType.SP3, rdkit.Chem.rdchem.HybridizationType.SP3D, rdkit.Chem.rdchem.HybridizationType.SP3D2]
    formal_charges = [-2, -1, 0, 1, 2]
    degrees = [0, 1, 2, 3, 4, 5, 6]
    chirals = [0, 1, 2, 3]
    aromatics = [0, 1]
    
    feature_standard = [atom_char, hybrids, formal_charges, degrees, chirals, aromatics]
    
    atom_features = [atom.GetSymbol(), atom.GetHybridization(), atom.GetFormalCharge(), atom.GetDegree(), atom.GetChiralTag(), atom.GetIsAromatic()]
    
    vectors = []
    for i, feature in enumerate(feature_standard):
        one_hot = torch.zeros(len(feature))
        if atom_features[i] in feature:
            index = feature.index(atom_features[i])
        else: # for the case other characteristics
            index = 0
        one_hot[index] = 1
        vectors.append(one_hot)
    
    return torch.cat(vectors, dim = 0).to(torch.float32)


def Bond_index2Bond_Atom_feature(smiles):
    mol = mol_generate(smiles)
    bonds = mol.GetBonds()
    features = []
    for bond in bonds:
        atom_first = bond.GetBeginAtom()
        atom_second = bond.GetEndAtom()
        bond_feat = Bond_feature(bond)
        features.append(torch.cat([bond_feat, Atom_feature(atom_first)]))
        features.append(torch.cat([bond_feat, Atom_feature(atom_second)]))
    return torch.stack(features).to(torch.float32)

def Atom_index2Atom_feature(smiles):
    mol = mol_generate(smiles)
    return torch.stack([Atom_feature(atom) for atom in mol.GetAtoms()])

def Bond_relation(smiles):
    mol = mol_generate(smiles)
    bonds = mol.GetBonds()
    relations = []
    for bond in bonds:
        begin, end = bond.GetBeginAtom().GetIdx(), bond.GetEndAtom().GetIdx()
        relations.extend([[begin, end], [end, begin]])
    return torch.tensor(relations, dtype=torch.long)

# @njit(parallel=True)
# def making_bgraph_numba(bond_relation, n_bonds, max_neighbors=6):
#     bgraph_t = np.ones((n_bonds, max_neighbors), dtype=np.int64) * n_bonds

#     for bond_index in prange(n_bonds):
#         cnt = 0
#         for i in range(bond_relation.shape[0]):
#             if bond_relation[bond_index, 0] == bond_relation[i, 1]:
#                 if cnt < max_neighbors:
#                     bgraph_t[bond_index, cnt] = i
#                     cnt += 1
#                 else:
#                     break
#     return bgraph_t


def making_bgraph(Bond_block_input, bond_relation):
    bgraph = making_abgraph_numba(bond_relation.numpy(), Bond_block_input.shape[0])
    return torch.from_numpy(bgraph).to(torch.int64)

# @jit(nopython=True, parallel=True)
# def making_abgraph_numba(bond_relation, n_atoms, max_neighbors=12):
#     abgraph_t = np.ones((n_atoms, max_neighbors), dtype=np.int16) * n_atoms
#     for atom_index in prange(n_atoms):
#         cnt_atom, cnt_bond = 0, 6
#         for i in range(bond_relation.shape[0]):
#             if atom_index == bond_relation[i, 1]:
#                 if cnt_atom < 6:
#                     abgraph_t[atom_index, cnt_atom] = bond_relation[i, 0]
#                     cnt_atom += 1
#                 if cnt_bond < 12:
#                     abgraph_t[atom_index, cnt_bond] = i
#                     cnt_bond += 1
#     return abgraph_t


@njit(parallel=True)
def making_abgraph_numba(bond_relation, n_atoms, max_neighbors=12):
    abgraph = np.zeros((n_atoms, max_neighbors), dtype=np.int64)
    neighbor_count = np.zeros(n_atoms, dtype=np.int64)
    
    for i in prange(bond_relation.shape[0]):
        atom1 = bond_relation[i, 0]
        atom2 = bond_relation[i, 1]
        
        if neighbor_count[atom1] < max_neighbors:
            abgraph[atom1, neighbor_count[atom1]] = atom2
            neighbor_count[atom1] += 1
        
        if neighbor_count[atom2] < max_neighbors:
            abgraph[atom2, neighbor_count[atom2]] = atom1
            neighbor_count[atom2] += 1
    
    return abgraph

def making_abgraph(Atom_block_input, bond_relation):
    abgraph = making_abgraph_numba(bond_relation.numpy(), Atom_block_input.shape[0])
    return torch.from_numpy(abgraph).to(torch.int64)

def making_agraph(abgraph):
    return abgraph[:, :6].to(torch.float32)

# Protein feature extractor optimization
amino_char = 'ACBEDGFIHKMLONQPSRUTWVYXZ?'
amino_dict = {c: i for i, c in enumerate(amino_char)}

def profile_generation(ptn_seq, max_len=1000):
    ptn_seq = ptn_seq.upper()[:max_len]
    ptn_seq = ptn_seq.ljust(max_len, '?')
    indices = np.array([amino_dict.get(c, 25) for c in ptn_seq], dtype=np.int8)
    profile = np.eye(26, dtype=np.float16)[indices].T
    return torch.from_numpy(profile).to(torch.float32)


def feature_extractor(file_path, save_dir = None):
    data_csv = pd.read_csv(file_path)
    x_drug, x_ptn, y, index = data_csv['SMILES'], data_csv['AAseq'], data_csv['IC50_label'], data_csv['BindingDB Reactant_set_id'].tolist()
    
    drug_features, protein_features, labels = [], [], []
    start_time = time.time()

    print("----Generating drug, protein features----")

    for idx in tqdm(range(len(x_drug))):
        # drug features
        Bond_block_input = Bond_index2Bond_Atom_feature(x_drug.iloc[idx])
        Atom_block_input = Atom_index2Atom_feature(x_drug.iloc[idx])
        bond_relation = Bond_relation(x_drug.iloc[idx])
        bgraph = making_bgraph(Bond_block_input, bond_relation)
        abgraph = making_abgraph(Atom_block_input, bond_relation)
        agraph = making_agraph(abgraph)

        # protein features
        x_ptn_profile = profile_generation(x_ptn.iloc[idx], max_len=1000)
        
        drug_features.append([Bond_block_input, Atom_block_input, bgraph, abgraph, agraph])
        protein_features.append(x_ptn_profile)
        labels.append(torch.tensor(y.iloc[idx]))

    print("----Features are Extracted----")

    end_time = time.time()
    execution_time = end_time - start_time

    print(f"시작 시간: {time.ctime(start_time)}")
    print(f"종료 시간: {time.ctime(end_time)}")
    print(f"실행 시간: {execution_time:.2f} 초")

    # pickle generate
    data_dict = {
        'drug_features': drug_features,
        'protein_features': protein_features,
        'labels': labels,
        'index': index
    }
    if save_dir:
        save_dir = os.path.join(save_dir, os.path.splitext(os.path.basename(file_path))[0] + '.pkl')
        with open(save_dir, 'wb') as f:
            pickle.dump(data_dict, f)  
        print(f"데이터가 {save_dir}에 저장되었습니다.")

if __name__ == '__main__':
    train_path = '/Users/bioai/Documents/GitHub/BioAI/classification_binding_DB_training_0.9.csv'
    test_path = '/Users/bioai/Documents/GitHub/BioAI/classification_binding_DB_test_0.1.csv'
    save_path = '/Users/bioai/Documents/Datasets/BindingDB_dataset/features'
    
    #feature_extractor(test_path, save_path)    
    feature_extractor(train_path, save_path)
