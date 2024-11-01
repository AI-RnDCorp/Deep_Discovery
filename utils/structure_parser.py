
from typing import Optional, Dict, Any
import requests
# Importing third-party libraries
from rdkit import Chem
from rdkit.Chem import inchi, AllChem
import json
from functools import lru_cache  

def get_smiles_info(inchi_key: str) -> Optional[Dict[str, Any]]:
    """InChIKey를 이용해 PubChem에서 SMILES 정보 가져오기"""
    pubchem_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/inchikey/{inchi_key}/JSON"
    try:
        response = requests.get(pubchem_url)
        if response.status_code == 200:
            data = response.json()
            compound_info = data['PC_Compounds'][0]
            smiles_info = {
                "Compound ID": compound_info['id']['id']['cid'],
                "SMILES": next(
                    (prop['value']['sval'] for prop in compound_info['props']
                     if prop['urn'].get('label') == 'SMILES' and prop['urn'].get('name') == 'Canonical'), None),
                "Properties": {
                    "Canonicalized": next(
                        (prop['value']['ival'] for prop in compound_info['props']
                         if prop['urn'].get('label') == 'Compound' and prop['urn'].get('name') == 'Canonicalized'), None),
                    "Compound Complexity": next(
                        (prop['value']['fval'] for prop in compound_info['props']
                         if prop['urn'].get('label') == 'Compound Complexity'), None),
                    "Hydrogen Bond Acceptor": next(
                        (prop['value']['ival'] for prop in compound_info['props']
                         if prop['urn'].get('name') == 'Hydrogen Bond Acceptor'), None),
                    "Hydrogen Bond Donor": next(
                        (prop['value']['ival'] for prop in compound_info['props']
                         if prop['urn'].get('name') == 'Hydrogen Bond Donor'), None),
                    "Rotatable Bond": next(
                        (prop['value']['ival'] for prop in compound_info['props']
                         if prop['urn'].get('name') == 'Rotatable Bond'), None),
                    "Log P (XLogP3-AA)": next(
                        (prop['value']['fval'] for prop in compound_info['props']
                         if prop['urn'].get('label') == 'Log P'), None),
                    "Exact Mass": next(
                        (prop['value']['sval'] for prop in compound_info['props']
                         if prop['urn'].get('label') == 'Mass' and prop['urn'].get('name') == 'Exact'), None),
                    "Molecular Formula": next(
                        (prop['value']['sval'] for prop in compound_info['props']
                         if prop['urn'].get('label') == 'Molecular Formula'), None),
                    "Molecular Weight": next(
                        (prop['value']['sval'] for prop in compound_info['props']
                         if prop['urn'].get('label') == 'Molecular Weight'), None)
                },
                "IUPAC Names": {
                    "Preferred": next(
                        (prop['value']['sval'] for prop in compound_info['props']
                         if prop['urn'].get('label') == 'IUPAC Name' and prop['urn'].get('name') == 'Preferred'), None),
                    "Systematic": next(
                        (prop['value']['sval'] for prop in compound_info['props']
                         if prop['urn'].get('label') == 'IUPAC Name' and prop['urn'].get('name') == 'Systematic'), None)
                },
                "InChI Information": {
                    "InChI": next(
                        (prop['value']['sval'] for prop in compound_info['props']
                         if prop['urn'].get('label') == 'InChI'), None),
                    "InChIKey": next(
                        (prop['value']['sval'] for prop in compound_info['props']
                         if prop['urn'].get('label') == 'InChIKey'), None)
                },
                "Full JSON": data
            }
            return smiles_info
    except Exception as e:
        print(f'Error retreiving PubChem data: {e}')
        #logger.error(f"Error retrieving PubChem data: {e}")
    return None

def generate_3d_structure(smiles: str):
    """SMILES로부터 3D 구조를 생성하고 MolBlock 형식으로 반환"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError("유효하지 않은 SMILES 문자열입니다.")
        mol = Chem.AddHs(mol)
        result = AllChem.EmbedMolecule(mol, AllChem.ETKDG())
        if result != 0:
            raise ValueError("3D 구조 생성 실패: EmbedMolecule 실패.")
        AllChem.UFFOptimizeMolecule(mol)
        return Chem.MolToMolBlock(mol)
    except Exception as e:
        print(f"3D 구조 생성 오류: {e}")
        #logger.error(f"3D 구조 생성 오류: {e}")
    return None

def process_inchikey_data(inchi_key: str, rank: int = 1):
    """InChIKey를 입력으로 받아 관련 정보를 조회하고 JSON 파일로 저장"""
    smiles_info = get_smiles_info(inchi_key)
    if smiles_info:
        smiles = smiles_info.get("SMILES")
        if smiles:
            mol_block = generate_3d_structure(smiles)
            if mol_block:
                result = {
                    "ranking": rank,
                    "InChIKey": inchi_key,
                    "Compound ID": smiles_info.get("Compound ID"),
                    "SMILES": smiles,
                    "Properties": smiles_info.get("Properties"),
                    "IUPAC Names": smiles_info.get("IUPAC Names"),
                    "InChI Information": smiles_info.get("InChI Information"),
                    "3D Structure": mol_block
                }
                return result
            #    save_to_json(result, f"inchikey_info_rank_{rank}_{inchi_key}")
            else:
                None
                #logger.warning(f"3D 구조를 생성할 수 없습니다: {smiles}")
        else:
            None
            #logger.warning(f"SMILES 정보를 찾을 수 없습니다 for InChIKey: {inchi_key}")
    else:
        None
    # logger.warning(f"SMILES 정보를 가져올 수 없습니다 for InChIKey: {inchi_key}")
    
    
    
@lru_cache(maxsize=1000)
def get_pdb_info(pdb_id: str) -> Optional[Dict[str, Any]]:
    """PDB ID를 통해 RCSB에서 PDB 정보 가져오기"""
    url = f"https://data.rcsb.org/rest/v1/core/entry/{pdb_id}"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            return {
                "PDB ID": pdb_id,
                "Researcher Information": [{"name": author.get("name", "N/A")} for author in data.get('audit_author', [])],
                "Cell Information": {
                    "angle_alpha": data.get("cell", {}).get("angle_alpha", "N/A"),
                    "angle_beta": data.get("cell", {}).get("angle_beta", "N/A"),
                    "angle_gamma": data.get("cell", {}).get("angle_gamma", "N/A"),
                    "length_a": data.get("cell", {}).get("length_a", "N/A"),
                    "length_b": data.get("cell", {}).get("length_b", "N/A"),
                    "length_c": data.get("cell", {}).get("length_c", "N/A"),
                    "zpdb": data.get("cell", {}).get("zpdb", "N/A")
                },
                "Citation": [
                    {
                        "title": citation.get("title", "N/A"),
                        "journal": citation.get("journal_abbrev", "N/A"),
                        "year": citation.get("year", "N/A")
                    }
                    for citation in data.get("citation", [])
                ],
                "Resolution and Methodology": {
                    "resolution": data.get('rcsb_entry_info', {}).get("resolution_combined", ["N/A"])[0],
                    "ls_rfactor_obs": data.get('refine', [{}])[0].get("ls_rfactor_obs", "N/A"),
                    "deposited_atom_count": data.get('rcsb_entry_info', {}).get("deposited_atom_count", "N/A"),
                    "molecular_weight": data.get('rcsb_entry_info', {}).get("molecular_weight", "N/A")
                },
                "Primary Citation": {
                    "title": data.get('rcsb_primary_citation', {}).get("title", "N/A"),
                    "journal": data.get('rcsb_primary_citation', {}).get("journal_abbrev", "N/A"),
                    "year": data.get('rcsb_primary_citation', {}).get("year", "N/A")
                }
            }
    except Exception as e:
        logger.error(f"Error retrieving PDB info for {pdb_id}: {e}")
    return None

def visualize_pdb_structure(pdb_id: str) -> Optional[str]:
    """PDB ID로부터 3Dmol.js에서 사용할 수 있는 PDB 내용을 가져옴"""
    try:
        url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
        response = requests.get(url)
        if response.status_code == 200:
            return response.text
        else:
            print(f"PDB 파일을 가져올 수 없습니다: {pdb_id}")
            #logger.error(f"PDB 파일을 가져올 수 없습니다: {pdb_id}")
    except Exception as e:
        #logger.error(f"PDB 시각화 오류: {e}")
        print(f"PDB 시각화 오류: {e}")
    return None

def safe_float(value, default=float('inf')):
    """문자열 값을 안전하게 float로 변환."""
    try:
        return float(value)
    except (ValueError, TypeError):
        return default

def process_aaseq_data(pdb_id_list: list, top_n=5):
    """PDB ID 리스트를 처리하고 상위 N개의 결과를 JSON 파일로 저장"""
    pdb_info_list = []
    
    for pdb_id in pdb_id_list:
        pdb_info = get_pdb_info(pdb_id.strip())
        pdb_content = visualize_pdb_structure(pdb_id.strip())
        if pdb_info:
            pdb_info['PDB Content'] = pdb_content if pdb_content else "Unavailable"
            pdb_info_list.append(pdb_info)
        else:
            print(f"PDB 정보를 가져올 수 없습니다: {pdb_id}")
            #logger.error(f"PDB 정보를 가져올 수 없습니다: {pdb_id}")
    
    if not pdb_info_list:
        print("처리할 PDB 정보가 없습니다.")
   #     logger.warning("처리할 PDB 정보가 없습니다.")
        return
    
    # PDB 정보 정렬
    sorted_pdbs = sorted(
        pdb_info_list,
        key=lambda x: (
            safe_float(x['Resolution and Methodology'].get('resolution', float('inf'))),
            -int(x['Primary Citation'].get('year', '0')) if str(x['Primary Citation'].get('year', '0')).isdigit() else 0,
            safe_float(x['Resolution and Methodology'].get('ls_rfactor_obs', float('inf'))),
            -x['Resolution and Methodology'].get('deposited_atom_count', 0)
        )
    )[:top_n]
    
    # 결과를 DataFrame으로 변환
    results = []
    for rank, pdb in enumerate(sorted_pdbs, start=1):
        results.append({
            "ranking": rank,
            "PDB ID": pdb["PDB ID"],
            "Researcher Information": "; ".join([author["name"] for author in pdb["Researcher Information"]]),
            "Cell Information": json.dumps(pdb["Cell Information"], ensure_ascii=False),
            "Citation": json.dumps(pdb["Citation"], ensure_ascii=False),
            "Resolution and Methodology": json.dumps(pdb["Resolution and Methodology"], ensure_ascii=False),
            "Primary Citation": json.dumps(pdb["Primary Citation"], ensure_ascii=False),
            "PDB Content": pdb["PDB Content"]
        })
    
    # JSON 파일로 저장
    return results
    #save_to_json(results, f"pdb_info_top_{top_n}")