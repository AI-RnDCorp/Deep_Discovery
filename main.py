import os
os.environ['KMP_WARNINGS'] = 'OFF'
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
from options.base_options import BaseOptions
from data.Dataloader import *
from tqdm import tqdm
from torch import nn
from model.drug_discovery import *
from utils.util import *
import wandb


def run(opt):   
    
    if opt.gpu_ids:
        device = opt.gpu_ids[0]
    else:
        device = 'cpu'
    print("device:", device)


    print('---------- Dataset initialized -------------')

    data_loader = BD_dataloader(opt)

    print('---------- Networks initialized -------------')    
    # 0: 66% 1: 33%
    if opt.weighted_loss:
        neg_weight, pos_weight = 0.66, 0.33
        weight = pos_weight / neg_weight
        pos_weight = torch.tensor([weight]).to(device)

        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        criterion = nn.BCEWithLogitsLoss()
    
    if opt.isTrain:
        save_path = os.path.join(opt.save_dir, opt.experiment_name)
        mkdir(save_path)
        
        docking_model, optimizer = define_deepdiscovery_initializer(opt, device, opt.weight_pt)
        train_loader = data_loader.get_train_loader()
        if opt.split_size:
            val_loader = data_loader.get_val_loader()    
        
        print('---------- Model Training--------------')
        
        train_epoch_losses, train_epoch_accuracy, validation_epoch_losses, validation_epoch_accuracy = [], [], [], []

        for epoch in range(opt.start_epoch, opt.n_epochs):                   
            print(f"Epoch {epoch+1}/{opt.n_epochs}")
            # train
            avg_loss, avg_accuracy, train_elapsed_time  = train_epoch(opt, docking_model, train_loader, criterion, optimizer, device)
            train_epoch_losses.append(avg_loss)
            train_epoch_accuracy.append(avg_accuracy)     
        
            # val
            if opt.split_size:
                print('---------- Validation--------------')
                val_avg_loss, val_accuracy, val_elapsed_time = evaluate(opt, docking_model, val_loader, criterion, device)
                validation_epoch_losses.append(val_avg_loss)
                validation_epoch_accuracy.append(val_accuracy)
            if opt.wandb:
                if opt.split_size:
                    wandb.log(
                    {"Epoch": epoch+1,
                "train_epoch_loss": round(avg_loss, 4),
                "train_epoch_accuracy": round(avg_accuracy, 4),
                "validation_epoch_loss": round(val_avg_loss, 4),
                "validation_epoch_accuracy": round(val_accuracy, 4),
                "train_elapsed_time": round(train_elapsed_time,3),
                "val_elapsed_time": round(val_elapsed_time,3)
                } )
                else:
                    wandb.log(
                {"Epoch": epoch+1,
                "train_epoch_loss": round(avg_loss, 4),
                "train_epoch_accuracy": round(avg_accuracy, 4),
                "train_elapsed_time": round(train_elapsed_time,3), 
                } )
            
            # model save
            model_save(save_path, docking_model, optimizer, epoch)  
        
    else:
        docking_model, optimizer = define_deepdiscovery_initializer(opt=opt, device=device, model_path=opt.weight_pt)
    
        print('---------- Model Test--------------')
        print('모델을 불러옵니다')
        test_loader = data_loader.get_test_loader()
        test_results = test(opt, docking_model, test_loader, criterion, device)
        
        print("최종 테스트 결과:")
        for metric, value in test_results.items():
            print(f"{metric}: {value:.4f}")
        


if __name__ == '__main__': # adain update
    #torchrun --nnodes=1 --nproc-per-node=4 main.py --distributed --wandb
    # python main.py --isTrain --wandb --weighted_loss --experiment_name deepdiscovery_run_weighted
    # python main.py --isTrain --wandb
    opt  = BaseOptions().parse()
    # for test python main.py
    
    run(opt)
    
