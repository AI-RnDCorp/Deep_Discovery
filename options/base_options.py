import argparse
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
from utils.util import *
import wandb
import sys

class BaseOptions():

    def __init__(self):
        #self.parser = argparse.ArgumentParser()
        self.parser = argparse.ArgumentParser(add_help=False)  # add_help=False로 변경
        self.initialized = False

    def initialize(self):
        # seed initialization
        self.parser.add_argument('--seed', type=int, default=42, help='set seed')
        self.parser.add_argument('--gpu_ids', type=str, default='-1', help='gpu ids: e.g. 0, 0,1,2, 0,2. use -1 for CPU')

        # wandb
        self.parser.add_argument('--wandb', action='store_true', help='if use wandb, default = False') # wandb login -> add flag --wandb
        self.parser.add_argument('--proj_name', type=str, default='training_experiment', help='specify the name of the wandb project')

        # experiments
        # self.parser.add_argument('--name', type=str, default ='ABCnet',help='name of the experiment. It decides where to store samples and models') # cycleGAN_256x256_vfp290k_tl
        self.parser.add_argument('--save_dir', type=str, default='./checkpoints', help='models are saved here')
        self.parser.add_argument('--experiment_name', type=str, default ='deepdiscovery_run',help='experiment name')

        # dataset
        self.parser.add_argument('--csvpath',  type=str, default='/Users/bioai/Documents/Datasets/BindingDB_dataset/classification_binding_DB_training.csv',help='path to original processed csv)')
        self.parser.add_argument('--train_datapath',  type=str, default='/Users/bioai/Documents/Datasets/BindingDB_dataset/final_training.csv',help='path to processed feature data)')
        self.parser.add_argument('--test_datapath',  type=str, default='/root/easymaker/DeepDiscovery/Dataset/classification_binding_DB_test_0.1.csv',help='path to processed feature data)')
        
        # self.parser.add_argument('--source_reference',  type=str, default='/Users/bioai/Documents/GitHub/BioAI/utils/Target Source Organism Categories',help='source and organism categories reference json path')
        # self.parser.add_argument('--source',  type=str, default='all',help='source from the dataset ex) all, "US Patent", "Curated from the literature by BindingDB", "PubChem", "CSAR", "D3R", "WIPO", "PDSP Ki", "ChEMBL"')
        # self.parser.add_argument('--category',  type=str, default='all',help='organism category from the dataset ex) all, homosapiens, disease_related, other')
        # val/train split # 0 for complete train and test
        self.parser.add_argument('--split_size', default=0.8, type=float, help='train/val split)')
        #dataloader
        self.parser.add_argument('--num_workers', default=16, type=int, help='Number of workers used in dataloading') #  cat /proc/cpuinfo | grep -c processor
        self.parser.add_argument('--batchsize', type=int, default=512, help='batch_size for training')
        self.parser.add_argument('--isTrain', action='store_true', help='Train, default = True') # wandb login -> add flag --wandb
        
        # train options        
        self.parser.add_argument('--lr', type=float, default=0.001, help='learning rate ADAM optimizer')
        self.parser.add_argument('--weight_decay', type=float, default=0.0000, help='weight decay for ADAM optimizer')
        self.parser.add_argument('--weighted_loss', action='store_true', help='if apply weighted loss, default = False') # wandb login -> add flag --weighted_loss

        self.parser.add_argument('--start_epoch', type=int, default=0, help='which epoch to load if continuing training, set 0 for new')
        self.parser.add_argument('--n_epochs', type=int, default=30, help='number of epochs for training')
        
        # test options
        self.parser.add_argument('--weight_pt',  type=str, default='/Users/bioai/Documents/GitHub/Deep_Discovery/deepdiscovery_weight.pt',help='pt file directory (saved model)')

    def print_options(self):

        print('------------ Options -------------')
        for k, v in sorted(self.args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

    def save_options(self):
        # save args to the disk

        expr_dir = os.path.join(self.opt.checkpoints_dir, self.opt.name)
        mkdir(expr_dir)
        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write('------------ Options -------------\n')
            for k, v in sorted(self.args.items()):
                opt_file.write('%s: %s\n' % (str(k), str(v)))
            opt_file.write('-------------- End ----------------\n')
            
    def parse(self):
        if not self.initialized:
            self.initialize()
        
        if '--f' in sys.argv:
            sys.argv.remove('--f')
            sys.argv.remove([arg for arg in sys.argv if arg.startswith('/Users/')][0])
        
        self.opt, _ = self.parser.parse_known_args()  # parse_args() 대신 parse_known_args() 사용
#        self.opt = self.parser.parse_args()
        self.args = vars(self.opt)

        self.print_options()
        #  self.save_options()            
        str_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.opt.gpu_ids.append(id)
        if len(self.opt.gpu_ids) > 0:
            if torch.cuda.is_available():
                torch.cuda.set_device(self.opt.gpu_ids[0]) # let main to 0.
            if self.opt.wandb:
                wandb.init(project=self.opt.proj_name, name=self.opt.experiment_name, entity="jinpark1104-deepdiscovery", config=self.args)
                print('-------------wandb initialized------------------')

        seed_everything(self.opt.seed)
        
        return self.opt
    
    
    
