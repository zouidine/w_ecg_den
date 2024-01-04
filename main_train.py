import os
import torch
import numpy as np
from w_ecg_den.ddpm import DDPM
from w_ecg_den.models import WModel
from w_ecg_den.training import train, evaluate
from sklearn.model_selection import train_test_split
from w_ecg_den.datapreparation import Data_Preparation
from torch.utils.data import DataLoader, Subset, TensorDataset

foldername = "/content/drive/MyDrive/Colab/ISIVC"
print('folder: ', foldername)
os.makedirs(foldername, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
config = {
    "train":{
        "feats": 32,
        "epochs": 400,
        "batch_size": 96 ,
        "lr": 1.0e-3
        },
    "diffusion":{
        "beta_start": 0.0001,
        "beta_end": 0.5,
        "num_steps": 50,
        "schedule": "linear"
        }
    }


# To enhance reproducibility
seed = 1234
np.random.seed(seed=seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


[X_train, y_train, X_test, y_test] = Data_Preparation(DataPath=foldername+'/data/')

X_train = torch.FloatTensor(X_train)
X_train = X_train.permute(0,2,1)

y_train = torch.FloatTensor(y_train)
y_train = y_train.permute(0,2,1)

X_test = torch.FloatTensor(X_test)
X_test = X_test.permute(0,2,1)

y_test = torch.FloatTensor(y_test)
y_test = y_test.permute(0,2,1)

train_val_set = TensorDataset(y_train, X_train)
test_set = TensorDataset(y_test, X_test)

train_idx, val_idx = train_test_split(list(range(len(train_val_set))),
                                      test_size=0.3)
train_set = Subset(train_val_set, train_idx)
val_set = Subset(train_val_set, val_idx)

train_loader = DataLoader(train_set, batch_size=config['train']['batch_size'],
                          shuffle=True, drop_last=True, num_workers=0)
val_loader = DataLoader(val_set, batch_size=config['train']['batch_size'],
                        drop_last=True, num_workers=0)
test_loader = DataLoader(test_set, batch_size=50, num_workers=0)

base_model = WModel(w_name="haar", feats=config['train']['feats']).to(device)
model = DDPM(base_model, config, device)
print('training ...')
train(model, config['train'], train_loader, device,
      valid_loader=val_loader, valid_epoch_interval=1,
      foldername=foldername+'/models')
