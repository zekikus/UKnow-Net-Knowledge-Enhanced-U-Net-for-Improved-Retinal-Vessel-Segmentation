import os
import torch
import random
import pickle
import numpy as np
import pandas as pd
from unet import build_unet
from torch.utils.data import DataLoader
from utils.dataset import vessel_dataset

def writePickle(data_flag, data, fname):
        # Write History
        with open(f"Datasets/MIXED/training_pro/{fname}.pkl", "wb") as pkl:
            pickle.dump(data, pkl)

def main():

    best_config = {"DRIVE" : 0, 'CHUAC' : 3074, 'DCA1' : 0, 'CHASEDB1' : 0}
    for data_flag, seed in best_config.items():
        counter = 0
        data_path = f"Datasets/{data_flag}"

        # Directory exist?
        if not os.path.exists(f"Datasets/{data_flag}/training_pseudo_pro/"):
            os.makedirs(f"Datasets/{data_flag}/training_pseudo_pro/")

        # Parameters
        BATCH_SIZE = 1
        ENCODER_NAME = 'unet'
        print("BATCH SIZE:", BATCH_SIZE)
        print("DATASET:", data_flag, "ENCODER:", ENCODER_NAME, "SEED:", seed)
        
        # load the data
        train_dataset = vessel_dataset(data_path, mode="training")
        
        # encapsulate data into dataloader form
        train_loader = DataLoader(train_dataset, BATCH_SIZE, shuffle=False, pin_memory=True, drop_last=True)

                
        device = torch.device('cuda')
        model = build_unet()

        model.load_state_dict(torch.load(f"results/{data_flag}/{ENCODER_NAME}/model_{ENCODER_NAME}_seed_{seed}.pt"))
        model.to(device)
                            
        # Test Phase
        model.eval()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            with torch.no_grad():
                logits_mask = model(inputs)
                prob_mask = logits_mask.sigmoid()
                pred_mask = (prob_mask > 0.5).float()
                writePickle(data_flag, pred_mask.squeeze(0).cpu().numpy(), f"gt_patch_{counter}_{data_flag}")
                writePickle(data_flag, inputs.squeeze(0).cpu().numpy(), f"img_patch_{counter}_{data_flag}")

            counter += 1
            del logits_mask
            del inputs
            del labels

    torch.cuda.empty_cache()

## Main
if __name__ == '__main__':
    main()