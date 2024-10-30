import torch
import pandas as pd
from tester import Tester
from unet import build_unet
from torch.utils.data import DataLoader
from utils.dataset import vessel_dataset

def main(path):
    device = torch.device('cuda')
    result_df = pd.DataFrame(columns=['DATASET', 'ENCODER_NAME', 'SEED', 'AUC',	'F1', 'Acc', 'Sen', 'Spe', 'pre', 'IOU'])
    for data_flag in ['DCA1']:
        print(data_flag)
        for seed in [0, 42, 3074]:
            data_path = f"Datasets/{data_flag}"

            model = build_unet()
            model.load_state_dict(torch.load(f"results/{path}/unet/model_unet_seed_{seed}.pt"))
            model.to(device)

            test_dataset = vessel_dataset(data_path, mode="test")
            test_loader = DataLoader(test_dataset, 1,
                                    shuffle=False, pin_memory=True)

            test = Tester(model, test_loader, data_path, path, seed, show=False)
            result_df = test.test(result_df, data_flag, 'unet', seed)
    
    result_df.to_excel(f"results/all_result_pseudo_labels_{path}.xlsx")

if __name__ == '__main__':
    main('MIXED_except_DCA1')