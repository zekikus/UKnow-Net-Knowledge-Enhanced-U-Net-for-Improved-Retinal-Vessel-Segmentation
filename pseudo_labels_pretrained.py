import os
import torch
import pickle
from torch.utils.data import DataLoader
from utils.dataset_pretrained import vessel_dataset
import segmentation_models_pytorch as smp

def writePickle(config, data, fname):
        # Write History
        with open(f"Datasets/{config}/training_pro/{fname}.pkl", "wb") as pkl:
            pickle.dump(data, pkl)

def get_best_config(encoder_name, epoch):
    best_configs = {}
    for data_flag in ["DRIVE", "CHUAC", "DCA1", "CHASEDB1"]:
        max_values = []
        for seed in [0, 42, 3074]:
            lines = []
            max_value = 0
            with open(f"results/{data_flag}/{encoder_name}_epoch_{epoch}/log_{encoder_name}_seed_{seed}.txt", "r") as f:
                lines = f.readlines()
        
            for line in lines:
                if line == '\n': continue
                data = float(line.split(",")[-1].replace("val_iou: ", "").replace("\n", "").strip())
                max_value = data if data > max_value else max_value

            max_values.append((seed, max_value))
        best_configs[data_flag] = sorted(max_values, key=lambda x: x[1], reverse=True)[0][0]

    return best_configs   
        
                    

def main():

    
    # Parameters
    BATCH_SIZE = 1
    NBR_EPOCH = 40
    ENCODER_NAME = 'resnet18'

    config = f"MIXED_{ENCODER_NAME}_epoch_{NBR_EPOCH}"
    best_config = get_best_config(ENCODER_NAME, NBR_EPOCH)
    for data_flag, seed in best_config.items():
        counter = 0
        data_path = f"Datasets/{data_flag}"

        # Directory exist?
        if not os.path.exists(f"Datasets/{config}/training_pro/"):
            os.makedirs(f"Datasets/{config}/training_pro/")

        print("BATCH SIZE:", BATCH_SIZE)
        print("DATASET:", data_flag, "ENCODER:", ENCODER_NAME, "SEED:", seed)
        
        # load the data
        train_dataset = vessel_dataset(data_path, mode="training")
        
        # encapsulate data into dataloader form
        train_loader = DataLoader(train_dataset, BATCH_SIZE, shuffle=False, pin_memory=True, drop_last=True)

                
        device = torch.device('cuda')
        model = smp.Unet(
                encoder_name=ENCODER_NAME,        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                encoder_weights='imagenet',     # use `imagenet` pre-trained weights for encoder initialization
                in_channels=1,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                classes=1,                      # model output channels (number of classes in your dataset)
        )

        model.load_state_dict(torch.load(f"results/{data_flag}/{ENCODER_NAME}_epoch_{NBR_EPOCH}/model_{ENCODER_NAME}_seed_{seed}.pt"))
        model.to(device)
                            
        # Test Phase
        model.eval()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            with torch.no_grad():
                logits_mask = model(inputs)
                prob_mask = logits_mask.sigmoid()
                pred_mask = (prob_mask > 0.5).float()
                writePickle(config, pred_mask.squeeze(0).cpu().numpy(), f"gt_patch_{counter}_{data_flag}")
                writePickle(config, inputs.squeeze(0).cpu().numpy(), f"img_patch_{counter}_{data_flag}")

            counter += 1
            del logits_mask
            del inputs
            del labels

    torch.cuda.empty_cache()

## Main
if __name__ == '__main__':
    main()