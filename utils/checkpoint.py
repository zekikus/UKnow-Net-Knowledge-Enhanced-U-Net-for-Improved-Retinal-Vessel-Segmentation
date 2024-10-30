import torch

class BestModelCheckPoint:

    def __init__(self, model_name):
        self.best_score = 0
        self.model_name = model_name
    
    def check(self, score, model, seed, data_flag, epoch=None):
        if score > self.best_score:
            print("Best Score:", score)
            self.best_score = score
            torch.save(model.state_dict(), f"results/{data_flag}/{self.model_name}{'_epoch_' + str(epoch) if epoch is not None else ''}/model_{self.model_name}_seed_{seed}.pt")