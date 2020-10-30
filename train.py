import time
import math

import torch
from torch.utils.data import DataLoader
from torch.optim import Adam, lr_scheduler

from configs import configs
from utils import * 
from dataset import *
from models import *

if __name__ == "__main__":
    # initialize the deive and random seed
    configs.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    seed_torch(configs.random_seed)

    # data transform
    data_transforms = get_transform()

    if(configs.use_pretrain):
        print('#' * 10, 'Use pretrained models', '#' * 10)
        mean_model = torch.load('./pretrained_models/mean_model.pth', map_location=configs.device)
        std_model = torch.load('./pretrained_models/std_model.pth', map_location=configs.device)
    else:
        
        ################ Train Mean Target #################
        print('#' * 10, 'Mean training', '#' * 10)

        # training datasets and dataloaders
        image_dataset = MeanDataset(configs.train_dir, 0, transform=data_transforms)
        tr_dataloader = DataLoader(
            image_dataset, 
            batch_size=configs.batch_size, 
            shuffle=True, 
            num_workers=2) 
        
        # initialize model, optimizer, loss
        mean_model = Mean_Model().to(configs.device)
        optimizer = Adam(mean_model.parameters(), lr=configs.lr)
        criterion = nn.MSELoss(reduction='sum').to(configs.device)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

        # training procedure
        best_loss = 100.0
        start_time = time.time()
        for e in range(1, configs.epoch_mean + 1):
            epoch_start_time = time.time()
            train_loss = train_mean(tr_dataloader, mean_model, optimizer, criterion)
            avg_loss = math.sqrt(train_loss / len(image_dataset))
            scheduler.step()
            elapsed_time = time.time() - epoch_start_time
            print('Epoch {}/{} \t loss={:.4f} \t time={:.2f}s'.format(
                e, configs.epoch_mean, avg_loss, elapsed_time))
            if(avg_loss < best_loss):
                best_loss = avg_loss

        time_elapsed = time.time() - start_time
        print('Mean training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Best training RMSE: {:4f}'.format(best_loss))
        torch.save(mean_model, './pretrained_models/mean_model.pth')

        ################ Train Std Target #################
        print('#' * 10, 'Std training', '#' * 10)

        # training datasets and dataloaders
        image_dataset = StdDataset(configs.train_dir, 0, transform=data_transforms)
        tr_dataloader = DataLoader(
            image_dataset, 
            batch_size=configs.batch_size, 
            shuffle=True, 
            num_workers=2)

        # initialize model, optimizer, loss
        std_model = Std_Model().to(configs.device)
        optimizer = Adam(std_model.parameters(), lr=configs.lr)
        criterion = nn.MSELoss(reduction='sum').to(configs.device)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

        # training procedure
        best_loss = 100.0
        start_time = time.time()
        for e in range(1, configs.epoch_std + 1):
            epoch_start_time = time.time()
            train_loss = train_std(tr_dataloader, std_model, optimizer, criterion)
            avg_loss = math.sqrt(train_loss / len(image_dataset))
            scheduler.step()
            elapsed_time = time.time() - epoch_start_time
            print('Epoch {}/{} \t loss={:.4f} \t time={:.2f}s'.format(
                e, configs.epoch_std, avg_loss, elapsed_time))

            if(avg_loss < best_loss):
                best_loss = avg_loss
        time_elapsed = time.time() - start_time
        print('Std training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Best training RMSE: {:4f}'.format(best_loss))
        torch.save(std_model, './pretrained_models/std_model.pth')

    ################ Mean Prediction #################
    print('#' * 10, 'Mean prediction', '#' * 10)
    pb_dataset = MeanDataset(configs.val_dir, 2, transform=data_transforms)
    pb_dataloader = DataLoader(pb_dataset, batch_size=32, shuffle=False, num_workers=2)

    arr_mean = None
    with torch.no_grad():
        for inputs, inputs_img in pb_dataloader:
            inputs, inputs_img = inputs.to(configs.device), inputs_img.to(configs.device)
            temp_1 = mean_model(inputs, inputs_img).cpu().data.numpy().reshape(-1, 1)
            if(isinstance(arr_mean, np.ndarray)):
                arr_mean = np.vstack((arr_mean, temp_1))
            else:
                arr_mean = temp_1
            print(arr_mean.shape, end=',')
    
    ################ Std Prediction #################
    print('#' * 10, 'Std prediction', '#' * 10)
    pb_dataset = StdDataset(configs.val_dir, 2, transform=data_transforms)
    pb_dataloader = DataLoader(pb_dataset, batch_size=32, shuffle=False, num_workers=2)

    arr_std = None
    with torch.no_grad():
        for inputs in pb_dataloader:
            inputs = inputs.to(configs.device)
            temp_1 = std_model(inputs).cpu().data.numpy()
            if(isinstance(arr_std, np.ndarray)):
                arr_std = np.vstack((arr_std, temp_1))
            else:
                arr_std = temp_1
            print(arr_std.shape, end=',')

    output = np.hstack((arr_mean, arr_std))
    np.savetxt('./output.txt', output, delimiter=' ')


