#%%

import torch
import numpy as np
import matplotlib.pyplot as plt


from src.dataset import SurfWeatherDataset
from torch.utils.data import DataLoader
from src.transform import MultiCropTranform
from src.loss import SwaVLoss
from src.model import SwaV, swav_train

import os
import yaml

# from src.model import SwaV, swav_train

surf_stats =  {'Tmin': (19.884305000538596, 3.5784525624804515),
                'Tmax': (30.66388006868657, 3.613170686518841),
                'pr': (4.803210990399181, 8.075623155365308)}


########## Params ############
fname = 'surfweather.yaml'
        
with open(os.path.join('configs',fname), 'r') as y_file:
    params = yaml.load(y_file, Loader=yaml.FullLoader)

data_params = params['data']
training_params = params['training']

checkpoint_folder = f"{'-'.join(data_params['surf_vars'])}_{training_params['n_prototypes']}proto"
checkpoint_folder = os.path.join(training_params['checkpoints_file_path'], checkpoint_folder)
os.makedirs(checkpoint_folder, exist_ok=True)

log_file = open(os.path.join(checkpoint_folder, "log.txt"), "a")
with open(os.path.join(checkpoint_folder, 'params.yaml'), 'w') as f:
    yaml.dump(params, f)
    
########## Data ############
dataset = SurfWeatherDataset(data_params['surf_vars'], data_params['static_vars'], surf_stats)
log_file.write(f'Total images: {len(dataset)}\n')

dataloader = DataLoader(dataset, batch_size=training_params['batch_size'], shuffle=True, drop_last=True)

transform = MultiCropTranform(
    crop_sizes=data_params['crop_sizes'],
    crop_min_scales=data_params['min_scales'],
    crop_max_scales=data_params['max_scales'],
    crop_counts=(data_params['n_hr_views'], data_params['n_lr_views']),
)

########### Model ###########
ln_proto = np.round(np.log(training_params['n_prototypes']), 5)
print(f"############# ln(proto): {ln_proto} #############")
log_file.write(f"ln(proto): {ln_proto}\n")

model = SwaV(
    dataset.surf_vars,
    data_params['patch_size'],
    data_params['n_hr_views'],
    training_params['n_prototypes'],
    training_params['n_features_swav'],
    training_params['batch_size']
).to(training_params['device'])


criterion = SwaVLoss(sinkhorn_epsilon=training_params['sinkhorn_epsilon'])
optimizer = torch.optim.Adam(model.parameters(), lr=training_params['learning_rate'])


# ########################### Training ###########################
patience, epoch = 0, 0
min_loss = np.inf
training_params = params['training']
data_params = params['data']

print(f'Starting Training')

while True:
    total_loss = 0
    for batch in dataloader:

        views = transform(batch)
        high_resolution, low_resolution = views[:data_params['n_hr_views']], views[data_params['n_hr_views']:]

        high_resolution, low_resolution, queue = model(
            high_resolution, low_resolution, epoch
        )

        loss = criterion(high_resolution, low_resolution, queue)
        total_loss += loss.detach()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    avg_loss = total_loss / len(dataloader)
    patience += 1
    epoch += 1
    loss_print = f'epoch: {epoch:>04}, loss: {avg_loss:.5f}'

    if avg_loss < min_loss:
        min_loss = avg_loss
        loss_print += ' (*)'
        patience = 0
        torch.save(model.state_dict(), f'{checkpoint_folder}/minloss.pth')
    
    if (epoch % 100) == 0:
        torch.save(model.state_dict(), f'{checkpoint_folder}/epoch_{epoch}.pth')
    
    print(loss_print)
    log_file.write(loss_print + "\n")

    if patience >= training_params['max_patience'] and epoch >= training_params['min_epochs']:
        torch.save(model.state_dict(), f'{checkpoint_folder}/epoch_{epoch}.pth')
        break
    
log_file.close()