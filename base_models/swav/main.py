#%%

import argparse
import os
import yaml
import numpy as np
import torch

from torch.utils.data import DataLoader
from lightly.loss import SwaVLoss

from src.model import SwaV, swav_train
from src.transform import SwaVTransform
from src.dataset import SwAVDataset

# Define the main function
def main(fname):

    ########################### Config ###########################
    with open(os.path.join('configs',fname), 'r') as y_file:
        params = yaml.load(y_file, Loader=yaml.FullLoader)

    data_params = params['data']
    training_params = params['training']

    ########################### Data ###########################
    imgs_path = os.path.join(data_params['image_folder'], f"{data_params['dataset_name']}")
    checkpoint_folder = f"{data_params['dataset_name']}_{data_params['patch_size']}patches_{training_params['n_prototypes']}proto"
    checkpoint_folder = os.path.join(training_params['checkpoints_file_path'], checkpoint_folder)
    os.makedirs(checkpoint_folder, exist_ok=True)

    log_file = open(os.path.join(checkpoint_folder, "log.txt"), "a")
    with open(os.path.join(checkpoint_folder, 'params.yaml'), 'w') as f:
        yaml.dump(params, f)

    print('Loading images dataset into RAM')
    
    dataset = SwAVDataset(
        imgs_path,
        patch_size=data_params['patch_size'],
        patch_stride=data_params['patch_stride'],
        low_info_thresh=data_params['low_info_thresh'],
        adjust_scale=data_params['adjust_scale'],
        filter_data=data_params['filter_data']
    )
    
    print(f'Total images: {len(dataset)}')
    log_file.write(f'Total images: {len(dataset)}\n')

    dataloader = DataLoader(dataset, batch_size=training_params['batch_size'], shuffle=True, drop_last=True)
    
    transform = SwaVTransform(
        crop_sizes=data_params['crop_sizes'],
        crop_min_scales=data_params['min_scales'],
        crop_max_scales=data_params['max_scales'],
        crop_counts=(data_params['n_hr_views'], data_params['n_lr_views']),
        normalize={'mean': [dataset.corrected_mean], 'std': [dataset.corrected_std]}
    )

    ########################### Model ###########################
    ln_proto = np.round(np.log(training_params['n_prototypes']), 5)
    print(f"############# ln(proto): {ln_proto} #############")
    log_file.write(f"ln(proto): {ln_proto}\n")

    model = SwaV(
        training_params['backbone_model'],
        data_params['patch_size'],
        data_params['n_hr_views'],
        training_params['n_prototypes'],
        training_params['n_features_swav'],
        training_params['batch_size']
    ).to(training_params['device'])

    if training_params['weights_file_name'] is not None:
        
        weights_path = os.path.join(checkpoint_folder, training_params['weights_file_name'])
        
        try:
            model.load_state_dict(torch.load(weights_path, weights_only=True))
        except:
            raise ValueError(
                f"Could not load weights from {weights_path}. "
                f"Please check the file path and ensure the file exists."
            )

    criterion = SwaVLoss(sinkhorn_epsilon=training_params['sinkhorn_epsilon'])
    optimizer = torch.optim.Adam(model.parameters(), lr=training_params['learning_rate'])

    ########################### Training ###########################
    swav_train(model, dataset, dataloader, transform, criterion, optimizer, params, checkpoint_folder, log_file)
    
    log_file.close()


fname='modelv2_test.yaml'

# Entry point
# if __name__ == '__main__':

#     parser = argparse.ArgumentParser(description="Run SwaV training.")
#     parser.add_argument('fname', type=str, help="Path to the YAML configuration file.")
#     args = parser.parse_args()

#     fname = args.fname

main(fname)

# %%
