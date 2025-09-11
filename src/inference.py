import os, sys
import yaml, torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import torch
from tqdm import tqdm
import torchvision.transforms as transforms
# from torchvision.transforms import v2
from sklearn.manifold import TSNE

from src.data_manager import init_data
from src.msn_train import init_model

def read_data(directory,config_file,validation=True):
    
    with open(os.path.join(directory, config_file), 'r') as y_file:
        params = yaml.load(y_file, Loader=yaml.FullLoader)
    
    rand_transform = transforms.Compose([
        transforms.RandomResizedCrop(128, scale=(0.5, 1.0)),
        transforms.Normalize(
            params['data']['norm_means'], #'Tmin', 'Tmax',
            params['data']['norm_stds'])
    ])


    (unsupervised_loader, unsupervised_sampler) = init_data(
        transform=rand_transform,
        batch_size=8,
        surf_vars=params['data']['surf_vars'],
        static_vars=params['data']['static_vars'],
        lat_lim=params['data']['lat_limit'], lon_lim=params['data']['lon_limit'],
        split_val = validation,
    )
    
    if validation:
        transformed_imgs = torch.stack([rand_transform(img) for img in unsupervised_loader.dataset.validation_imgs])
        unsupervised_loader.dataset.validation_imgs = transformed_imgs
    
    return params, unsupervised_loader.dataset

def load_model(params):
        
    target_encoder = init_model(
        device = "cuda:0",
        model_name = params['meta']['model_name'],
        use_bn = params['meta']['use_bn'],
        hidden_dim = params['meta']['hidden_dim'],
        output_dim = params['meta']['output_dim'],
        drop_path_rate = params['meta']['drop_path_rate'],
        log=False
    )

    latest_checkpoint = os.path.join(params['logging']['folder'], params['logging']['write_tag'] + "-latest.pth.tar")
    checkpoint = torch.load(latest_checkpoint, map_location='cpu',weights_only=False)

    target_encoder.load_state_dict(checkpoint['target_encoder'], strict=False)
    target_encoder.eval()

    prot = checkpoint['prototypes'].to("cuda:0")
    prot = torch.nn.functional.normalize(prot)
    
    return target_encoder, prot

def get_model_results(read_path, validation=False, **kwargs ):
    """Get model results for the latent space.

        E, F = gp.get_model_results(read_path,
                                    params=params, dataset=dataset,
                                    encoder=target_encoder, prototypes=prot)
    """

    E_file_name = f"E{'_val' if validation else ''}.pt"
    F_file_name = f"F{'_val' if validation else ''}.pt"
        
    try:
       E = torch.load(os.path.join(read_path, E_file_name),weights_only=False)
       F = torch.load(os.path.join(read_path, F_file_name),weights_only=False)

    except:
        
        params = kwargs['params']
        encoder = kwargs['encoder']
        dataset = kwargs['dataset']
        prototypes = kwargs['prototypes']

        E = torch.empty(0, params['meta']['output_dim']).to("cuda:0")
        F = torch.empty(0, prototypes.shape[0]).to("cuda:0")

        with torch.no_grad():
            for img in tqdm(dataset, desc="Encoding"):
                
                if not validation: img, _ = img
                
                img = img.unsqueeze(0).to("cuda:0")
                enc = encoder(img)
                enc = torch.nn.functional.normalize(enc)
                E = torch.cat((E, enc), 0)
                F = torch.cat((F, enc @ prototypes.T), 0)

        os.makedirs(read_path, exist_ok=True)
        torch.save(E, os.path.join(read_path,E_file_name))
        torch.save(F, os.path.join(read_path, F_file_name))
        print(f"Saved E and F to {read_path}")
        
    else:
        print(f"Loaded E and F from {read_path}")
        
    return E, F

def get_TSNE(read_path, validation=True, **kwargs):
    """Get t-SNE embeddings for the latent space.

        tsne_E, tsne_prot, tsne_Eval = gp.get_TSNE("results/", 
                                      E=E, prot=prot, E_val=E_val) #kwargs
    """
    try:
        tsne_E = torch.load(os.path.join(read_path, 'tsne_E.pt'),weights_only=False)
        tsne_prot = torch.load(os.path.join(read_path, 'tsne_prot.pt'),weights_only=False)
        
        if validation:
            tsne_Eval = torch.load(os.path.join(read_path, 'tsne_Eval.pt'),weights_only=False)
            
    except:
        
        latent_data = torch.cat(list(kwargs.values()))
        
        features_shape = kwargs['E'].shape[0]
        prot_shape = kwargs['prot'].shape[0]
        
        tsne = TSNE(n_components=2).fit_transform(latent_data)

        tsne_E = tsne[:features_shape]
        tsne_prot = tsne[features_shape:features_shape+prot_shape]
        
        torch.save(tsne_E, os.path.join(read_path,'tsne_E.pt'))
        torch.save(tsne_prot, os.path.join(read_path,'tsne_prot.pt'))

        if validation:
            tsne_Eval = tsne[features_shape+prot_shape:]
            torch.save(tsne_Eval, os.path.join(read_path,'tsne_Eval.pt'))
    
    return tsne_E, tsne_prot, tsne_Eval

