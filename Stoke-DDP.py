#python -m torch.distributed.launch Stoke-DDP.py --projectName "PyTorch-4K-2X" --batchSize 20 --nEpochs 2 --lr 1e-3 --threads 8
#env CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 Stoke-DDP.py --projectName "Stoke-4K-2X-DDP" --batchSize 18 --nEpochs 2 --lr 1e-3 --weight_decay 1e-4 --grad_clip 0.1

import argparse, os, sys
import random
import numpy as np
import time
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn 
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

from torch.nn.parallel import DistributedDataParallel as DDP

from torch.utils.data.distributed import DistributedSampler
from stoke import DeepspeedConfig, DeepspeedZeROConfig, Stoke, StokeOptimizer
from stoke import AMPConfig
from stoke import ClipGradNormConfig
from stoke import DDPConfig
from stoke import DistributedOptions
from stoke import FairscaleOSSConfig
from stoke import FP16Options
from stoke import Stoke

from torch.optim import AdamW, SGD

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

#from models.sr_4k_2x import Net
from models.network_swinir import SwinIR

from PyTorchPercept import feat_loss

from old_dataset import CustomDataset
import metrics

from tqdm import tqdm
    
import wandb
wandb.login()



def train_log(loss, example_ct, epoch):
    # Where the magic happens
    wandb.init()
    wandb.log({"epoch": epoch, "train_loss": loss}) # , step=example_ct
    print(f"Loss after " + str(example_ct).zfill(5) + f" examples: {loss:.3f}")
    
    
def val_log(loss, avg_mae, avg_psnr, example_ct, epoch):
    # Where the magic happens
    wandb.init()
    wandb.log({"epoch": epoch, "val_loss": loss, "PSNR": avg_psnr, "MAE": avg_mae}) # , step=example_ct
    print(f"-----VALIDATION Loss after " + str(example_ct).zfill(5) + f" examples: {loss:.3f}--------")
    
    
def train(train_dataloader, stoke_model: Stoke, scheduler1, scheduler2, epoch: int):
    
    example_ct = 0  # number of examples seen
    batch_ct = 0
    sum_loss = 0
    
    stoke_model.print_on_devices(f"Starting Epoch {epoch + 1}")
    stoke_model.model_access.train()
    
    for idx, (inputs, targets) in enumerate(train_dataloader):
        
        # call the model through the stoke onkect interface
        outputs = stoke_model.model(inputs)
        train_loss = stoke_model.loss(outputs, targets)
        
        stoke_model.print_ema_loss(prepend_msg=f"Step {idx+1} -- EMA Loss")
        
        # Call backward through the stoke object interface
        stoke_model.backward(loss=train_loss)
        
        # Call step through the stoke object interface
        stoke_model.step()
        scheduler1.step()
        scheduler2.step
        
        sum_loss += stoke_model.detach_and_sync_loss(loss=train_loss)

        example_ct +=  len(inputs)
        batch_ct += 1

        # Report metrics every 50th batch
        if ((batch_ct + 1) % 50) == 0:
            train_log(train_loss, example_ct, epoch)
            #print(train_loss,  example_ct, epoch)

    avg_loss = sum_loss / len(train_dataloader)
    
    return avg_loss
        
        
def validate(val_dataloader, stoke_model: Stoke, epoch):
    
    # Switch to eval mode
    stoke_model.model_access.eval()
    total_y = 0
    total_correct = 0
    
    val_loss, example_ct = 0, 0
    mae, psnr = 0, 0
    
    # Wrap with no grads context just to be safe
    with torch.no_grad():
        for inputs, targets in val_dataloader:
            example_ct += len(inputs)
            
            outputs = stoke_model.model(inputs)

            val_loss += stoke_model.loss(outputs, targets)

            mae += metrics.mae(outputs, targets)
            psnr += metrics.psnr(outputs, targets)
            
    val_avg_loss = val_loss/len(val_dataloader)
        
    avg_mae = mae/len(val_dataloader)
    avg_psnr = psnr/len(val_dataloader)    
    
    val_log(val_avg_loss, avg_mae, avg_psnr, example_ct, epoch)
          
    stoke_model.print_on_devices(
        msg=f"Current Average Validation Loss: {val_avg_loss}, PSNR : {avg_psnr}"
    )
    
    return val_avg_loss
    
    
def save_checkpoint(stoke_model, epoch, train_loss, val_loss):
    
    if not os.path.exists("checkpoint/"):
        os.makedirs("checkpoint/")
        
    path, tag = stoke_model.save(
        path='checkpoint/',
        name='model_{}_{:.2f}_{:.2f}'.format(epoch, train_loss, val_loss),
    )
    
    print("Checkpoint saved after epoch {}".format(epoch))

    
def main():


    os.environ['LOCAL_RANK'] = str(os.getenv('LOCAL_RANK')) #'0' #ddp_config.local_rank
    os.environ['PYTHONWARNINGS'] = 'ignore:semaphore_tracker:UserWarning'
    
    parser = argparse.ArgumentParser(description="PyTorch-W&B-Training")
    
    parser.add_argument("--projectName", default="Stoke-4K-2X-DDP", type=str, help="Project Name for W&B")
    parser.add_argument("--batchSize", type=int, default=18, help="Training batch size")
    parser.add_argument("--nEpochs", type=int, default=10, help="Number of epochs to train for")
    parser.add_argument("--start-epoch", default=1, type=int, help="Manual epoch number (useful on restarts)")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning Rate. Default=0.1")
    parser.add_argument("--weight_decay", "--wd", default=1e-4, type=float, help="Weight decay, Default: 1e-4")
    parser.add_argument("--grad_clip", type=float, default=0.1, help="Clipping Gradients. Default=0.1")
    
    parser.add_argument("--local_rank", default=-1, type=int, help="rank (default: 0)")
    parser.add_argument("--threads", type=int, default=16, help="Number of threads for data loader to use, Default: 4")
    
    parser.add_argument("--inputDir", type=str, default='/opt/hubshare/vectorly-share/shared/Image_Superresolution/Dataset/Flickr2K/Patches/LRPatch_128/', help="Training Dataset Path") # Train/CombinedALL/2X/
    parser.add_argument("--targetDir", type=str, default='/opt/hubshare/vectorly-share/shared/Image_Superresolution/Dataset/Flickr2K/Patches/HR_256/' , help="Training Dataset Path")
    
    global opt
    opt = parser.parse_args()
    
    #os.environ['LOCAL_RANK'] = opt.local_rank
    epochs = opt.nEpochs
    
        #trainDir = '/opt/hubshare/vectorly-share/shared/Image_Superresolution/Dataset/Flickr2K/Patches/LRPatch_256/' # HR_512
    
    # Custom AMP configuration
    # Change the initial scale factor of the loss scaler
    amp_config = AMPConfig(
        init_scale=2.**14
    )
    
    # Custom DDP configuration
    # Automatically swap out batch_norm layers with sync_batch_norm layers
    # Notice here we have to deal with the local rank parameter that DDP needs (from env or cmd line)
    
    ddp_config = DDPConfig(
        local_rank= int(os.getenv('LOCAL_RANK')),
        convert_to_sync_batch_norm=True
    )

    # Custom OSS configuration
    # activate broadcast_fp16 -- Compress the model shards in fp16 before sharing them in between ranks
    oss_config = FairscaleOSSConfig(
        broadcast_fp16=True
    )


    print("===> Building model")
    
    #model = Net(upscale_factor=2)
    
    model = SwinIR(upscale=2, in_chans=3, img_size=64, window_size=8,
                    img_range=1., depths=[6, 6, 6, 6], embed_dim=60, num_heads=[6, 6, 6, 6],
                    mlp_ratio=2, upsampler='pixelshuffledirect', resi_connection='1conv')
    param_key_g = 'params'
    model_path = 'model_zoo/002_lightweightSR_DIV2K_s64w8_SwinIR-S_x2.pth'

    pretrained_model = torch.load(model_path)
    model.load_state_dict(pretrained_model[param_key_g] if param_key_g in pretrained_model.keys() else pretrained_model, strict=True)

#     from mdfloss import MDFLoss
#     path_disc = "mdf/weights/Ds_SISR.pth"
#     discriminator = torch.load(path_disc)
#     discrminator = nn.DataParallel(discriminator)
    
    #loss = MDFLoss(path_disc, cuda_available=True)
    #loss = torch.nn.parallel.DistributedDataParallel(loss)
    
    
    loss = feat_loss #nn.MSELoss() 
    
    optimizer = StokeOptimizer(
         optimizer = AdamW,
         optimizer_kwargs = {
             "lr" : 1e-3,          
             "betas" : (0.9, 0.99),
             "eps" : 1e-8,
             "weight_decay" : opt.weight_decay       
         }
        
     )


    # Build the object with the correct options/choices (notice how DistributedOptions and FP16Options are already provided
# to make choices simple) and configurations (passed to configs as a list)
    stoke_model = Stoke(
        model=model,
        verbose=True,    
        optimizer=optimizer,
        loss=loss,
        batch_size_per_device= opt.batchSize,   
        gpu=True,   
        fp16= None, #FP16Options.amp.value, 
        distributed=DistributedOptions.ddp.value,
        fairscale_oss=True, 
        fairscale_sddp=True, 
        grad_accum_steps=2,
        configs= [amp_config, ddp_config, oss_config],     
        grad_clip=ClipGradNormConfig(max_norm = opt.grad_clip, norm_type=2.0),
    )

    
    print("===> Loading datasets")                    
    
    input_path =  opt.inputDir                           
    target_path = opt.targetDir
    
    print("--Input Directory--", input_path)
    
    full_dataset = CustomDataset(input_path, target_path)
    
    train_size = int(0.9 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])
    
    
    train_sampler = DistributedSampler(
            dataset=train_dataset,
            num_replicas=stoke_model.world_size,
            rank=stoke_model.rank
        )
    

    val_sampler = DistributedSampler(
    	val_dataset,
    	num_replicas=stoke_model.world_size,
    	rank=stoke_model.rank
    )
    
    
    train_dataloader = stoke_model.DataLoader(
        dataset=train_dataset,
        sampler=train_sampler,
        num_workers= opt.threads,
        multiprocessing_context='spawn'
    )
    
    val_dataloader = stoke_model.DataLoader(
        dataset=val_dataset,
        sampler=val_sampler,
        multiprocessing_context='spawn',
        num_workers=8
    )
    
    scheduler1 = optim.lr_scheduler.OneCycleLR(stoke_model.optimizer, max_lr=0.01, pct_start = 0.9, steps_per_epoch=len(train_dataloader), epochs=epochs)
    scheduler2 = optim.lr_scheduler.ReduceLROnPlateau(stoke_model.optimizer,
                                                         mode='min',
                                                         factor=0.2,
                                                         patience=2,
                                                         min_lr=5e-5,
                                                         verbose = True)
        
    config = dict(
        epochs=opt.nEpochs,
        batch_size=opt.batchSize,
        learning_rate=opt.lr,
        dataset="DemoVal",
        architecture="4K-2X-DDP"
    )
    
    while True:
        try:
            wandb.init(project = opt.projectName, config=config, reinit=True) # mode="disabled" for disabling logging
            break
        except:
            print("Retrying")
            time.sleep(10)

        #wandb sync
    config = wandb.config

    print("===> Training")

    
    for epoch in tqdm(range(epochs), leave=True): 
        
        train_loss = train(train_dataloader, stoke_model, scheduler1, scheduler2, epoch)
        val_loss = validate(val_dataloader, stoke_model, epoch)
        save_checkpoint(stoke_model, epoch, train_loss, val_loss)

        print("--------Train Loss after Epoch {} - {} --------".format(epoch, train_loss))
        print("--------Val Loss after Epoch {} - {} --------".format(epoch, val_loss))

    wandb.finish()

if __name__ == "__main__":
    main()
    
    
