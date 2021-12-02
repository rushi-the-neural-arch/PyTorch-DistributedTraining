import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
from fairscale.optim.oss import OSS
from fairscale.nn.data_parallel import ShardedDataParallel as ShardedDDP

from torch.optim import AdamW

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from models.sr_4k_2x import Net

#from MultiPerceptualLoss import feat_loss
from old_dataset import CustomDataset
import metrics
from test_dist_gpu import *

def train(
    rank: int,
    world_size: int,
    epochs: int
    ):
    
    # DDP init example
    dist.init_process_group(backend='gloo', init_method="env://", rank=rank, world_size=world_size)
    
    
    print("===> Loading datasets")
    
    input_path = '/opt/hubshare/vectorly-share/shared/Image_Superresolution/Dataset/Flickr2K/Patches/LRPatch_256/'                               
    target_path = '/opt/hubshare/vectorly-share/shared/Image_Superresolution/Dataset/Flickr2K/Patches/512/'  
    
    print("--Input Directory--", input_path)
    
    full_dataset = CustomDataset(input_path, target_path)
    
    
    train_size = int(0.99 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])
    
    train_sampler = torch.utils.data.distributed.DistributedSampler(
    	train_dataset,
    	num_replicas=4,
    	rank=rank
    )
    
    val_sampler = torch.utils.data.distributed.DistributedSampler(
    	val_dataset,
    	num_replicas=4,
    	rank=rank
    )
    
    BATCH_SIZE = 40
    
    training_dataloader = DataLoader(
                dataset=train_dataset, num_workers=16, batch_size=BATCH_SIZE, drop_last=True, shuffle=False, pin_memory=True, sampler=train_sampler)

    val_dataloader = DataLoader(
        dataset=val_dataset, num_workers=8, batch_size=BATCH_SIZE, shuffle=False,
                                               sampler=val_sampler)
    
    
    x, y = next(iter(training_dataloader))
    a, b = next(iter(val_dataloader))
    
    print("Length of Training dataset - ", len(train_dataset))
    print("--Shape--", a.shape, b.shape)
    
    print("===> Building model")
    model = Net(upscale_factor=2)
    
    loss_fn =  nn.MSELoss() # feat_loss
    
    base_optimizer = AdamW
    
    base_optimizer_arguments = { 
             "lr" : 1e-3,
             "betas" : (0.9, 0.99),
             "eps" : 1e-8,
             "weight_decay" : 1e-4}
    
    optimizer = OSS(params=model.parameters(), optim=base_optimizer, **base_optimizer_arguments)

    # Wrap the model into ShardedDDP, which will reduce gradients to the proper ranks
    model = ShardedDDP(model, optimizer)
    
     # Any relevant training loop, nothing specific to OSS. For example:
    model.train()
    for e in range(epochs):
        for iteration, batch in enumerate(training_dataloader, 1):
        #for batch in training_dataloader:
            # Train
            model.zero_grad()
            outputs = model(batch[0])
            loss = loss_fn(outputs, batch[1])
            loss.backward()
            optimizer.step()
            
            if iteration%25 == 0:
                print(loss)

            
        print("For Epoch {}, loss: {:.2f}".format(e, loss))
            
    dist.destroy_process_group()
    
    
if __name__ == "__main__":
    # Supposing that WORLD_SIZE and EPOCHS are somehow defined somewhere
    nr = 0
    gpus = 4
    WORLD_SIZE = 4
    
    EPOCHS = 2
    
    rank = nr * gpus 
    
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = find_free_port()
    
    mp.spawn(
        train,
        args=(
            WORLD_SIZE,
            EPOCHS,
        ),
        nprocs=WORLD_SIZE,
        join=True,
    )