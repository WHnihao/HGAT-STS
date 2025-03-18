import torch
from torch import nn
import os
import dgl
import pytorch_lightning as pl
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning import Trainer
from pytorch_lightning import seed_everything
from pytorch_lightning import callbacks
from pytorch_lightning import loggers

from utils.config import config
from model.trainer import Train_GraphSTS
from utils.data_reader import Vocab
#os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
#Model = None
#logger = None
if __name__ == "__main__":
    #global logger
    global index_epoch
    seed_everything(config.seed)
    dgl.random.seed(config.seed)    #在DGL中设置随机方法的种子。随机方法包括各种采样器和随机游走例程。
    index_epoch = 0
    model = Train_GraphSTS(index_epoch, config)
    logger = loggers.TensorBoardLogger(
        save_dir=config.save_dir
    )
    save_dir=config.save_dir
    ##logger = loggered
    checkpoint_args = dict(
        monitor='eval_f1',
        mode='max',
    )                  #字典，两个参数
    early_stopping = callbacks.EarlyStopping(
        patience=3,
        strict=True,
        verbose=True,
        **checkpoint_args
    )            #提前停止条件
    ckpt_callback = callbacks.ModelCheckpoint(
        filepath=os.path.join(logger.log_dir, '{epoch}-{val_loss:.4f}-{acc:.4f}'),
        save_top_k=1,
        verbose=True,
        prefix='',
        **checkpoint_args,
    )                           #保存模型
    
    trainer_args = dict(
        gpus=config.gpus, 
        num_nodes=config.num_nodes, 
        precision=config.precision, 
        early_stop_callback=False,  # early_stopping
        checkpoint_callback=ckpt_callback,
        logger=logger,
        limit_train_batches=1.0, #0.1
        limit_val_batches=1.0, #1.0
        limit_test_batches=1.0,#1.0
        val_check_interval=50, 
        auto_select_gpus=True,
        #check_val_every_n_epoch=1.0,
        deterministic=True, # True,
        benchmark=False, # True,
        gradient_clip_val=5,
        profiler=True,
        progress_bar_refresh_rate=1,
        # auto_lr_find=True,
        # auto_scale_batch_size = 'bin', # None
        accumulate_grad_batches= config.actual_batch_size // config.batch_size,
    )
    
    trainer = Trainer(**trainer_args, resume_from_checkpoint=config.ckpt_path if config.ckpt_path else None)
    
    
    if config.mode == 'test':
        trainer.test(model)
    elif config.mode == 'train':
        trainer.fit(model)
        trainer.test()
