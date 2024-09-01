import init_path
import os
import numpy as np
import pandas as pd
from pycocotools.coco import COCO
import skimage.io as io
import nibabel as nib
import pickle
import seaborn as sns
import os
import re
import warnings
from tqdm import tqdm
import json
import random

from scipy import signal, stats
from scipy.interpolate import interp1d
from sklearn.model_selection import KFold

import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import pytorch_lightning as pl
from omegaconf import OmegaConf
from omegaconf import DictConfig

from settings import settings
from utils import *
from filters import filters
from funcs import *

import timm.optim.optim_factory as optim_factory
from datasets_transfer import NSDTransferDataset
from model_utils import CosineWarmupScheduler
from einops import repeat

from baseline_models import BrainNetworkTransformer
from baseline_models import GraphTransformer
from baseline_models import BrainNetCNN
from baseline_models import FBNETGEN
from baseline_models import ModelSTAGIN
from baseline_models.STAGIN import process_dynamic_fc

def load_model_args(baseline_id, verbose=True, use_DictConfig=False):
    def _to_DataContainer(obj):
        if type(obj) is dict:
            obj_ = DataContainer()
            setattr(obj_, '_contains', list(obj.keys()))
            for key, d in obj.items():
                setattr(obj_, key, _to_DataContainer(d))
            return obj_
        else:
            return obj
        
    fName = 'NSD_baselines.json'
    fPath = settings.CONFIG_DIR / fName
    with open(fPath, "r") as file:
        dat = json.load(file)
        dat = dat[baseline_id]
    
    if verbose:
        print('***** Model configuration *****')
        print(json.dumps(dat, indent=4))
    if use_DictConfig:
        dat = OmegaConf.create(dat)
    else:
        dat = _to_DataContainer(dat)
    return dat

def transfer_learning_loss(preds, batch, args):
    
    loss_by_tgts = {}
    losses = []
    for tgt_name in args.training.output_continuous_targets:
        idx = getattr(args.model.head_tgt2dims, tgt_name)
        loss = F.mse_loss(preds[:,idx], batch[tgt_name])
        losses.append(loss)
        loss_by_tgts[tgt_name] = loss.detach().cpu().item()

    for tgt_name in args.training.output_discrete_targets:
        idxs = getattr(args.model.head_tgt2dims, tgt_name)
        loss = F.cross_entropy(preds[:,idxs[0]:idxs[1]+1], batch[tgt_name])
        losses.append(loss)
        loss_by_tgts[tgt_name] = loss.detach().cpu().item()
    
    loss = torch.stack(losses).mean()
    loss_by_tgts['total'] = loss.detach().cpu().item()
    return loss, loss_by_tgts

def train_model_one_epoch(model, train_loader, val_loader, 
                          args, optimizer, scheduler, criterion):
    
    ###### Training ######
    model.train()
    optimizer.zero_grad()
    train_losses = []
    train_lrs = []    
    dataIter = iter(train_loader)

    for i_iter in tqdm(range(len(dataIter)), 
            desc='train epoch [{:d}|{:d}]'.format(args.training.epoch, args.training.max_epoch)):
        
        batch = next(dataIter)
        batch = {key: item.to(args.training.device) for key, item in batch.items()}
        
        preds = model(batch['timeseries'], batch['fc'])
        loss, loss_by_tgts = criterion(preds, batch, args)
        loss /= args.training.accum_iter
        loss.backward()
        
        if (((i_iter + 1)) % args.training.accum_iter == 0) or ((i_iter + 1) == len(dataIter)):
            """ update model parameters """
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.training.clip_grad)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
        train_losses.append(loss_by_tgts)
        train_lrs.append(scheduler.get_last_lr())
        
    ###### Validation ######
    model.eval()
    val_losses = []
    with torch.no_grad():
        dataIter = iter(val_loader)
        
        for i_iter in tqdm(range(len(dataIter)), 
                desc='test epoch [{:d}|{:d}]'.format(args.training.epoch, args.training.max_epoch)):
            
            batch = next(dataIter)
            batch = {key: item.to(args.training.device) for key, item in batch.items()}
            
            preds = model(batch['timeseries'], batch['fc'])
            loss, loss_by_tgts = criterion(preds, batch, args)
            loss /= args.training.accum_iter

            val_losses.append(loss_by_tgts)
            
    return train_losses, val_losses, np.array(train_lrs)

def train_STAGIN_one_epoch(model, train_loader, val_loader, 
                          args, optimizer, scheduler, criterion):
    ###### Training ######
    model.train()
    optimizer.zero_grad()
    train_losses = []
    train_lrs = []    
    dataIter = iter(train_loader)

    for i_iter in tqdm(range(len(dataIter)), 
            desc='train epoch [{:d}|{:d}]'.format(args.training.epoch, args.training.max_epoch)):
        batch = next(dataIter)
        batch['timeseries'] = batch['timeseries'].permute(0,2,1)
        # process the data
        dyn_a, sampling_points = process_dynamic_fc(batch['timeseries'], 
                                    args.model.window_size, args.model.window_stride, args.model.dynamic_length)
        sampling_endpoints = [p + args.model.window_size for p in sampling_points]

        if i_iter==0: dyn_v = repeat(torch.eye(args.model.num_nodes), 'n1 n2 -> b t n1 n2', 
                                t=len(sampling_points), b=args.training.batch_size)
        if len(dyn_a) < args.training.batch_size: dyn_v = dyn_v[:len(dyn_a)]
        t = batch['timeseries'].permute(1,0,2)

        # prediction
        batch = {key: item.to(args.training.device) for key, item in batch.items()}
        preds, _, _, _ = model(dyn_v.to(args.training.device), 
                               dyn_a.to(args.training.device), 
                               t.to(args.training.device), 
                               sampling_endpoints)
        loss, loss_by_tgts = criterion(preds, batch, args)
        loss /= args.training.accum_iter
        loss.backward()

        if (((i_iter + 1)) % args.training.accum_iter == 0) or ((i_iter + 1) == len(dataIter)):
            """ update model parameters """
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.training.clip_grad)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
        train_losses.append(loss_by_tgts)
        train_lrs.append(scheduler.get_last_lr())

    ###### Validation ######
    model.eval()
    val_losses = []
    with torch.no_grad():
        dataIter = iter(val_loader)
        
        for i_iter in tqdm(range(len(dataIter)), 
                desc='test epoch [{:d}|{:d}]'.format(args.training.epoch, args.training.max_epoch)):
            batch = next(dataIter)
            batch['timeseries'] = batch['timeseries'].permute(0,2,1)
            # process the data
            dyn_a, sampling_points = process_dynamic_fc(batch['timeseries'], 
                                        args.model.window_size, args.model.window_stride, args.model.dynamic_length)
            sampling_endpoints = [p + args.model.window_size for p in sampling_points]

            if i_iter==0: dyn_v = repeat(torch.eye(args.model.num_nodes), 'n1 n2 -> b t n1 n2', 
                                    t=len(sampling_points), b=args.training.batch_size)
            if len(dyn_a) < args.training.batch_size: dyn_v = dyn_v[:len(dyn_a)]
            t = batch['timeseries'].permute(1,0,2)

            # prediction
            batch = {key: item.to(args.training.device) for key, item in batch.items()}
            preds, _, _, _ = model(dyn_v.to(args.training.device), 
                                dyn_a.to(args.training.device), 
                                t.to(args.training.device), 
                                sampling_endpoints)
            loss, loss_by_tgts = criterion(preds, batch, args)
            loss /= args.training.accum_iter

            val_losses.append(loss_by_tgts)
    return train_losses, val_losses, np.array(train_lrs)

def baseline_model_training(baseline_id, kf_num_splits):

    def _process_data_split(kf_split_id, random_seed=42):
        def _save_model(model, savePath):
            print('save model to {:s} ...'.format(str(savePath)))
            torch.save(model.cpu().state_dict(), str(savePath))
            print('... done')
            model.to(args.training.device)
            return str(savePath)
        
        def _combine_additional_model_args(args):
            assert hasattr(args, 'model')
            assert hasattr(args, 'add_model_opts')
            if isinstance(args, DictConfig):
                for key in args.add_model_opts.keys():
                    args.model[key] = args.add_model_opts[key]
            else:
                for opt_name in args.add_model_opts._contains:
                    setattr(args.model, opt_name, getattr(args.add_model_opts, opt_name))
            return args
        
        pl.seed_everything(random_seed)
        # Ensure that all operations are deterministic on GPU (if used) for reproducibility
        torch.backends.cudnn.determinstic = True
        torch.backends.cudnn.benchmark = False

        """ *** load model arguments *** """
        args = load_model_args(baseline_id, use_DictConfig=True)

        train_dataset = NSDTransferDataset(settings, region_roi='Yeo100Parc',
                                output_continuous_targets = args.training.output_continuous_targets,
                                output_discrete_targets = args.training.output_discrete_targets,
                                standardized_output = True,
                                subject_id = args.training.subject_id,
                                output_fmri_size = args.training.output_fmri_size,
                                kf_split_id=kf_split_id,
                                overlapping_segments=args.training.overlapping_segments, 
                                normalize_fmri=args.training.normalize_fmri,
                                output_fc=True,
                                output_timeseries=True,
                                random_seed=random_seed)
        train_dataset.train()
        train_dataloader = DataLoader(train_dataset, batch_size=args.training.batch_size, shuffle=True, num_workers=8)
        args.training.sessIDs = [(row.SUBJECT, row.SESSION) for _, row in train_dataset.train_data_infos.iterrows()]
        assert train_dataset.kf_num_splits == kf_num_splits

        val_dataset = NSDTransferDataset(settings, region_roi='Yeo100Parc',
                                output_continuous_targets = args.training.output_continuous_targets,
                                output_discrete_targets = args.training.output_discrete_targets,
                                standardized_output = True,
                                output_fmri_size = args.training.output_fmri_size,
                                subject_id = args.training.subject_id,
                                kf_split_id=kf_split_id,
                                overlapping_segments=args.training.overlapping_segments, 
                                normalize_fmri=args.training.normalize_fmri,
                                output_fc=True,
                                output_timeseries=True,
                                random_seed=random_seed)
        val_dataset.test()
        val_dataset.tgt_norms = train_dataset.tgt_norms
        val_dataloader = DataLoader(val_dataset, batch_size=args.training.batch_size, shuffle=False, num_workers=8)
        
        args.training.eff_batch_size = args.training.batch_size * args.training.accum_iter
        print('process HCP-3T split-{:d}: training samples {:d}, test samples {:d}, num rois {:d}'.format(
                            kf_split_id, len(train_dataset), len(val_dataset), train_dataset.num_rois))

        """ construct model """
        assert hasattr(args, 'model')
        args = _combine_additional_model_args(args)
        if args.model.name == 'ModelSTAGIN':
            model = ModelSTAGIN(
                    input_dim=args.model.num_nodes,
                    hidden_dim=args.model.hidden_dim,
                    num_classes=args.model.output_dim,
                    num_heads=args.model.num_heads,
                    num_layers=args.model.num_layers,
                    sparsity=args.model.sparsity,
                    dropout=args.model.dropout,
                    cls_token=args.model.cls_token,
                    readout=args.model.readout,
                )
        else:
            model = eval(args.model.name)(args)
        model = model.to(args.training.device)

        param_groups = optim_factory.param_groups_weight_decay(model, args.training.weight_decay)
        optimizer = torch.optim.AdamW(param_groups, lr=args.training.learning_rate, betas=(0.9, 0.95))

        max_iters = args.training.max_epoch * len(train_dataloader)
        args.training.warmup_iters = args.training.warmup_epochs * len(train_dataloader)
        scheduler = CosineWarmupScheduler(optimizer, 
                                        warmup=int(args.training.warmup_iters / args.training.accum_iter), 
                                        max_iters=np.ceil(max_iters / args.training.accum_iter), 
                                        base_lr=args.training.learning_rate, 
                                        min_lr=args.training.min_learning_rate)
        criterion = transfer_learning_loss

        """ model training """
        train_loss_all = []
        val_loss_all = []
        train_lrs_all = []
        best_loss = np.inf

        saved_model_files = {}
        mdl_saveDir = settings.projectData.dir.general_NSD
        mdl_saveDir = mdl_saveDir / settings.projectData.rel_dir.general_NSD.Models
        for epoch in range(1, args.training.max_epoch+1):
            args.training.epoch = epoch

            if args.model.name == 'ModelSTAGIN':
                train_func = train_STAGIN_one_epoch
            else:
                train_func = train_model_one_epoch

            train_losses, val_losses, train_lrs = train_func(model, 
                                                            train_dataloader, 
                                                            val_dataloader, 
                                                            args,
                                                            optimizer, 
                                                            scheduler,
                                                            criterion)

            train_losses = pd.DataFrame(train_losses).mean(axis=0)
            val_losses = pd.DataFrame(val_losses).mean(axis=0)
            
            s = '[NSD split{:d}/{:d}] train loss = {:.4f}, test loss = {:.4f}, train lr = {:.4f}\n'.format(
                kf_split_id+1, kf_num_splits, train_losses.total, val_losses.total, train_lrs.mean())
            s += '  [Train]: '
            s += ', '.join(['{:s} = {:.3f}'.format(meas, getattr(train_losses, meas)) for meas in train_losses.iloc[:5].index])
            s += '\n  [Test]: '
            s += ', '.join(['{:s} = {:.3f}'.format(meas, getattr(val_losses, meas)) for meas in val_losses.iloc[:5].index])
            s += '\n'
            print(s)
            
            train_loss_all.append(train_losses)
            val_loss_all.append(val_losses)
            train_lrs_all.append(train_lrs)
            
            if val_losses.total < best_loss:
                print('update best validation loss from {:.3f} to {:.3f}'.format(best_loss, val_losses.total))
                best_loss = val_losses.total

                if epoch >= args.training.save_best_min_epochs:
                    mdl_fName = '{:s}_iter{:d}_best.pt'.format(baseline_id, kf_split_id)
                    mdl_fPath = mdl_saveDir / mdl_fName
                    _save_model(model, mdl_fPath)
                    saved_model_files['best'] = str(mdl_fPath)

            if (epoch % args.training.save_epochs_interval == 0) or (epoch == args.training.max_epoch):
                mdl_fName = '{:s}_iter{:d}_epoch_{:d}.pt'.format(baseline_id, kf_split_id, epoch)
                mdl_fPath = mdl_saveDir / mdl_fName
                _save_model(model, mdl_fPath)
                saved_model_files['epoch_{:d}'.format(epoch)] = str(mdl_fPath)

        saved_model_files = pd.DataFrame.from_dict(saved_model_files, 
                                                orient='index', columns=['file_path']).reset_index(names='checkpoint')

        modelInfo = DataContainer()
        modelInfo.iter_id = kf_split_id
        modelInfo.args = args
        modelInfo.train_losses = pd.concat(train_loss_all, axis=1).T
        modelInfo.val_losses = pd.concat(val_loss_all, axis=1).T
        modelInfo.learning_rates = np.array(train_lrs_all)
        modelInfo.model_files = saved_model_files
        modelInfo.used_random_seed = random_seed

        torch.cuda.empty_cache()
        return modelInfo
    
    varName = 'BaselineModels'
    fName = getattr(settings.projectData.files.general_NSD, varName)
    fPath = settings.projectData.dir.general_NSD / fName
    with open(fPath, 'rb') as handle:
        BaselineModels = pickle.load(handle)

    transferredInfo = DataContainer()
    for kf_split_id in range(kf_num_splits):
        modelInfo = _process_data_split(kf_split_id)
        
        attr = 'iter{:d}'.format(kf_split_id)
        setattr(transferredInfo, attr, modelInfo)

    setattr(BaselineModels, baseline_id, transferredInfo)

    var2save = ['BaselineModels']
    for varName in var2save:
        fName = getattr(settings.projectData.files.general_NSD, varName)
        fPath = settings.projectData.dir.general_NSD / fName

        with open(fPath, 'wb') as handle:
            print('save {:s} to {:s} ...'.format(varName, str(fPath)))
            pickle.dump(eval(varName), handle, protocol=pickle.HIGHEST_PROTOCOL)
            print('... done')

def initilization():
    BaselineModels = DataContainer()

    var2save = ['BaselineModels']
    for varName in var2save:
        fName = getattr(settings.projectData.files.general_NSD, varName)
        fPath = settings.projectData.dir.general_NSD / fName

        with open(fPath, 'wb') as handle:
            print('save {:s} to {:s} ...'.format(varName, str(fPath)))
            pickle.dump(eval(varName), handle, protocol=pickle.HIGHEST_PROTOCOL)
            print('... done')

def proc_baseline_training(resume=False):

    # baseline_model_training(baseline_id='BrainNetworkTransformer', kf_num_splits=5)

    # baseline_model_training(baseline_id='GraphTransformer', kf_num_splits=5)

    # baseline_model_training(baseline_id='FBNETGEN', kf_num_splits=5)

    # baseline_model_training(baseline_id='BrainNetCNN', kf_num_splits=5)

    baseline_model_training(baseline_id='STAGIN_SERO', kf_num_splits=5)

    # baseline_model_training(baseline_id='STAGIN_GARO', kf_num_splits=5)


def main():

    # initilization()

    proc_baseline_training()

if __name__ == "__main__":
    main()