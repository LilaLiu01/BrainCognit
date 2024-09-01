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

from settings import settings
from utils import *
from filters import filters
from funcs import *

import timm.optim.optim_factory as optim_factory
from datasets_transfer import NSDTransferDataset
from model_utils import CosineWarmupScheduler
from models_autoencoder import fMRIStateTransferModel

def load_model_args(transfer_id, verbose=True):
    def _to_DataContainer(obj):
        if type(obj) is dict:
            obj_ = DataContainer()
            setattr(obj_, '_contains', list(obj.keys()))
            for key, d in obj.items():
                setattr(obj_, key, _to_DataContainer(d))
            return obj_
        else:
            return obj
        
    fName = 'NSD_transfer.json'
    fPath = settings.CONFIG_DIR / fName
    with open(fPath, "r") as file:
        dat = json.load(file)
        dat = dat[transfer_id]

    if verbose:
        print('***** Model configuration *****')
        print(json.dumps(dat, indent=4))

    dat = _to_DataContainer(dat)
    return dat

def transfer_learning_loss(preds, batch, args, reg_preds=None, reg_loss=False):
    
    loss_by_tgts = {}
    losses = []
    for tgt_name in args.training.output_continuous_targets:
        idx = getattr(args.model.head_tgt2dims, tgt_name)
        loss = 0.5 * F.mse_loss(preds[:,idx], batch[tgt_name])
        losses.append(loss)
        loss_by_tgts[tgt_name] = loss.detach().cpu().item()

    for tgt_name in args.training.output_discrete_targets:
        idxs = getattr(args.model.head_tgt2dims, tgt_name)
        loss = F.cross_entropy(preds[:,idxs[0]:idxs[1]+1], batch[tgt_name])
        losses.append(loss)
        loss_by_tgts[tgt_name] = loss.detach().cpu().item()
        
    if reg_loss:
        for tgt_name in args.training.trainsient_targets:
            idx = getattr(args.model.reg_head_tgt2dims, tgt_name)
            _tgts = batch[tgt_name].reshape(-1)
            _pred = reg_preds[:,:,idx].reshape(-1)
            namsk = _tgts.isnan()
            loss = args.training.reg_loss_ratio * F.mse_loss(_pred[~namsk], _tgts[~namsk])
            losses.append(loss)
            loss_by_tgts[tgt_name] = loss.detach().cpu().item()
    
    loss = torch.stack(losses).sum()
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
        
        mask_ratio = np.random.random() * args.training.max_mask_ratio
        if model.has_reg_head:
            preds, reg_preds = model.forward(batch['fmri_segs'], mask_ratio=mask_ratio)
            loss, loss_by_tgts = criterion(preds, batch, args, reg_preds=reg_preds, reg_loss=True)
        else:
            preds = model.forward(batch['fmri_segs'], mask_ratio=mask_ratio)
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
            
            # mask_ratio = np.random.random() * args.training.max_mask_ratio
            mask_ratio = 0
            if model.has_reg_head:
                preds, reg_preds = model.forward(batch['fmri_segs'], mask_ratio=mask_ratio)
                loss, loss_by_tgts = criterion(preds, batch, args, reg_preds=None, reg_loss=False)
            else:
                preds = model.forward(batch['fmri_segs'], mask_ratio=mask_ratio)
                loss, loss_by_tgts = criterion(preds, batch, args)
            loss /= args.training.accum_iter

            val_losses.append(loss_by_tgts)
            
    return train_losses, val_losses, np.array(train_lrs)

def state_model_transfer_learning(transfer_id, kf_num_splits):

    def _process_data_split(kf_split_id, random_seed=42):
        def _save_model(model, savePath):
            print('save model to {:s} ...'.format(str(savePath)))
            torch.save(model.cpu().state_dict(), str(savePath))
            print('... done')
            model.to(args.training.device)
            return str(savePath)
        
        def _get_pretrained_model(pretrained_dataset,
                                pretrain_id,
                                check_point):
            
            varName = 'PretrainedStateModels'
            if pretrained_dataset == 'HCP3T':
                fName = getattr(settings.projectData.files.general_HCP3T, varName)
                fPath = settings.projectData.dir.general_HCP3T / fName
                with open(fPath, 'rb') as handle:
                    modelInfo = pickle.load(handle)
                    modelInfo = getattr(modelInfo, pretrain_id)
            elif pretrained_dataset == 'NSD':
                fName = getattr(settings.projectData.files.general_NSD, varName)
                fPath = settings.projectData.dir.general_NSD / fName
                with open(fPath, 'rb') as handle:
                    modelInfo = pickle.load(handle)
                    modelInfo = getattr(modelInfo, pretrain_id)
            else:
                raise NotImplementedError
            modelInfo.checkpoint_path = modelInfo.model_files.set_index('checkpoint').loc[check_point].file_path
            return modelInfo
        
        def _combine_additional_model_args(args):
            assert hasattr(args, 'model')
            assert hasattr(args, 'add_model_opts')
            for opt_name in args.add_model_opts._contains:
                setattr(args.model, opt_name, getattr(args.add_model_opts, opt_name))
            return args
        
        pl.seed_everything(random_seed)
        # Ensure that all operations are deterministic on GPU (if used) for reproducibility
        torch.backends.cudnn.determinstic = True
        torch.backends.cudnn.benchmark = False

        """ *** load model arguments *** """
        args = load_model_args(transfer_id)

        train_dataset = NSDTransferDataset(settings, region_roi='Yeo100Parc',
                                output_continuous_targets = args.training.output_continuous_targets,
                                output_discrete_targets = args.training.output_discrete_targets,
                                standardized_output = True,
                                output_fmri_size = args.training.output_fmri_size,
                                subject_id = args.training.subject_id,
                                kf_split_id=kf_split_id,
                                overlapping_segments=args.training.overlapping_segments, 
                                normalize_fmri=args.training.normalize_fmri,
                                random_seed=random_seed)
        train_dataset.train()
        train_dataloader = DataLoader(train_dataset, batch_size=args.training.batch_size, shuffle=True, num_workers=8)
        args.training.sessIDs = [(row.SUBJECT, row.SESSION, row.RUN) for _, row in train_dataset.train_data_infos.iterrows()]
        assert train_dataset.kf_num_splits == kf_num_splits

        val_dataset = NSDTransferDataset(settings, region_roi='Yeo100Parc',
                                output_continuous_targets = args.training.output_continuous_targets,
                                output_discrete_targets = args.training.output_discrete_targets,
                                standardized_output = True,
                                subject_id = args.training.subject_id,
                                output_fmri_size = 301,
                                kf_split_id=kf_split_id,
                                overlapping_segments=args.training.overlapping_segments, 
                                normalize_fmri=args.training.normalize_fmri,
                                random_seed=random_seed)
        val_dataset.test()
        val_dataset.tgt_norms = train_dataset.tgt_norms
        val_dataloader = DataLoader(val_dataset, batch_size=args.training.batch_size, shuffle=False, num_workers=8)

        args.training.eff_batch_size = args.training.batch_size * args.training.accum_iter
        print('process NSD split-{:d}: training samples {:d}, test samples {:d}, num rois {:d}'.format(
                            kf_split_id, len(train_dataset), len(val_dataset), train_dataset.num_rois))
        
        """ construct model """
        if args.training.use_pretrained_model:
            # use pretrained model
            pretrain_modelInfo = _get_pretrained_model(args.finetune_model.pretrained_dataset,
                                                    args.finetune_model.pretrain_id,
                                                    args.finetune_model.check_point)
            args.model = pretrain_modelInfo.args.model
            args = _combine_additional_model_args(args)
            args.model.head_norm_layer = getattr(nn, args.model.head_norm_layer)

            model = fMRIStateTransferModel(args.model)
            msg = model.load_state_dict(torch.load(pretrain_modelInfo.checkpoint_path), strict=False)
            if args.model.global_pool:
                assert (set(msg.missing_keys) == {'head.weight', 'head.bias', 'fc_norm.weight', 'fc_norm.bias'}) or \
                    (set(msg.missing_keys) == {'head.weight', 'head.bias', 'fc_norm.weight', 'fc_norm.bias',
                                                'reg_head.weight', 'reg_head.bias'})
            else:
                assert (set(msg.missing_keys) == {'head.weight', 'head.bias'}) or \
                    (set(msg.missing_keys) == {'head.weight', 'head.bias', 'reg_head.weight', 'reg_head.bias'})

            """ freeze all layers """
            if args.training.freeze_layers:
                for _, p in model.named_parameters():
                    p.requires_grad = False
            if args.training.freeze_embeddings:
                for _, p in model.tsn.roiEmbed.named_parameters():
                    p.requires_grad = False
        else:
            # trained from scratch
            assert hasattr(args, 'model')
            args.model.TSN_norm_layer = getattr(nn, args.model.TSN_norm_layer)
            args.model.AE_norm_layer = getattr(nn, args.model.AE_norm_layer)
            args.model.head_norm_layer = getattr(nn, args.model.head_norm_layer)

            model = fMRIStateTransferModel(args.model)

        """ fine-tuning final layers """
        layers2finetune = [model.blocks[-i] for i in range(1, args.training.finetune_depth+1)]
        layers2finetune += [model.norm, model.fc_norm, model.head]
        for layers in layers2finetune:
            for _, p in layers.named_parameters():
                p.requires_grad = True

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

            train_losses, val_losses, train_lrs = train_model_one_epoch(model, 
                                                            train_dataloader, 
                                                            val_dataloader, 
                                                            args,
                                                            optimizer, 
                                                            scheduler,
                                                            criterion)
            train_losses = pd.DataFrame(train_losses).mean(axis=0)
            val_losses = pd.DataFrame(val_losses).mean(axis=0)
            
            s = '[NSD split{:d}/{:d}] train loss = {:.4f}, test loss = {:.4f}, train lr = {:.4f}, best loss = {:.4f}\n'.format(
                kf_split_id+1, kf_num_splits, train_losses.total, val_losses.total, train_lrs.mean(), best_loss)
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
                    mdl_fName = '{:s}_iter{:d}_best.pt'.format(transfer_id, kf_split_id)
                    mdl_fPath = mdl_saveDir / mdl_fName
                    _save_model(model, mdl_fPath)
                    saved_model_files['best'] = str(mdl_fPath)

            if (epoch % args.training.save_epochs_interval == 0) or (epoch == args.training.max_epoch):
                mdl_fName = '{:s}_iter{:d}_epoch_{:d}.pt'.format(transfer_id, kf_split_id, epoch)
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
    
    varName = 'TransferredStateModels'
    fName = getattr(settings.projectData.files.general_NSD, varName)
    fPath = settings.projectData.dir.general_NSD / fName
    with open(fPath, 'rb') as handle:
        TransferredStateModels = pickle.load(handle)

    transferredInfo = DataContainer()
    for kf_split_id in range(kf_num_splits):
        modelInfo = _process_data_split(kf_split_id)
        
        attr = 'iter{:d}'.format(kf_split_id)
        setattr(transferredInfo, attr, modelInfo)

    setattr(TransferredStateModels, transfer_id, transferredInfo)

    var2save = ['TransferredStateModels']
    for varName in var2save:
        fName = getattr(settings.projectData.files.general_NSD, varName)
        fPath = settings.projectData.dir.general_NSD / fName

        with open(fPath, 'wb') as handle:
            print('save {:s} to {:s} ...'.format(varName, str(fPath)))
            pickle.dump(eval(varName), handle, protocol=pickle.HIGHEST_PROTOCOL)
            print('... done')

def initilization():
    TransferredStateModels = DataContainer()

    var2save = ['TransferredStateModels']
    for varName in var2save:
        fName = getattr(settings.projectData.files.general_NSD, varName)
        fPath = settings.projectData.dir.general_NSD / fName

        with open(fPath, 'wb') as handle:
            print('save {:s} to {:s} ...'.format(varName, str(fPath)))
            pickle.dump(eval(varName), handle, protocol=pickle.HIGHEST_PROTOCOL)
            print('... done')

def proc_pretrain_state_models(resume=False):

    # state_model_transfer_learning(transfer_id='AE_MaskTSN_TRANSFERv1', kf_num_splits=5)

    # state_model_transfer_learning(transfer_id='AE_DynamicMaskTSN_TRANSFERv1', kf_num_splits=5)

    # state_model_transfer_learning(transfer_id='AE_MaskTSN_TRANSFERv2', kf_num_splits=5)
    
    # state_model_transfer_learning(transfer_id='AE_DynamicMaskTSN_TRANSFERv2', kf_num_splits=5)

    # state_model_transfer_learning(transfer_id='AE_MaskTSN_TRANSFERv3', kf_num_splits=5)

    # state_model_transfer_learning(transfer_id='AE_MaskTSN_TRANSFERv4', kf_num_splits=5)

    state_model_transfer_learning(transfer_id='AE_MaskTSN_TRANSFERv5', kf_num_splits=5)

    # state_model_transfer_learning(transfer_id='AE_MaskTSN_TRANSFER_TREGv3', kf_num_splits=5)

    # state_model_transfer_learning(transfer_id='AE_DynamicMaskTSN_TRANSFERv3', kf_num_splits=5)

    # state_model_transfer_learning(transfer_id='AE_DynamicMaskTSN_TRANSFER_TREGv3', kf_num_splits=5)

    # state_model_transfer_learning(transfer_id='AE_VanillaTSN_TRANSFERv1', kf_num_splits=5)

    # state_model_transfer_learning(transfer_id='AE_VanillaTSN_TRANSFERv2', kf_num_splits=5)

def main():

    # initilization()

    proc_pretrain_state_models()

if __name__ == "__main__":
    main()