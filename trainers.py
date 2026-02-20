import torch
torch.backends.cuda.matmul.allow_tf32 = True
import torch.nn.functional as F
import torch.nn as nn
import transformers
from omegaconf import DictConfig

import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    StateDictType,
    BackwardPrefetch,
    ShardingStrategy,
    CPUOffload,
)
from torch.distributed.fsdp.api import FullStateDictConfig, FullOptimStateDictConfig
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
import tensor_parallel as tp
import contextlib

from preference_datasets import get_batch_iterator
from utils import (
    slice_and_move_batch_for_device,
    formatted_dict,
    all_gather_if_needed,
    pad_to_length,
    get_block_class_from_model,
    rank0_print,
    get_local_dir,
)
import numpy as np
import wandb
import tqdm

import random
import os
from collections import defaultdict
import time
import json
import functools
from typing import Optional, Dict, List, Union, Tuple


def preference_loss(
    policy_chosen_logps: torch.FloatTensor,
    policy_rejected_logps: torch.FloatTensor,
    reference_chosen_logps: torch.FloatTensor,
    reference_rejected_logps: torch.FloatTensor,
    beta: float,
    label_smoothing: float = 0.0,
    ipo: bool = False,
    reference_free: bool = False,
    # Cal-DPO
    use_cal_dpo: bool = True,
    cal_chosen_target: Optional[float] = None,
    cal_rejected_target: Optional[float] = None,
    device: Optional[torch.device] = None,
) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
    """
    Cal-DPO
     = DPO/IPO + 0.5*(chosenMSE + rejectedMSE)
    """
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = reference_chosen_logps - reference_rejected_logps

    if reference_free:
        ref_logratios = 0

    logits = pi_logratios - ref_logratios  # h_{\pi_\theta}^{y_w,y_l}

    # DPO/IPO
    if ipo:
        base_losses = (logits - 1/(2 * beta)) ** 2  # IPO
    else:
        # DPO
        base_losses = (-F.logsigmoid(beta * logits) * (1 - label_smoothing) 
                      - F.logsigmoid(-beta * logits) * label_smoothing)

    # 
    chosen_rewards = beta * (policy_chosen_logps - reference_chosen_logps).detach()
    rejected_rewards = beta * (policy_rejected_logps - reference_rejected_logps).detach()

    # Cal-DPO
    if use_cal_dpo and cal_chosen_target is not None and cal_rejected_target is not None:
        # MSE
        cal_chosen_loss = F.mse_loss(
            chosen_rewards, 
            torch.full_like(chosen_rewards, cal_chosen_target, device=device)
        )
        cal_rejected_loss = F.mse_loss(
            rejected_rewards, 
            torch.full_like(rejected_rewards, cal_rejected_target, device=device)
        )
        # 0.5

        total_losses = base_losses + 0.5 * (cal_chosen_loss + cal_rejected_loss)
    else:
        total_losses = base_losses

    return total_losses, chosen_rewards, rejected_rewards


def _get_batch_logps(logits: torch.FloatTensor, labels: torch.LongTensor, average_log_prob: bool = False) -> torch.FloatTensor:
    """logits"""
    assert logits.shape[:-1] == labels.shape

    labels = labels[:, 1:].clone()
    logits = logits[:, :-1, :]
    loss_mask = (labels != -100)

    # mask
    labels[labels == -100] = 0

    per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)

    if average_log_prob:
        return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
    else:
        return (per_token_logps * loss_mask).sum(-1)


def concatenated_inputs(batch: Dict[str, Union[List, torch.LongTensor]]) -> Dict[str, torch.LongTensor]:
    """chosenrejected"""
    max_length = max(batch['chosen_input_ids'].shape[1], batch['rejected_input_ids'].shape[1])
    concatenated_batch = {}
    for k in batch:
        if k.startswith('chosen') and isinstance(batch[k], torch.Tensor):
            pad_value = -100 if 'labels' in k else 0
            concatenated_key = k.replace('chosen', 'concatenated')
            concatenated_batch[concatenated_key] = pad_to_length(batch[k], max_length, pad_value=pad_value)
    for k in batch:
        if k.startswith('rejected') and isinstance(batch[k], torch.Tensor):
            pad_value = -100 if 'labels' in k else 0
            concatenated_key = k.replace('rejected', 'concatenated')
            concatenated_batch[concatenated_key] = torch.cat((
                concatenated_batch[concatenated_key],
                pad_to_length(batch[k], max_length, pad_value=pad_value),
            ), dim=0)
    return concatenated_batch


class BasicTrainer(object):
    def __init__(self, policy: nn.Module, config: DictConfig, seed: int, run_dir: str, reference_model: Optional[nn.Module] = None, rank: int = 0, world_size: int = 1):
        """SFTDPOCal-DPO"""
        self.seed = seed
        self.rank = rank
        self.world_size = world_size
        self.config = config
        self.run_dir = run_dir

        # tokenizer
        tokenizer_name_or_path = config.model.tokenizer_name_or_path or config.model.name_or_path
        rank0_print(f'tokenizer: {tokenizer_name_or_path}')
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_name_or_path, cache_dir=get_local_dir(config.local_dirs))
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # 
        data_iterator_kwargs = dict(
            names=config.datasets,
            tokenizer=self.tokenizer,
            shuffle=True,
            max_length=config.max_length,
            max_prompt_length=config.max_prompt_length,
            sft_mode=config.loss.name == 'sft',
        )

        self.policy = policy
        self.reference_model = reference_model

        # 
        self.train_iterator = get_batch_iterator(** data_iterator_kwargs, split='train', n_epochs=config.n_epochs, 
                                           n_examples=config.n_examples, batch_size=config.batch_size, 
                                           silent=rank != 0, cache_dir=get_local_dir(config.local_dirs))
        rank0_print(f'')
        self.eval_iterator = get_batch_iterator(**data_iterator_kwargs, split='test', n_examples=config.n_eval_examples, 
                                          batch_size=config.eval_batch_size, silent=rank != 0, 
                                          cache_dir=get_local_dir(config.local_dirs))
        self.eval_batches = list(self.eval_iterator)
        rank0_print(f': {len(self.eval_batches)}{config.eval_batch_size}')

        # Cal-DPO
        self.use_cal_dpo = getattr(config.loss, 'use_cal_dpo', True)
        self.beta = 0.1##0.1ok20lossok
        # ±1/(2*beta)Cal-DPO
        self.cal_chosen_target = getattr(config.loss, 'cal_chosen_target', 1.0 / (2 * self.beta))
        self.cal_rejected_target = getattr(config.loss, 'cal_rejected_target', -1.0 / (2 * self.beta))
        rank0_print(f"Cal-DPO: use_cal_dpo={self.use_cal_dpo}, "
                   f"chosen={self.cal_chosen_target}, rejected={self.cal_rejected_target}")

    def get_batch_samples(self, batch: Dict[str, torch.LongTensor]) -> Tuple[str, str]:
        """"""
        # FSDP
        ctx = lambda: (FSDP.summon_full_params(self.policy, writeback=False, recurse=False) 
                      if 'FSDP' in self.config.trainer else contextlib.nullcontext())
        with ctx():
            policy_output = self.policy.generate(
                batch['prompt_input_ids'], attention_mask=batch['prompt_attention_mask'], 
                max_length=self.config.max_length, do_sample=True, pad_token_id=self.tokenizer.pad_token_id)

        if self.config.loss.name in {'dpo', 'ipo'}:
            ctx = lambda: (FSDP.summon_full_params(self.reference_model, writeback=False, recurse=False) 
                          if 'FSDP' in self.config.trainer else contextlib.nullcontext())
            with ctx():
                reference_output = self.reference_model.generate(
                    batch['prompt_input_ids'], attention_mask=batch['prompt_attention_mask'], 
                    max_length=self.config.max_length, do_sample=True, pad_token_id=self.tokenizer.pad_token_id)

        policy_output = pad_to_length(policy_output, self.config.max_length, self.tokenizer.pad_token_id)
        policy_output = all_gather_if_needed(policy_output, self.rank, self.world_size)
        policy_output_decoded = self.tokenizer.batch_decode(policy_output, skip_special_tokens=True)

        if self.config.loss.name in {'dpo', 'ipo'}:
            reference_output = pad_to_length(reference_output, self.config.max_length, self.tokenizer.pad_token_id)
            reference_output = all_gather_if_needed(reference_output, self.rank, self.world_size)
            reference_output_decoded = self.tokenizer.batch_decode(reference_output, skip_special_tokens=True)
        else:
            reference_output_decoded = []

        return policy_output_decoded, reference_output_decoded
    
    def concatenated_forward(self, model: nn.Module, batch: Dict[str, Union[List, torch.LongTensor]]) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """"""
        concatenated_batch = concatenated_inputs(batch)
        # float32Cal-DPOMSE
        all_logits = model(concatenated_batch['concatenated_input_ids'], 
                          attention_mask=concatenated_batch['concatenated_attention_mask']).logits.to(torch.float32)
        all_logps = _get_batch_logps(all_logits, concatenated_batch['concatenated_labels'], average_log_prob=False)
        chosen_logps = all_logps[:batch['chosen_input_ids'].shape[0]]
        rejected_logps = all_logps[batch['chosen_input_ids'].shape[0]:]
        return chosen_logps, rejected_logps

#  get_batch_metrics 
# metricsextend
# 

    def get_batch_metrics(self, batch: Dict[str, Union[List, torch.LongTensor]], loss_config: DictConfig, train=True):
        """Cal-DPO"""
        metrics = {}
        train_test = 'train' if train else 'eval'
        device = batch['chosen_input_ids'].device  # 

        if loss_config.name in {'dpo', 'ipo'}:
            # log
            policy_chosen_logps, policy_rejected_logps = self.concatenated_forward(self.policy, batch)
            # log
            with torch.no_grad():
                reference_chosen_logps, reference_rejected_logps = self.concatenated_forward(self.reference_model, batch)

            # 
            loss_kwargs = {
                'beta': loss_config.beta,
                'reference_free': getattr(loss_config, 'reference_free', False),
                'label_smoothing': getattr(loss_config, 'label_smoothing', 0.0),
                'ipo': loss_config.name == 'ipo',
                # Cal-DPO
                'use_cal_dpo': self.use_cal_dpo,
                'cal_chosen_target': self.cal_chosen_target,
                'cal_rejected_target': self.cal_rejected_target,
                'device': device
            }

            # Cal-DPO
            losses, chosen_rewards, rejected_rewards = preference_loss(
                policy_chosen_logps, policy_rejected_logps, 
                reference_chosen_logps, reference_rejected_logps,** loss_kwargs)

            # chosen > rejected
            reward_accuracies = (chosen_rewards > rejected_rewards).float()

            # 
            chosen_rewards = all_gather_if_needed(chosen_rewards, self.rank, self.world_size)
            rejected_rewards = all_gather_if_needed(rejected_rewards, self.rank, self.world_size)
            reward_accuracies = all_gather_if_needed(reward_accuracies, self.rank, self.world_size)

            # 
            metrics[f'rewards_{train_test}/chosen'] = chosen_rewards.cpu().numpy().tolist()
            metrics[f'rewards_{train_test}/rejected'] = rejected_rewards.cpu().numpy().tolist()
            metrics[f'rewards_{train_test}/accuracies'] = reward_accuracies.cpu().numpy().tolist()
            metrics[f'rewards_{train_test}/margins'] = (chosen_rewards - rejected_rewards).cpu().numpy().tolist()
            metrics[f'logps_{train_test}/rejected'] = all_gather_if_needed(
                policy_rejected_logps.detach(), self.rank, self.world_size).cpu().numpy().tolist()

            # Cal-DPO - 
            if self.use_cal_dpo:
                # 
                chosen_error = F.mse_loss(
                    chosen_rewards, torch.full_like(chosen_rewards, self.cal_chosen_target)
                ).cpu().item()
                rejected_error = F.mse_loss(
                    rejected_rewards, torch.full_like(rejected_rewards, self.cal_rejected_target)
                ).cpu().item()
                
                # 
                metrics[f'calibration_{train_test}/chosen_error'] = [chosen_error]
                metrics[f'calibration_{train_test}/rejected_error'] = [rejected_error]
                metrics[f'calibration_{train_test}/chosen_target'] = [self.cal_chosen_target]
                metrics[f'calibration_{train_test}/rejected_target'] = [self.cal_rejected_target]

        elif loss_config.name == 'sft':
            # SFT
            policy_chosen_logits = self.policy(batch['chosen_input_ids'], 
                                            attention_mask=batch['chosen_attention_mask']).logits.to(torch.float32)
            policy_chosen_logps = _get_batch_logps(policy_chosen_logits, batch['chosen_labels'], average_log_prob=False)
            losses = -policy_chosen_logps

        # 
        policy_chosen_logps = all_gather_if_needed(policy_chosen_logps.detach(), self.rank, self.world_size)
        metrics[f'logps_{train_test}/chosen'] = policy_chosen_logps.cpu().numpy().tolist()
        all_devices_losses = all_gather_if_needed(losses.detach(), self.rank, self.world_size)
        metrics[f'loss/{train_test}'] = all_devices_losses.cpu().numpy().tolist()

        return losses.mean(), metrics
        


    def train(self):
        """Cal-DPO"""
        rank0_print(f': {self.config.optimizer}')
        self.optimizer = getattr(torch.optim, self.config.optimizer)(
            self.policy.parameters(), lr=self.config.lr)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda=lambda step: min(1.0, (step + 1) / (self.config.warmup_steps + 1)))
    
        # 
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

        # 
        if self.config.loss.name in {'dpo', 'ipo'}:
            self.reference_model.eval()

        self.example_counter = 0
        self.batch_counter = 0
        last_log = None

        for batch in self.train_iterator:
            ####  ####
            if self.example_counter % self.config.eval_every == 0 and (self.example_counter > 0 or self.config.do_first_eval):
                rank0_print(f' {self.example_counter} ')
                self.policy.eval()

                all_eval_metrics = defaultdict(list)
                if self.config.sample_during_eval:
                    all_policy_samples, all_reference_samples = [], []
                    policy_text_table = wandb.Table(columns=["step", "prompt", "sample"])
                    if self.config.loss.name in {'dpo', 'ipo'}:
                        reference_text_table = wandb.Table(columns=["step", "prompt", "sample"])

                # 
                for eval_batch in (tqdm.tqdm(self.eval_batches, desc='') if self.rank == 0 else self.eval_batches):
                    local_eval_batch = slice_and_move_batch_for_device(eval_batch, self.rank, self.world_size, self.rank)
                    with torch.no_grad():
                        _, eval_metrics = self.get_batch_metrics(local_eval_batch, self.config.loss, train=False)

                    for k, v in eval_metrics.items():
                        all_eval_metrics[k].extend(v)

                # 
                if self.config.sample_during_eval:
                    if self.config.n_eval_model_samples < self.config.eval_batch_size:
                        rank0_print(f': n_eval_model_samples ({self.config.n_eval_model_samples}) < eval_batch_size ({self.config.eval_batch_size})')
                        sample_batches = self.eval_batches[:1]
                    else:
                        n_sample_batches = self.config.n_eval_model_samples // self.config.eval_batch_size
                        sample_batches = self.eval_batches[:n_sample_batches]
                    for eval_batch in (tqdm.tqdm(sample_batches, desc='') if self.rank == 0 else sample_batches):
                        local_eval_batch = slice_and_move_batch_for_device(eval_batch, self.rank, self.world_size, self.rank)
                        policy_samples, reference_samples = self.get_batch_samples(local_eval_batch)

                        all_policy_samples.extend(policy_samples)
                        all_reference_samples.extend(reference_samples)

                        for prompt, sample in zip(eval_batch['prompt'], policy_samples):
                            policy_text_table.add_data(self.example_counter, prompt, sample)
                        if self.config.loss.name in {'dpo', 'ipo'}:
                            for prompt, sample in zip(eval_batch['prompt'], reference_samples):
                                reference_text_table.add_data(self.example_counter, prompt, sample)

                # 
                mean_eval_metrics = {k: sum(v) / len(v) for k, v in all_eval_metrics.items()}
                rank0_print(f' ({self.example_counter}): {formatted_dict(mean_eval_metrics)}')
                if self.config.sample_during_eval:                    
                    rank0_print(json.dumps(all_policy_samples[:10], indent=2))
                    if self.config.loss.name in {'dpo', 'ipo'}:
                        rank0_print(json.dumps(all_reference_samples[:10], indent=2))

                # WandB
                if self.config.wandb.enabled and self.rank == 0:
                    wandb.log(mean_eval_metrics, step=self.example_counter)
                    if self.config.sample_during_eval:
                        wandb.log({"policy_samples": policy_text_table}, step=self.example_counter)
                        if self.config.loss.name in {'dpo', 'ipo'}:
                            wandb.log({"reference_samples": reference_text_table}, step=self.example_counter)

                # 
                if self.example_counter > 0 and not self.config.debug:
                    output_dir = os.path.join(self.run_dir, f'step-{self.example_counter}')
                    rank0_print(f' {output_dir}')
                    self.save(output_dir, mean_eval_metrics)

            ####  ####
            self.policy.train()

            start_time = time.time()
            batch_metrics = defaultdict(list)
            # 
            for microbatch_idx in range(self.config.gradient_accumulation_steps):
                global_microbatch = slice_and_move_batch_for_device(batch, microbatch_idx, self.config.gradient_accumulation_steps, self.rank)
                local_microbatch = slice_and_move_batch_for_device(global_microbatch, self.rank, self.world_size, self.rank)
                loss, metrics = self.get_batch_metrics(local_microbatch, self.config.loss, train=True)
                (loss / self.config.gradient_accumulation_steps).backward()

                for k, v in metrics.items():
                    batch_metrics[k].extend(v)

            # 
            grad_norm = self.clip_gradient()
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()

            # 
            step_time = time.time() - start_time
            examples_per_second = self.config.batch_size / step_time
            batch_metrics['examples_per_second'].append(examples_per_second)
            batch_metrics['grad_norm'].append(grad_norm)

            self.batch_counter += 1
            self.example_counter += self.config.batch_size

            # 
            if last_log is None or time.time() - last_log > self.config.minimum_log_interval_secs:
                mean_train_metrics = {k: sum(v) / len(v) for k, v in batch_metrics.items()}
                mean_train_metrics['counters/examples'] = self.example_counter
                mean_train_metrics['counters/updates'] = self.batch_counter
                rank0_print(f' ({self.example_counter}): {formatted_dict(mean_train_metrics)}')

                if self.config.wandb.enabled and self.rank == 0:
                    wandb.log(mean_train_metrics, step=self.example_counter)

                last_log = time.time()
            else:
                rank0_print(f'')


    def clip_gradient(self):
        """FSDP"""
        return torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.max_grad_norm).item()

    def write_state_dict(self, step: int, state: Dict[str, torch.Tensor], metrics: Dict, filename: str, dir_name: Optional[str] = None):
        """"""
        if dir_name is None:
            dir_name = os.path.join(self.run_dir, f'LATEST')

        os.makedirs(dir_name, exist_ok=True)
        output_path = os.path.join(dir_name, filename)
        rank0_print(f' {output_path}')
        torch.save({
            'step_idx': step,
            'state': state,
            'metrics': metrics if metrics is not None else {},
        }, output_path)
    
    def save(self, output_dir: Optional[str] = None, metrics: Optional[Dict] = None):
        """"""
        policy_state_dict = self.policy.state_dict()
        self.write_state_dict(self.example_counter, policy_state_dict, metrics, 'policy.pt', output_dir)
        del policy_state_dict

        optimizer_state_dict = self.optimizer.state_dict()
        self.write_state_dict(self.example_counter, optimizer_state_dict, metrics, 'optimizer.pt', output_dir)
        del optimizer_state_dict

        scheduler_state_dict = self.scheduler.state_dict()
        self.write_state_dict(self.example_counter, scheduler_state_dict, metrics, 'scheduler.pt', output_dir)


class FSDPTrainer(BasicTrainer):
    def __init__(self, policy: nn.Module, config: DictConfig, seed: int, run_dir: str, reference_model: Optional[nn.Module] = None, rank: int = 0, world_size: int = 1):
        """FSDPCal-DPO"""
        super().__init__(policy, config, seed, run_dir, reference_model, rank, world_size)
        assert config.model.block_name is not None, 'model.block_nameGPT2BlockGPTNeoXLayerFSDP'

        # 
        wrap_class = get_block_class_from_model(policy, config.model.block_name)
        model_auto_wrap_policy = functools.partial(
            transformer_auto_wrap_policy, 
            transformer_layer_cls={wrap_class},
        )

        # FSDP
        shared_fsdp_kwargs = dict(
            auto_wrap_policy=model_auto_wrap_policy,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            cpu_offload=CPUOffload(offload_params=False),
            backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
            device_id=rank,
            ignored_modules=None,
            limit_all_gathers=False,
            use_orig_params=False,
            sync_module_states=False
        )

        # FSDP
        rank0_print('...')
        mp_dtype = getattr(torch, config.model.fsdp_policy_mp) if config.model.fsdp_policy_mp is not None else None
        policy_mp_policy = MixedPrecision(
            param_dtype=mp_dtype, 
            reduce_dtype=mp_dtype, 
            buffer_dtype=mp_dtype
        )
        self.policy = FSDP(self.policy, **shared_fsdp_kwargs, mixed_precision=policy_mp_policy)

        # 
        if config.activation_checkpointing:
            rank0_print('...')
            try:
                from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
                    checkpoint_wrapper,
                    apply_activation_checkpointing,
                    CheckpointImpl,
                )
                non_reentrant_wrapper = functools.partial(
                    checkpoint_wrapper,
                    offload_to_cpu=False,
                    checkpoint_impl=CheckpointImpl.NO_REENTRANT,
                )
            except Exception as e:
                rank0_print('FSDP:', e)
            else:
                check_fn = lambda submodule: isinstance(submodule, wrap_class)
                rank0_print('...')
                apply_activation_checkpointing(self.policy, checkpoint_wrapper_fn=non_reentrant_wrapper, check_fn=check_fn)
                rank0_print('FSDP!')

        # FSDP
        if config.loss.name in {'dpo', 'ipo'}:
            rank0_print('...')
            self.reference_model = FSDP(self.reference_model,** shared_fsdp_kwargs)
        
        print(f'Rank {rank} ')
        dist.barrier()

    def clip_gradient(self):
        """FSDP"""
        return self.policy.clip_grad_norm_(self.config.max_grad_norm).item()
    
    def save(self, output_dir=None, metrics=None):
        """FSDPrank 0"""
        # 
        save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(self.policy, StateDictType.FULL_STATE_DICT, state_dict_config=save_policy):
            policy_state_dict = self.policy.state_dict()

        if self.rank == 0:
            self.write_state_dict(self.example_counter, policy_state_dict, metrics, 'policy.pt', output_dir)
        del policy_state_dict
        dist.barrier()

        # 
        save_optim = FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(self.policy, StateDictType.FULL_STATE_DICT, optim_state_dict_config=save_optim):
            optimizer_state_dict = FSDP.optim_state_dict(self.policy, self.optimizer)

        if self.rank == 0:
            self.write_state_dict(self.example_counter, optimizer_state_dict, metrics, 'optimizer.pt', output_dir)
        del optimizer_state_dict
        dist.barrier()

        # 
        if self.rank == 0:
            scheduler_state_dict = self.scheduler.state_dict()
            self.write_state_dict(self.example_counter, scheduler_state_dict, metrics, 'scheduler.pt', output_dir)
        dist.barrier()
        

class TensorParallelTrainer(BasicTrainer):
    def __init__(self, policy, config, seed, run_dir, reference_model=None, rank=0, world_size=1):
        """Cal-DPO"""
        super().__init__(policy, config, seed, run_dir, reference_model, rank, world_size)
        
        rank0_print('...')
        self.policy = tp.tensor_parallel(policy, sharded=True)
        if config.loss.name in {'dpo', 'ipo'}:
            rank0_print('...')
            self.reference_model = tp.tensor_parallel(reference_model, sharded=False)

    def save(self, output_dir=None, metrics=None):
        """"""
        with tp.save_tensor_parallel(self.policy):
            policy_state_dict = self.policy.state_dict()
    
        self.write_state_dict(self.example_counter, policy_state_dict, metrics, 'policy.pt', output_dir)
        del policy_state_dict

