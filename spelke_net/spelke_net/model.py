"""
Pixel-wise Causal Counterfactual World Model (CCWM) implmentation

The model utilizes a GPT-style
"""

import math
import inspect
from typing import Tuple, Union, List
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import tqdm

from spelke_net.utils.model_wrapper import WrappedModel
from spelke_net.utils.modeling import LayerNorm, Block
from spelke_net.utils.image_processing import patchify, unpatchify, unpatchify_logits, convert_from_16bit_color



class LRAS(WrappedModel):

    def __init__(self, config):
        super().__init__(config)
        # print("using config:", config, flush=True)
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            token_embedding = nn.Embedding(config.vocab_size, config.n_embd),
            positional_embedding = nn.Embedding(config.block_size, config.n_embd),
            # positional_embedding = nn.Embedding(3*4096, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # self.lm_head = nn.Linear(config.n_embd, 512, bias=False)
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate

        # self.transformer.token_embedding.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))
        
        # set spmd mesh to None by default, set to the actual mesh if using spmd
        self.spmd_mesh = None
        self.unsharded_param_count = self.get_num_params()

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.positional_embedding.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        
    def forward(
            self, 
            seq: torch.Tensor, 
            pos: torch.Tensor = None, 
            tgt: torch.Tensor = None, 
            mask: torch.Tensor = None,
            exotic_mask: str = None,
        ) -> torch.Tensor:
        """
        Forward pass of the model

        Parameters:
            seq (torch.Tensor) of size b, t: The input sequence
            pos (torch.Tensor) of size b, t: The positional indices of the sequence
            tgt (torch.Tensor) of size b, t_tgt: The target sequence
            mask (torch.Tensor) of size b, t, t: The mask of the sequence
        
        Returns:
            torch.Tensor: The logits of the model. Size b, t if tgt is None, else b, t_tgt
        """

        # grab device to perform operations on
        device = seq.device
        # grab dimensions
        b, t = seq.size()
        if tgt is not None:
            b, t_tgt = tgt.size()

        assert t <= self.config.block_size, \
            f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        if tgt is not None:
            assert t_tgt <= t, \
                f"Target seqeunce length {t_tgt} must be shorter than or equal to sequence length {t}"

        assert t <= self.config.block_size, \
            f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"

        # create a tensor of position indices, if not provided
        if pos is None:
            pos = torch.arange(t, device=device).unsqueeze(0).expand(b, -1)

        # forward the GPT model itself
        tok_emb = self.transformer.token_embedding(seq) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.positional_embedding(pos) # position embeddings of shape (t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)

        if self.spmd_mesh is not None:
            import torch_xla.distributed.spmd.xla_sharding as xs
            xs.mark_sharding(x, self.spmd_mesh,  (('dcn', 'data'), None, 'model'))

        # create attention mask
        with torch.no_grad():
            # Generate a mask template which will be used to mask out the attention

            if len(mask.shape) == 2:
                
                ### WARNING: Currently skipping all custom masks during training and just forcing fully causal
                mask = torch.triu(torch.ones(t, t, 
                        device=x.device, requires_grad=False) * -float('inf'), diagonal=1).view(1, 1, t, t)
                
                if exotic_mask == "blockwise_parallel":
                    N = 2560
                    tokens_per_patch = self.config.patch_size**2 + 1
                    tokens_per_block = tokens_per_patch * 32
                    for i in range(N, t, tokens_per_block): 
                        # Mask the positions not within the same block
                        mask[0, 0, i+tokens_per_block:, i:i+tokens_per_block] = -float('inf')

                elif exotic_mask == "blockwise_parallel_campose_cond_32":
                    N = 2560 + 8
                    tokens_per_patch = self.config.patch_size**2 + 1
                    tokens_per_block = tokens_per_patch * 32
                    for i in range(N, t, tokens_per_block): 
                        # Mask the positions not within the same block
                        mask[0, 0, i+tokens_per_block:, i:i+tokens_per_block] = -float('inf')

                elif exotic_mask == "blockwise_parallel_campose_cond_32_flowcond":
                    N = 2560 + 750
                    tokens_per_patch = self.config.patch_size**2 + 1
                    tokens_per_block = tokens_per_patch * 32
                    for i in range(N, t, tokens_per_block): 
                        # Mask the positions not within the same block
                        mask[0, 0, i+tokens_per_block:, i:i+tokens_per_block] = -float('inf')

                elif exotic_mask is not None and 'blockwise_parallel_flow' in exotic_mask:
                    num_flow_patches = int(exotic_mask.split('_')[-1]) # indicate the num_flow_patches in the exotic string
                    tokens_per_flow_patch = self.config.patch_size**2 * 2 + 1
                    tokens_per_patch = self.config.patch_size ** 2 + 1
                    N = 2560 + tokens_per_flow_patch * num_flow_patches
                    tokens_per_block = tokens_per_patch * 32
                    for i in range(N, t, tokens_per_block): 
                        # Mask the positions not within the same block
                        mask[0, 0, i+tokens_per_block:, i:i+tokens_per_block] = -float('inf')
                
                elif exotic_mask == "no_mask":
                    mask = torch.zeros(t, t, device=x.device, requires_grad=False).view(1, 1, t, t)

                elif exotic_mask == "patchwise_parallel":
                    N = 2560 + 256
                    tokens_per_patch = self.config.patch_size**2 + 1
                    P = t - N

                    # # Define the size of the tensor
                    # N = 4096
                    # block_size = 5

                    # Initialize the tensor T with -inf
                    T = torch.full((P, P), -float('inf'))

                    # Create index tensors
                    i = torch.arange(P).unsqueeze(1).expand(P, P)  # Row indices
                    j = torch.arange(P).unsqueeze(0).expand(P, P)  # Column indices

                    # Compute positions within each block
                    pos_in_block_row = i % tokens_per_patch
                    pos_in_block_col = j % tokens_per_patch

                    # Create the mask: within each 5x5 block, set to True if row position >= column position
                    zero_mask = pos_in_block_row >= pos_in_block_col

                    # Set the desired positions to 0 according to the mask
                    T[zero_mask] = 0.0

                    # insert the patchwise parallel mask into the bottom right corner of the mask
                    mask[0, 0, N:, N:] = T

                    # # Create row and column indices
                    # row_indices = torch.arange(P).unsqueeze(1)
                    # col_indices = torch.arange(P).unsqueeze(0)
                    # # Compute the difference matrix D = i - j
                    # D = row_indices - col_indices
                    # # Initialize the tensor T with -inf
                    # T = torch.full((P, P), -float('inf'))
                    # # Create a mask where D >= 0 and D modulo 5 equals 0
                    # mask_zero = (D >= 0) & (D % 5 == 0)
                    # # Set the desired positions to 0 according to the mask
                    # T[mask_zero] = 0.0
                    # # insert the patchwise parallel mask into the bottom right corner of the mask
                    # mask[0, 0, N:, N:] = T

                # # The mask starts out with a "fully causal" mask, where each token can only attend to itself and tokens before it
                # full_mask = torch.triu(torch.ones(t, t, 
                #         device=x.device, requires_grad=False) * -float('inf'), diagonal=1).view(1, 1, t, t)
                # if self.spmd_mesh is not None:
                #     xs.mark_sharding(full_mask, self.spmd_mesh, (('dcn', 'data'), 'model', None, None))
                # # Next we set the "full attention cutoff" which is the number of leading tokens that can attend to each other
                # # These tokens represent frame 0 and since we are not predicting them, they should be able to attend to each other
                # if not causal_frame0:
                #     full_attention_cutoff = t - t_tgt + 1 # we assume that any token we are not decoding into a prediction can cross attend
                #     full_mask[:, :, :full_attention_cutoff, :full_attention_cutoff] = 0
                # # Finally we expand the provided custom sample specific mask
                # unfolded_mask = torch.einsum('bn, bm -> bnm', mask, mask).view(b, 1, t, t)
                # if self.spmd_mesh is not None:
                #     xs.mark_sharding(unfolded_mask, self.spmd_mesh, (('dcn', 'data'), 'model', None, None))
                # # We then combine the custom mask (which in practice specifies the attention patter of the lower right quadrant)
                # # with the generic mask template to create custom masks
                # mask = torch.masked_fill(full_mask, unfolded_mask.bool(), 0)
                
            if self.spmd_mesh is not None:
                xs.mark_sharding(mask, self.spmd_mesh, (('dcn', 'data'), 'model', None, None))

        for block in self.transformer.h:
            x = block(x, spmd_mesh=self.spmd_mesh, mask=mask)
            # print(x.shape, x.abs().mean().item(), x.abs().std().item(), x.min().item(), x.max().item(), flush=True)
        
        x = self.transformer.ln_f(x)

        # if tgt is not none, compute the logits for the entire sequence
        if tgt is None:
            logits = self.lm_head(x)
            return logits, None
        
        # if tgt is not none, compute the logits and the loss for the target sequence
        logits = self.lm_head(x[:, -tgt.size(1):])

        if self.spmd_mesh is not None:
            xs.mark_sharding(logits, self.spmd_mesh, (('dcn', 'data'), None, 'model'))

        # set all target tokens above 65535 to -1 so they are not included in the loss
        # we do this to ignore the prediction of the patch indexes since they are in random order
        # tgt[((65535+512) > tgt) & (tgt > 65535)] = -1
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1), ignore_index=-1)
        return logits, loss


    def sample_logits(self, logits: torch.FloatTensor, temp: float = 1.0, 
                      top_k: int = 1000, top_p: float = 0.9, allowed_tokens=None, 
                      blacklist: List[int] = None) -> torch.LongTensor:
        """
        Samples an integer from the distribution of logits

        Parameters:
            logits (torch.FloatTensor): The logits of the distribution
            temp (float): The temperature of the sampling, if 0.0, then argmax is used
            top_k (int): The number of top k tokens to consider during sampling
            top_p (float): The cumulative probability threshold for nucleus (top-p) sampling
            allowed_tokens (List[int]): The list of allowed tokens to sample from. If None, all tokens are allowed.
            blacklist (List[int]): The list of tokens to blacklist during sampling
        Returns:
            torch.LongTensor: The sampled integer
        """
        # Remove blacklisted tokens
        if blacklist is not None:
            logits[:, :, blacklist] = -float('Inf')
            # Set all logits not in uniform_flows to -inf
            # logits[:, :, [i for i in range(logits.size(-1)) if i not in uniform_flows]] = -float('Inf')
        else:
            pass #print("No blacklisted tokens provided", flush=True)

        # If temperature is 0.0, use argmax
        if temp == 0.0:
            return torch.argmax(logits, dim=-1)
        
        # Apply temperature
        logits = logits / temp

        # Apply top-k filtering if specified
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[..., [-1]]] = -float('Inf')

        # Apply top-p (nucleus) filtering if specified
        if top_p is not None:
            # Sort the logits in descending order
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            # Compute the sorted softmax probabilities
            sorted_probs = F.softmax(sorted_logits, dim=-1)
            # Compute the cumulative probabilities
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            # Create a mask for tokens to remove
            sorted_indices_to_remove = cumulative_probs > top_p
            # Shift the mask right to keep at least one token
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            # Scatter the mask back to the original indices
            indices_to_remove = sorted_indices_to_remove.scatter(dim=-1, index=sorted_indices, src=sorted_indices_to_remove)
            logits[indices_to_remove] = -float('Inf')
        
        if allowed_tokens is not None:
            mask = torch.ones_like(logits, dtype=torch.bool)
            allowed_tokens = torch.tensor(allowed_tokens, device=logits.device)
            mask[:, :, allowed_tokens] = False
            logits[mask] = -float('Inf')

        # Compute softmax probabilities
        probs = F.softmax(logits, dim=-1)
        # Flatten probabilities to (batch_size * sequence_length, vocab_size)
        flat_probs = probs.view(-1, probs.size(-1))
        # Sample from the distribution
        sampled = torch.multinomial(flat_probs, num_samples=1)
        # Reshape to original shape except for the last dimension
        sampled = sampled.view(*logits.shape[:-1])
        return sampled

    
    def sample_logits2(self, logits: torch.FloatTensor, temp: float = 0.9, top_k: int = None) -> torch.LongTensor:
        """
        Samples an integer from the distribution of logits

        Patameters:
            logits (torch.FloatTensor): The logits of the distribution
            temp (float): The temperature of the sampling, if 0.0, then argmax is used
            top_k (int): The number of top k tokens to consider during sampling
        Returns:
            torch.LongTensor: The sampled integer
        """
        # If temperature is 0.0, argmax is used
        if temp == 0.0:
            return torch.argmax(logits, dim=-1)
        
        # Apply temperature
        logits = logits / temp
        # Apply top_k
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, :, [-1]]] = -float('Inf')
        # Apply softmax
        probs = F.softmax(logits, dim=-1)
        # Flatten probs to batch x num_classes
        flat_ptobs = probs.view(-1, probs.size(-1))
        # Sample
        sampled = torch.multinomial(flat_ptobs, num_samples=1)
        # Reshape the sampled integers to the original logit shape, except
        # for the last dim (which got reduced since it was a distirbution we sampled from)
        sampled = sampled.view(*logits.shape[:-1])
        return sampled
        
    def compute_next_logits_and_update_kv_cache(self, k_cache, v_cache, current_token, current_token_pos):
        # forward the GPT model itself
        tok_emb = self.transformer.token_embedding(current_token)
        pos_emb = self.transformer.positional_embedding(current_token_pos)
        x = self.transformer.drop(tok_emb + pos_emb)

        k_list = []
        v_list = []
        for block_idx, block in enumerate(self.transformer.h):
            x, k, v = block(x, k_cache=k_cache[block_idx], v_cache=v_cache[block_idx])
            k_list.append(k)
            v_list.append(v)

        k_cache = torch.stack(k_list, dim=0)
        v_cache = torch.stack(v_list, dim=0)

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        return k_cache, v_cache, logits

    @torch.no_grad()
    def rollout_kv_cache_multiple_inputs(
        self, 
        cond_seq_list: List[torch.Tensor], 
        cond_pos_list: List[torch.Tensor], 
        scaling_factor_list: List[float], 
        patch_indexes: torch.Tensor, 
        rollout_pos: torch.Tensor,
        num_new_tokens: int = 4148, 
        temperature: float = 1.0, 
        top_k: int = None, 
        top_p: float = None):
        
        # basic checks and setup
        assert len(scaling_factor_list) == len(cond_seq_list) == len(cond_pos_list)
        print(f"Rolling out {len(cond_seq_list)} sequences with scaling factors {scaling_factor_list}")
        print(f"Each sequence have {cond_seq_list[0].size(1)} tokens, and will generate {num_new_tokens} tokens")
        n_tokens_per_patch = self.config.patch_size**2 + 1

        # compute the first kv cache and logits for each conditional sequence
        k_cache_list = []
        v_cache_list = []
        logits_list = []
        max_len_encode_kv_cache = 5000
        for cond_seq, cond_pos in zip(cond_seq_list, cond_pos_list):
            if cond_seq.size(1) > max_len_encode_kv_cache:
                cond_seq_temp = cond_seq[:, :max_len_encode_kv_cache]
                cond_pos_temp = cond_pos[:, :max_len_encode_kv_cache]
                k_cache, v_cache, logits = self.encode_kv_cache(cond_seq_temp, cond_pos_temp, return_last_logits=True)
                num_encode_per_iter = 100
                for i in tqdm.tqdm(range(max_len_encode_kv_cache, cond_seq.size(1), num_encode_per_iter)):
                    cond_seq_temp = cond_seq[:, i:i+num_encode_per_iter]
                    cond_pos_temp = cond_pos[:, i:i+num_encode_per_iter]
                    k_cache, v_cache, logits = self.encode_kv_cache(cond_seq_temp, cond_pos_temp, return_last_logits=True, k_cache=k_cache, v_cache=v_cache)
                if i+num_encode_per_iter < cond_seq.size(1):
                    cond_seq_temp = cond_seq[:, i+num_encode_per_iter:]
                    cond_pos_temp = cond_pos[:, i+num_encode_per_iter:]
                    k_cache, v_cache, logits = self.encode_kv_cache(cond_seq_temp, cond_pos_temp, return_last_logits=True, k_cache=k_cache, v_cache=v_cache)
            else:
                k_cache, v_cache, logits = self.encode_kv_cache(cond_seq, cond_pos, return_last_logits=True)
            k_cache_list.append(k_cache)
            v_cache_list.append(v_cache)
            logits_list.append(logits)

        # sample the first logits using scaling factor list
        logits_list = [logits * scaling_factor for logits, scaling_factor in zip(logits_list, scaling_factor_list)]
        logits = sum(logits_list) / len(logits_list)
        next_token = self.sample_logits(logits, temp=temperature, top_k=top_k, top_p=top_p)
        next_token = patch_indexes[:, 0].unsqueeze(-1)

        # rollout
        generated_seq = next_token
        all_logits = logits
        # num_new_tokens - 1 because we already sampled the first token
        for i in tqdm.tqdm(range(num_new_tokens-1)):
            current_token = next_token
            current_token_pos = rollout_pos[:, [i]]
            logits_list = []
            # # move things into cpu to save memory --> extremely slow. ~ 3 hours per rollout
            # k_cache_list = [k_cache.cpu() for k_cache in k_cache_list]
            # v_cache_list = [v_cache.cpu() for v_cache in v_cache_list]
            for j in range(len(k_cache_list)):
                k_cache = k_cache_list[j]
                v_cache = v_cache_list[j]
                k_cache, v_cache, logits = self.compute_next_logits_and_update_kv_cache(k_cache, v_cache, current_token, current_token_pos)
                k_cache_list[j] = k_cache
                v_cache_list[j] = v_cache
                logits_list.append(logits)

            # sample the logits using scaling factor list
            logits_list = [logits * scaling_factor for logits, scaling_factor in zip(logits_list, scaling_factor_list)]
            logits = sum(logits_list) / len(logits_list)
            next_token = self.sample_logits(logits, temp=temperature, top_k=top_k, top_p=top_p)

            # replace the token with the corresponding patch index
            if (i + 1) % n_tokens_per_patch == 0:
                next_token = patch_indexes[:, (i + 1) // n_tokens_per_patch].unsqueeze(-1)

            # append to the generated sequence & logits and continue
            generated_seq = torch.cat((generated_seq, next_token), dim=1)
            all_logits = torch.cat((all_logits, logits), dim=1)

        # all_seq = torch.cat(cond_seq_list + [generated_seq], dim=1)

        # return all_seq, all_logits
        return generated_seq, all_logits
    
    def encode_kv_cache(self, seq: torch.Tensor, pos: torch.Tensor, return_last_logits: bool = False,
                        k_cache: torch.Tensor = None, v_cache: torch.Tensor = None):
        """
        Encode the key and value cache for the given sequence

        Parameters:
            seq (torch.Tensor) of size b, t: The input sequence
        
        Returns:
            Tuple of torch.Tensor: The key and value cache of the sequence
        """

        # grab device to perform operations on
        device = seq.device
        # grab dimensions
        b, t = seq.size()

        assert t <= self.config.block_size, \
            f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"

        # create a tensor of position indices
        if pos is None:
            pos = torch.arange(t, device=device).unsqueeze(0).expand(b, -1)

        # forward the GPT model itself
        tok_emb = self.transformer.token_embedding(seq)
        pos_emb = self.transformer.positional_embedding(pos)
        x = self.transformer.drop(tok_emb + pos_emb)
        k_list = []
        v_list = []
        for block_idx, block in enumerate(self.transformer.h):
            if k_cache is not None and v_cache is not None:
                x, k, v = block(x, k_cache=k_cache[block_idx], v_cache=v_cache[block_idx], return_kv=True)
            else:
                x, k, v = block(x, return_kv=True)
            k_list.append(k)
            v_list.append(v)
        # k_cache and v_cache have shape (n_layer, b, n_head, t, n_embd//n_head)
        k_cache = torch.stack(k_list, dim=0)
        v_cache = torch.stack(v_list, dim=0)

        if return_last_logits:
            x = self.transformer.ln_f(x)
            logits = self.lm_head(x)
            logits = logits[:, [-1]]
            return k_cache, v_cache, logits
        else:
            return k_cache, v_cache
    
    @torch.no_grad()
    def rollout_kv_cache(
        self, 
        seq: torch.Tensor, 
        k_cache: torch.Tensor = None, 
        v_cache: torch.Tensor = None, 
        num_new_tokens: int = 4148, 
        temperature: float = 1.0, 
        return_cache: bool = False,
        patch_indexes: torch.Tensor = None, 
        pos: torch.Tensor = None,
        top_k = None, 
        top_p = None, 
        causal_mask_length: int = None,
        n_tokens_per_patch: int = None,
        sample_range: Tuple[int, int] = None,
        allowed_tokens: List[int] = None,
        sampling_blacklist: List[List[int]] = None,
        num_unmasked_tokens: int = 0,
        remaining_seq=None,
    ) -> torch.Tensor:
        """
        Rollout the key and value cache for the given sequence

        Parameters:
            seq (torch.Tensor) of size b, t: 
                The input sequence
            k_cache (torch.Tensor) of size n_layer, b, n_head, t, n_embd//n_head: 
                The key cache
            v_cache (torch.Tensor) of size n_layer, b, n_head, t, n_embd//n_head: 
                The value cache
            num_new_tokens (int): 
                The number of new tokens to rollout
            temperature (float): 
                The temperature of the sampling
            return_cache (bool):
                Whether to return the key and value cache
            patch_indexes (torch.Tensor) of size b, t:
                The indexes of the patches to rollout from
            pos (torch.Tensor) of size b, t:
                The positional indices of the sequence
            top_k (int):
                The number of top k tokens to consider during sampling
            top_p (float):
                The cumulative probability threshold for nucleus (top-p) sampling
        
        Returns:
            Tuple of torch.Tensor: 
                The key and value cache of the sequence
        """


        sample_range = None

        # if temp, top_k, and top_p are not lists, make them lists of length num_new_tokens
        if not isinstance(temperature, list):
            temperature = [temperature] * num_new_tokens
        if not isinstance(top_k, list):
            top_k = [top_k] * num_new_tokens
        if not isinstance(top_p, list):
            top_p = [top_p] * num_new_tokens

        # grab device to perform operations on

        # breakpoint()
        # print("num_unmasked_tokens", num_unmasked_tokens)

        device = seq.device
        # grab dimensions
        b, t = seq.size()
        # grab number of tokens per patch (patch_size^2 + 1 for the index token)
        if n_tokens_per_patch is None:
            n_tokens_per_patch = self.config.patch_size**2 + 1

        all_logits = []

        assert t <= self.config.block_size, \
            f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        
        # create a tensor of position indices if not provided
        if pos is None:
            pos = torch.arange(8710, device=device).unsqueeze(0).expand(b, -1)

        for i in tqdm.tqdm(range(num_new_tokens), desc='Rolling out sequence'):

            # create a tensor of position indices
            tok_pos = pos[:, :seq.size(1)]

            # forward the GPT model itself
            if i == 0: # if we are on the first pass push the entire sequence through
                tok_emb = self.transformer.token_embedding(seq)
                pos_emb = self.transformer.positional_embedding(tok_pos)
            else: # else just trim the last token and push it through
                tok_emb = self.transformer.token_embedding(seq[:, [-1]])
                pos_emb = self.transformer.positional_embedding(tok_pos[:, [-1]])
            x = self.transformer.drop(tok_emb + pos_emb)
            k_list = []
            v_list = []
            for block_idx, block in enumerate(self.transformer.h):
                # if k and v cache are passed in, use them
                if k_cache is not None and v_cache is not None:
                    x, k, v = block(x, k_cache=k_cache[block_idx], v_cache=v_cache[block_idx])
                # else if if k and v cache are not passed in compute them from the sequence
                else:
                    mask = None
                    if causal_mask_length is not None:
                        mask = torch.zeros(x.shape[1], x.shape[1], device=x.device).bool()
                        # first t - causal_mask_length tokens cannot attend to the last causal_mask_length tokens
                        mask[-causal_mask_length:, :-causal_mask_length] = True
                        # last causal_mask_length tokens can attend to all tokens before them (triangle shaped)
                        mask[-causal_mask_length:, -causal_mask_length:] = ~torch.triu(torch.ones(causal_mask_length, causal_mask_length, device=x.device).bool())
                        mask = mask.T
                    x, k, v = block(x, return_kv=True, mask=mask)
                k_list.append(k)
                v_list.append(v)
            # k_cache and v_cache have shape (n_layer, b, n_head, t, n_embd//n_head)
            k_cache = torch.stack(k_list, dim=0)
            v_cache = torch.stack(v_list, dim=0)

            x = self.transformer.ln_f(x)
            logits = self.lm_head(x[:, [-1]])
            # if sample range is not none, set all logits outside the range to -inf
            if sample_range is not None:
                logits[:, :, :sample_range[0]] = -float('inf')
                logits[:, :, sample_range[1]:] = -float('inf')
            # sample

            if i < num_unmasked_tokens:
                next_token = remaining_seq[i].view(1, 1)
            else:
                if i == 1:
                    next_token = self.sample_logits(
                        logits,
                        temp=temperature[i],
                        top_k=top_k[i],
                        top_p=top_p[i],
                        blacklist=sampling_blacklist, allowed_tokens=allowed_tokens
                    )
                else:
                    next_token = self.sample_logits(
                        logits,
                        temp=temperature[i],
                        top_k=top_k[i],
                        top_p=top_p[i],
                        blacklist=None, allowed_tokens=allowed_tokens
                    )

            # if this is an index token (evey 17th) and patch_indexes is not None
            # replace the token with the corresponding patch index
            # print(i)
            if i % n_tokens_per_patch == 0 and patch_indexes is not None:

                next_token = patch_indexes[:, i // n_tokens_per_patch].unsqueeze(-1)
                # print(i, next_token)
            # append to the sequence and continue
            seq = torch.cat((seq, next_token), dim=1)

            all_logits.append(logits)

        if return_cache:
            return seq, k_cache, v_cache
        
        all_logits = torch.cat(all_logits, dim=1)

        return seq, all_logits
    
    @torch.no_grad()
    def rollout_kv_cache_cfg(
        self, 
        seq: torch.Tensor, 
        k_cache: torch.Tensor = None, 
        v_cache: torch.Tensor = None, 
        num_new_tokens: int = 4148, 
        temperature: float = 1.0, 
        return_cache: bool = False,
        patch_indexes: torch.Tensor = None, 
        pos: torch.Tensor = None,
        top_k = None, 
        top_p = None, 
        causal_mask_length: int = None,
        n_tokens_per_patch: int = None,
        cfg_ratio: float = 2.0,
        allowed_tokens: List[int] = None,
    ) -> torch.Tensor:
        """
        Rollout the key and value cache for the given sequence

        Parameters:
            seq (torch.Tensor) of size b, t: 
                The input sequence
            k_cache (torch.Tensor) of size n_layer, b, n_head, t, n_embd//n_head: 
                The key cache
            v_cache (torch.Tensor) of size n_layer, b, n_head, t, n_embd//n_head: 
                The value cache
            num_new_tokens (int): 
                The number of new tokens to rollout
            temperature (float): 
                The temperature of the sampling
            return_cache (bool):
                Whether to return the key and value cache
            patch_indexes (torch.Tensor) of size b, t:
                The indexes of the patches to rollout from
            pos (torch.Tensor) of size b, t:
                The positional indices of the sequence
            top_k (int):
                The number of top k tokens to consider during sampling
            top_p (float):
                The cumulative probability threshold for nucleus (top-p) sampling
        
        Returns:
            Tuple of torch.Tensor: 
                The key and value cache of the sequence
        """

        # if temp, top_k, and top_p are not lists, make them lists of length num_new_tokens
        if not isinstance(temperature, list):
            temperature = [temperature] * num_new_tokens
        if not isinstance(top_k, list):
            top_k = [top_k] * num_new_tokens
        if not isinstance(top_p, list):
            top_p = [top_p] * num_new_tokens

        # grab device to perform operations on
        device = seq.device
        # grab dimensions
        b, t = seq.size()
        # grab number of tokens per patch (patch_size^2 + 1 for the index token)
        if n_tokens_per_patch is None:
            n_tokens_per_patch = self.config.patch_size**2 + 1

        all_logits = []

        assert t <= self.config.block_size, \
            f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        
        # create a tensor of position indices if not provided
        if pos is None:
            pos = torch.arange(8710, device=device).unsqueeze(0).expand(b, -1)

        for i in tqdm.tqdm(range(num_new_tokens), desc='Rolling out sequence'):

            # create a tensor of position indices
            tok_pos = pos[:, :seq.size(1)]

            # forward the GPT model itself
            if i == 0: # if we are on the first pass push the entire sequence through
                tok_emb = self.transformer.token_embedding(seq)
                pos_emb = self.transformer.positional_embedding(tok_pos)
            else: # else just trim the last token and push it through
                tok_emb = self.transformer.token_embedding(seq[:, [-1]])
                pos_emb = self.transformer.positional_embedding(tok_pos[:, [-1]])
            x = self.transformer.drop(tok_emb + pos_emb)
            # create the unconditional sequence cache
            if k_cache is None and v_cache is None:
                x_uncond = None
            else:
                x_uncond = self.transformer.drop(tok_emb + pos_emb)

            k_list = []
            v_list = []
            for block_idx, block in enumerate(self.transformer.h):
                # if k and v cache are passed in, use them
                if k_cache is not None and v_cache is not None:
                    x, k, v = block(x, k_cache=k_cache[block_idx], v_cache=v_cache[block_idx])
                    x_uncond, _, _ = block(x_uncond, k_cache=uncond_k_cache[block_idx], 
                                           v_cache=uncond_v_cache[block_idx])
                # else if if k and v cache are not passed in compute them from the sequence
                else:
                    mask = None
                    if causal_mask_length is not None:
                        mask = torch.zeros(x.shape[1], x.shape[1], device=x.device).bool()
                        # first t - causal_mask_length tokens cannot attend to the last causal_mask_length tokens
                        mask[-causal_mask_length:, :-causal_mask_length] = True
                        # last causal_mask_length tokens can attend to all tokens before them (triangle shaped)
                        mask[-causal_mask_length:, -causal_mask_length:] = ~torch.triu(torch.ones(causal_mask_length, causal_mask_length, device=x.device).bool())
                        mask = mask.T
                    x, k, v = block(x, return_kv=True, mask=mask)
                k_list.append(k)
                v_list.append(v)
            if k_cache is None and v_cache is None:
                uncond_k_cache = torch.stack(k_list, dim=0)
                uncond_v_cache = torch.stack(v_list, dim=0)
            # k_cache and v_cache have shape (n_layer, b, n_head, t, n_embd//n_head)
            k_cache = torch.stack(k_list, dim=0)
            v_cache = torch.stack(v_list, dim=0)
            # compute the logits
            x = self.transformer.ln_f(x)
            logits = self.lm_head(x[:, [-1]])
            # add the unconditional logits
            if x_uncond is not None:
                x_uncond = self.transformer.ln_f(x_uncond)
                logits_uncond = self.lm_head(x_uncond[:, [-1]])
                logits = logits + cfg_ratio * logits_uncond
            # sample
            next_token = self.sample_logits(logits, temp=temperature[i], top_k=top_k[i], top_p=top_p[i], allowed_tokens=allowed_tokens)
            # if this is an index token (evey 17th) and patch_indexes is not None
            # replace the token with the corresponding patch index
            if i % n_tokens_per_patch == 0 and patch_indexes is not None:
                next_token = patch_indexes[:, i // n_tokens_per_patch].unsqueeze(-1)
            # append to the sequence and continue
            seq = torch.cat((seq, next_token), dim=1)

            all_logits.append(logits)

        if return_cache:
            return seq, k_cache, v_cache
        
        all_logits = torch.cat(all_logits, dim=1)

        return seq, all_logits

    @torch.no_grad()
    def rollout_kv_cache_multiseq(
        self, 
        seq_list: list, 
        k_cache_list: list = None, 
        v_cache_list: list = None, 
        num_new_tokens: int = 4148, 
        temperature: float = 1.0, 
        return_cache: bool = False,
        rmi_list: list = None,
        pos_list: list = None,
        top_k = None, 
        top_p = None, 
        causal_mask_length: int = None,
        alpha: float = 1.0,
        beta: float = 0.0,
        gamma: float = 0.0
    ) -> torch.Tensor:
        """
        Rollout the key and value cache for the given sequence

        Parameters:
            seq (torch.Tensor) of size b, t: 
                The input sequence
            k_cache (torch.Tensor) of size n_layer, b, n_head, t, n_embd//n_head: 
                The key cache
            v_cache (torch.Tensor) of size n_layer, b, n_head, t, n_embd//n_head: 
                The value cache
            num_new_tokens (int): 
                The number of new tokens to rollout
            temperature (float): 
                The temperature of the sampling
            return_cache (bool):
                Whether to return the key and value cache
            patch_indexes (torch.Tensor) of size b, t:
                The indexes of the patches to rollout from
            pos (torch.Tensor) of size b, t:
                The positional indices of the sequence
            top_k (int):
                The number of top k tokens to consider during sampling
            top_p (float):
                The cumulative probability threshold for nucleus (top-p) sampling
        
        Returns:
            Tuple of torch.Tensor: 
                The key and value cache of the sequence
        """

        # if temp, top_k, and top_p are not lists, make them lists of length num_new_tokens
        if not isinstance(temperature, list):
            temperature = [temperature] * num_new_tokens
        if not isinstance(top_k, list):
            top_k = [top_k] * num_new_tokens
        if not isinstance(top_p, list):
            top_p = [top_p] * num_new_tokens

        seq = seq_list[0]
        pos = pos_list[0]
        patch_indexes = rmi_list[0]

        # grab device to perform operations on
        device = seq.device
        # grab dimensions
        b, t = seq.size()
        # grab number of tokens per patch (patch_size^2 + 1 for the index token)
        n_tokens_per_patch = self.config.patch_size**2 + 1

        all_logits = []

        assert t <= self.config.block_size, \
            f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        
        # create a tensor of position indices if not provided
        if pos is None:
            pos = torch.arange(8710, device=device).unsqueeze(0).expand(b, -1)

        k_cache_list = []
        v_cache_list = []
        for i in tqdm.tqdm(range(num_new_tokens), desc='Rolling out sequence'):

            logits_for_each_seq = []
            
            for seq_idx in range(len(seq_list)):
                seq = seq_list[seq_idx]
                pos = pos_list[seq_idx]

                # if i == 0, k_cache and v_cache do not exists. else it should exists
                k_cache = k_cache_list[seq_idx] if i != 0 else None
                v_cache = v_cache_list[seq_idx] if i != 0 else None

                tok_pos = pos[:, :seq.size(1)]

                # forward the GPT model itself
                if i == 0: # if we are on the first pass push the entire sequence through
                    tok_emb = self.transformer.token_embedding(seq)
                    pos_emb = self.transformer.positional_embedding(tok_pos)
                else: # else just trim the last token and push it through
                    tok_emb = self.transformer.token_embedding(seq[:, [-1]])
                    pos_emb = self.transformer.positional_embedding(tok_pos[:, [-1]])
                x = self.transformer.drop(tok_emb + pos_emb)
                k_list = []
                v_list = []
                for block_idx, block in enumerate(self.transformer.h):
                    # if k and v cache are passed in, use them
                    if k_cache is not None and v_cache is not None:
                        x, k, v = block(x, k_cache=k_cache[block_idx], v_cache=v_cache[block_idx])
                    # else if if k and v cache are not passed in compute them from the sequence
                    else:
                        mask = None
                        if causal_mask_length is not None:
                            mask = torch.zeros(x.shape[1], x.shape[1], device=x.device).bool()
                            # first t - causal_mask_length tokens cannot attend to the last causal_mask_length tokens
                            mask[-causal_mask_length:, :-causal_mask_length] = True
                            # last causal_mask_length tokens can attend to all tokens before them (triangle shaped)
                            mask[-causal_mask_length:, -causal_mask_length:] = ~torch.triu(torch.ones(causal_mask_length, causal_mask_length, device=x.device).bool())
                            mask = mask.T
                        x, k, v = block(x, return_kv=True, mask=mask)
                    k_list.append(k)
                    v_list.append(v)
                # k_cache and v_cache have shape (n_layer, b, n_head, t, n_embd//n_head)
                k_cache = torch.stack(k_list, dim=0)
                v_cache = torch.stack(v_list, dim=0)

                x = self.transformer.ln_f(x)
                logits = self.lm_head(x[:, [-1]])

                if i == 0:
                    k_cache_list.append(k_cache)
                    v_cache_list.append(v_cache)
                else:
                    k_cache_list[seq_idx] = k_cache
                    v_cache_list[seq_idx] = v_cache
                logits_for_each_seq.append(logits)


            logits = logits_for_each_seq[0] * alpha + logits_for_each_seq[-1] * beta
            if len(logits_for_each_seq) > 2:
                logits += sum(logits_for_each_seq[1:-1]) * (1 - alpha - beta) / (len(logits_for_each_seq) - 2)            
            
            # sample
            next_token = self.sample_logits(logits, temp=temperature[i], top_k=top_k[i], top_p=top_p[i])
            # if this is an index token (evey 17th) and patch_indexes is not None
            # replace the token with the corresponding patch index
            if i % n_tokens_per_patch == 0 and patch_indexes is not None:
                next_token = patch_indexes[:, i // n_tokens_per_patch].unsqueeze(-1)
            
            # append to the sequence and continue
            seq_list = [torch.cat((seq, next_token), dim=1) for seq in seq_list]

            all_logits.append(logits)

        if return_cache:
            return seq_list, k_cache_list, v_cache_list
        
        all_logits = torch.cat(all_logits, dim=1)

        return seq_list, all_logits

    @torch.no_grad()
    def rollout(
        self, 
        seq: torch.Tensor,
        pos: torch.Tensor,
        random_masked_indices,
        num_new_tokens: int,
        sampling_blacklist: List[List[int]] = None,
        temperature: Union[float, List[float]] = 1.0,
        top_k: Union[int, List[int]] = 1000,
        top_p: Union[float, List[float]] = 0.9,
        causal_mask_length: int = None,
        n_tokens_per_patch: int = None,
        cfg_ratio: float = None,
        allowed_tokens: List[int] = None,
        num_unmasked_tokens=0,
        remaining_seq=None,
    ):
        """
        Rollout an arbitrary number of tokens from the given sequence

        Parameters:
            x (torch.LongTensor) of shape b, t: 
                The input image of shape b, t, where t <= 2*img_seq_len. The shorter that t is,
                the higher the masking ratio
            pos (LongTensor) of shape b, t:
                The positional indices of the sequence
            random_masked_indices (torch.Tensor) of shape b, t:
                The indices of the patches to reveal during generation
            temperature (int or List[int]):
                The temperature of the sampling (optionally per token)
            top_k (int or List[int]):
                The number of top k tokens to consider during sampling (optionally per token)
            top_p (float or List[float]):
                The cumulative probability threshold for nucleus (top-p) sampling (optionally per token)
            causal_mask_length (int):
                The length of the causal mask
        
        Returns:
            img1_pred (torch.LongTensor) of shape b, img_seq_len:
                The predicted frame 1
        """

        # get the device
        device = seq.device
        # get the dimensions
        b, t = seq.size()
        # get the number of new tokens to generate, it is the number of indexes in 
        # random_masked_indices multiplied by (1 + patch_size^2), since each index is
        # followed by a patch of size patch_size x patch_size
        if n_tokens_per_patch is None:
            t_gen = num_new_tokens // (1 + self.config.patch_size**2)
        else: # add the else condition to handle flow where the n_tokens_per_patch is computed differently
            t_gen = num_new_tokens // n_tokens_per_patch

        # trimp the random_masked_indices to the number of tokens to generate
        random_masked_indices = random_masked_indices[:, :t_gen]
        # figure out the total number of tokens in the sequence
        t_total = t + num_new_tokens

        # Pass the sequence through the model to predict the next frame
        if cfg_ratio == 0.0 or cfg_ratio == None:
            # if we are not using classifier free guidance do a normal rollout
            pred_seq, logits = self.rollout_kv_cache(seq, num_new_tokens=num_new_tokens,
                temperature=temperature, patch_indexes=random_masked_indices, pos=pos, 
                top_k=top_k, top_p=top_p, causal_mask_length=causal_mask_length, 
                n_tokens_per_patch=n_tokens_per_patch, allowed_tokens=allowed_tokens, sampling_blacklist=sampling_blacklist,
                num_unmasked_tokens=num_unmasked_tokens, remaining_seq=remaining_seq)
        else:
            # else parallel rollout 
            pred_seq, logits = self.rollout_kv_cache_cfg(seq, num_new_tokens=num_new_tokens, 
                temperature=temperature, patch_indexes=random_masked_indices, pos=pos, 
                top_k=top_k, top_p=top_p, causal_mask_length=causal_mask_length, 
                n_tokens_per_patch=n_tokens_per_patch, cfg_ratio=cfg_ratio, allowed_tokens=allowed_tokens)

        return pred_seq, logits 

    @torch.no_grad()
    def factual_rollout_multiseq(
        self, 
        seq_list: list,
        pos_list: list,
        random_masked_indices: torch.Tensor,
        temperature: int = 1.0,
        top_k: int = None,
        top_p: float = None,
        causal_mask_length: int = None,
        rollout_params: dict = {}
    ) -> Tuple[np.array, np.array, np.array]:

        seq = seq_list[0]
        # get the device
        device = seq.device
        # get the dimensions
        b, t = seq.size()
        b, t_gen = random_masked_indices.size()
        # get the number of new tokens to generate, it is the number of indexes in 
        # random_masked_indices multiplied by (1 + patch_size^2), since each index is
        # followed by a patch of size patch_size x patch_size
        num_new_tokens = t_gen * (1 + self.config.patch_size**2)
        # figure out the total number of tokens in the sequence
        t_total = t + num_new_tokens

        # Pass the sequence through the model to predict the next frame
        pred_seq_list, logits = self.rollout_kv_cache_multiseq(seq_list, num_new_tokens=num_new_tokens, temperature=temperature,
                                            patch_indexes=random_masked_indices, pos_list=pos_list, top_k=top_k, top_p=top_p, 
                                            causal_mask_length=causal_mask_length, alpha=rollout_params['alpha'], beta=rollout_params['beta'], gamma=rollout_params['gamma'])
        
        if not torch.isclose(pred_seq_list[0], pred_seq_list[-1], atol=1e-5).all():
            print("Warning: The predicted sequences are not the same")
        # assert , "The predicted sequences are not the same"
        # assert pred_seq_list[0] == pred_seq_list[-1], "The predicted sequences are not the same"
        pred_seq = pred_seq_list[0]
        # Unpack the predicted frame1 from flat seq and order the patches according to the random indices
        # To do this we first need to grab the number of tokens in frame 1 (both prediced and seeded)
        # so we can reconstruct the whole frame properly
        num_patches_in_frame1 = self.config.rgb_patch_1_idx_range[1] - self.config.rgb_patch_1_idx_range[0]
        num_tokens_in_frame1 = (1 + self.config.patch_size**2) * num_patches_in_frame1
        # Grab the last num_tokens_in_frame1 tokens from the predicted sequence
        ordered_frame1_pred = self.unpack_and_sort_img_seq(pred_seq[:, -num_tokens_in_frame1:])

        return ordered_frame1_pred, logits

    @torch.no_grad()
    def factual_rollout(
        self, 
        seq: torch.Tensor,
        pos: torch.Tensor,
        random_masked_indices,
        temperature: int = 1.0,
        top_k: int = None,
        top_p: float = None,
        causal_mask_length: int = None,
    ) -> Tuple[np.array, np.array, np.array]:
        """
        Rollout next frame conditioned on the previous frame and parts of the current frame

        Parameters:
            x (torch.LongTensor) of shape b, t: 
                The input image of shape b, t, where t <= 2*img_seq_len. The shorter that t is,
                the higher the masking ratio
            pos (LongTensor) of shape b, t:
                The positional indices of the sequence
            random_masked_indices (torch.Tensor) of shape b, t:
                The indices of the patches to reveal during generation
            temperature (int): 
                The temperature of the sampling
        
        Returns:
            img1_pred (torch.LongTensor) of shape b, img_seq_len:
                The predicted frame 1
        """

        # get the device
        device = seq.device
        # get the dimensions
        b, t = seq.size()
        b, t_gen = random_masked_indices.size()
        # get the number of new tokens to generate, it is the number of indexes in 
        # random_masked_indices multiplied by (1 + patch_size^2), since each index is
        # followed by a patch of size patch_size x patch_size
        num_new_tokens = t_gen * (1 + self.config.patch_size**2)
        # figure out the total number of tokens in the sequence
        t_total = t + num_new_tokens

        # Pass the sequence through the model to predict the next frame
        pred_seq, logits = self.rollout_kv_cache(seq, num_new_tokens=num_new_tokens, temperature=temperature,
                                            patch_indexes=random_masked_indices, pos=pos, top_k=top_k, top_p=top_p, 
                                            causal_mask_length=causal_mask_length)
        
        # Unpack the predicted frame1 from flat seq and order the patches according to the random indices
        # To do this we first need to grab the number of tokens in frame 1 (both prediced and seeded)
        # so we can reconstruct the whole frame properly
        num_patches_in_frame1 = self.config.rgb_patch_1_idx_range[1] - self.config.rgb_patch_1_idx_range[0]
        num_tokens_in_frame1 = (1 + self.config.patch_size**2) * num_patches_in_frame1
        # Grab the last num_tokens_in_frame1 tokens from the predicted sequence
        ordered_frame1_pred = self.unpack_and_sort_img_seq(pred_seq[:, -num_tokens_in_frame1:])

        return ordered_frame1_pred, logits

    @torch.no_grad()
    def parallel_prediction(
            self, 
            seq: torch.Tensor, 
            pos: torch.Tensor,
            full_seq: torch.Tensor, 
            mask: torch.Tensor,
            n_tokens: int = 4148,
            causal_frame0: bool = False,
        ) -> torch.Tensor:
        """
        Forward pass of the model

        Parameters:
            seq (torch.Tensor) of size b, t: The input sequence
            pos (torch.Tensor) of size b, t: The positional indices of the sequence
            tgt (torch.Tensor) of size b, t_tgt: The target sequence
            mask (torch.Tensor) of size b, t, t: The mask of the sequence
        
        Returns:
            torch.Tensor: The logits of the model. Size b, t if tgt is None, else b, t_tgt
        """

        # grab device to perform operations on
        device = seq.device
        # grab dimensions
        b, t = seq.size()

        assert t <= self.config.block_size, \
            f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"

        # create a tensor of position indices, if not provided
        if pos is None:
            pos = torch.arange(t, device=device).unsqueeze(0).expand(b, -1)

        # forward the GPT model itself
        tok_emb = self.transformer.token_embedding(seq) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.positional_embedding(pos) # position embeddings of shape (t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)

        # create attention mask
        with torch.no_grad():
            # Generate a mask template which will be used to mask out the attention
            num_patches_in_frame0 = self.config.rgb_patch_0_idx_range[1] - self.config.rgb_patch_0_idx_range[0]

            # The mask starts out with a "fully causal" mask, where each token can only attend to itself and tokens before it
            full_mask = torch.triu(torch.ones(t, t, 
                    device=x.device, requires_grad=False) * -float('inf'), diagonal=1).view(1, 1, t, t)
            # Next we set the "full attention cutoff" which is the number of leading tokens that can attend to each other
            # These tokens represent frame 0 and since we are not predicting them, they should be able to attend to each other
            if not causal_frame0:
                full_attention_cutoff = num_patches_in_frame0 * (1 + self.config.patch_size**2) # frame 0 has full attention
                full_mask[:, :, :full_attention_cutoff, :full_attention_cutoff] = 0
            # Finally we expand the provided custom sample specific mask
            unfolded_mask = torch.einsum('bn, bm -> bnm', mask, mask).view(b, 1, t, t)
            # We then combine the custom mask (which in practice specifies the attention patter of the lower right quadrant)
            # with the generic mask template to create custom masks
            mask = torch.masked_fill(full_mask, unfolded_mask.bool(), 0)
            
        for block in self.transformer.h:
            x = block(x, spmd_mesh=self.spmd_mesh, mask=mask)
        x = self.transformer.ln_f(x)
        
        # if tgt is not none, compute the logits and the loss for the target sequence
        logits = self.lm_head(x[:, -n_tokens:])
        
        # set all target tokens above 65535 to -1 so they are not included in the loss
        # we do this to ignore the prediction of the patch indexes since they are in random order
        # tgt[((65535+512) > tgt) & (tgt > 65535)] = -1

        sampled_tokens = logits.argmax(dim=-1) # self.sample_logits(logits, temp=1.0) # logits.argmax(dim=-1)
        full_pred_seq = torch.cat([seq[:, :-(n_tokens-1)], sampled_tokens], dim=1)

        full_pred_seq_patches = full_pred_seq.view(b, -1, (1 + self.config.patch_size**2))
        full_seq_pathces = full_seq.view(b, -1, (1 + self.config.patch_size**2))

        # set the position tokens of full_pred_seq_patches to the position tokens of full_seq_patches
        full_pred_seq_patches[:, :, 0] = full_seq_pathces[:, :, 0]

        # unfold the patches into a sequence
        full_pred_seq = full_pred_seq_patches.view(b, -1)

        # Unpack the predicted frame1 from flat seq and order the patches according to the random indices
        # To do this we first need to grab the number of tokens in frame 1 (both prediced and seeded)
        # so we can reconstruct the whole frame properly
        num_patches_in_frame1 = self.config.rgb_patch_1_idx_range[1] - self.config.rgb_patch_1_idx_range[0]
        num_tokens_in_frame1 = (1 + self.config.patch_size**2) * num_patches_in_frame1

        # add fake logits for revealed patches to the logits
        logits = torch.cat([torch.zeros(b, num_tokens_in_frame1-logits.shape[1], logits.shape[2], device=logits.device), logits], dim=1)

        # Grab the last num_tokens_in_frame1 tokens from the predicted sequence
        ordered_frame1_pred, ordered_frame1_logits = self.unpack_and_sort_img_seq(full_pred_seq[:, -num_tokens_in_frame1:], logits=logits)

        return ordered_frame1_pred, ordered_frame1_logits
    

    @torch.no_grad()
    def campose_rollout(self, x: torch.Tensor, pos: torch.Tensor = None, temperature: int = 1.0, 
                        mask_patch_ratio: float = 1.0) -> Tuple[np.array, np.array, np.array]:
        """
        Rollout next frame conditioned on the previous frame and the change in camera pose
        specified as x, y, z translation and yaw, pitch, roll rotation

        Parameters:
            x (torch.Tensor) of shape b, 2*img_seq_len: 
                The input image of shape b, 2*img_seq_len+6, where img_seq_len is the length of
                a sqeunce of patch indexes and pixel tokens representing an image
            mask_patch_ratio (float): 
                The ratio of patches to mask
            temperature (int): 
                The temperature of the sampling
        
        Returns:
            Tuple of np.array: 
                The ground truth frame 1, ground truth frame 2, and predicted frame 2 each
                of which is a rgb image
        """

        # get the device
        device = x.device
        # get the dimensions
        b, t = x.size()
        t -= 6 # account for the 6 camera pose tokens

        # x is a sequence of frame 1 patches + frame 2 patches, so we split it into
        # the two component parts
        img1 = x[:, :t//2 + 6]
        img2 = x[:, -(t//2):]

        # reshape img2 into a sequence of patches each of which consist of 1 positional token
        # and 16 pixel value tokens for a tensor of shape b, t // (2*17), 17
        img2_patches = img2.view(b, -1, 17)

        # grab shuffled indices of frame 2 patches to reveal
        num_reveal_patches = int((1 - mask_patch_ratio) * img2_patches.size(1))
        random_indices = img2_patches[0, :, 0] - (65536 + 256)
        random_masked_indices = random_indices[num_reveal_patches:]
        revealed_patches = img2_patches[:, :num_reveal_patches]

        # get the number of new tokens to predict
        num_new_tokens = (img2_patches.size(1) - num_reveal_patches) * 17

        # concatenate the patches of img1 and the revealed patches of img2

        seq = torch.cat([img1, revealed_patches.view(b, -1)], dim=1)

        # shift the maksed indices to the right by 65536 + 256 to get the correct index
        # this way we decode the image in the order of patches determined by the random indices
        offset_masked_indexes = random_masked_indices.unsqueeze(0) + 65536 + 256
        # offset_masked_indexes = torch.sort(offset_masked_indexes)[0]

        # pass the sequence through the model to predict the next frame
        frame2_pred = self.rollout_kv_cache(seq, num_new_tokens=num_new_tokens, temperature=temperature, 
                                            patch_indexes=offset_masked_indexes, pos=pos)
        
        # Convert predicted frame from flat seq, to 16 bit color patches, to 24 bit rgb
        frame2_pred = self.unpack_and_sort_img_seq(frame2_pred[:, -t//2:], 
            num_revealed_patches=num_reveal_patches, mark_revealed_patches=63488)
        # Add red patches at the chosen indices of the predicted frame to display revealed patches
        frame2_pred = convert_from_16bit_color(frame2_pred.squeeze().cpu().numpy())

        # Convert gt frame 2 to 24 bit rgb
        img2 = self.unpack_and_sort_img_seq(img2)
        frame2_gt = convert_from_16bit_color(img2.squeeze().cpu().numpy())
        # Convert gt frame 1 to 24 bit rgb
        img1 = self.unpack_and_sort_img_seq(img1[:, :-6])
        frame1_gt = convert_from_16bit_color(img1.squeeze().cpu().numpy())

        return frame1_gt, frame2_gt, frame2_pred


    def counterfactual_campose_rollout(
            self, 
            x: torch.Tensor, 
            pos: torch.Tensor = None,
            random_masked_indices: torch.Tensor = None,
            temperature: int = 1.0,
        ) -> Tuple[np.array, np.array, np.array]:
        """
        Rollout next frame conditioned on the previous frame and the change in camera pose
        specified as x, y, z translation and yaw, pitch, roll rotation

        Parameters:
            x (torch.Tensor) of shape b, 2*img_seq_len: 
                The input image of shape b, 2*img_seq_len+6, where img_seq_len is the length of
                a sqeunce of patch indexes and pixel tokens representing an image
            pos (torch.Tensor) of shape b, t:
                The positional indices of the sequence
            random_masked_indices (torch.Tensor) of shape b, t:
                The indices of the patches to reveal
            temperature (int): 
                The temperature of the sampling
        
        Returns:
            Tuple of np.array: 
                The ground truth frame 1, ground truth frame 2, and predicted frame 2 each
                of which is a rgb image
        """

        # get the device
        device = x.device
        # get the dimensions
        b, t = x.size()
        t -= 6 # account for the 6 camera pose tokens

        # x is a sequence of frame 1 patches + frame 2 patches, so we split it into
        # the two component parts
        img1 = x[:, :t]

        # reshape img2 into a sequence of patches each of which consist of 1 positional token
        # and 16 pixel value tokens for a tensor of shape b, t // (2*17), 17
        img1_patches = img1.view(b, -1, 17)

        # grab shuffled indices of frame 2 patches to reveal
        if random_masked_indices is None:
            num_reveal_patches = 0
            random_indices = img1_patches[0, :, 0] - (65536)
            random_masked_indices = random_indices[num_reveal_patches:]

        # get the number of new tokens to predict
        num_new_tokens = t

        # concatenate the patches of img1 and the revealed patches of img2
        seq = x

        # shift the maksed indices to the right by 65536 + 256 to get the correct index
        # this way we decode the image in the order of patches determined by the random indices
        offset_masked_indexes = random_masked_indices.unsqueeze(0) + 65536 + 256
        # offset_masked_indexes = torch.sort(offset_masked_indexes)[0]

        # pass the sequence through the model to predict the next frame
        frame2_pred = self.rollout_kv_cache(seq, num_new_tokens=num_new_tokens, temperature=temperature, 
                                            patch_indexes=offset_masked_indexes, pos=pos)
        
        # Convert predicted frame from flat seq, to 16 bit color patches, to 24 bit rgb
        frame2_pred = self.unpack_and_sort_img_seq(frame2_pred[:, -t:])

        # Convert gt frame 1 to 24 bit rgb
        img1 = self.unpack_img_seq(img1)

        return img1, frame2_pred

    @torch.no_grad()
    def counterfactual_rollout(self, x, num_new_tokens=4096, temperature=1.0, masking_ratio=0.95,
                               reveal_patch_indices=None, reveal_patch_positions=None):
        """
        Rollout next frame conditioned 
        """

        ### Convert input images into flat sequences, select patches to use as conditioning

        # grab device to perform operations on
        device = x.device

        # grab batch size
        b, _, _, _ = x.size()

        # squeeze out the channel dimension
        img1 = x.squeeze(1)

        # convert images to patches
        patches1 = patchify(img1, patch_size=self.config.patch_size)

        # get number of patches from masking ratio
        n_patches = int(patches1.size(1) * masking_ratio)

        # sample n_patches patches from img2 patches for each sample in the batch
        if reveal_patch_indices is None:
            total_patches = patches1.size(1)
            noise = torch.rand((b, total_patches), device=device)
            _, patch_indices = torch.topk(noise, n_patches, dim=1)
        else:
            patch_indices = reveal_patch_indices.to(patches1.device)
        expanded_patch_indices = patch_indices.unsqueeze(-1).expand(-1, -1, patches1.size(2))

        # gather the top n_patches patches for each sample in the batch
        revealed_patches = torch.gather(patches1, 1, expanded_patch_indices)

        # flatten out the time dimension of the patches
        patches1 = patches1.reshape(b, -1)
        revealed_patches = revealed_patches.reshape(b, -1)

        # construct a sequence of frame 1 patches + revealed patches + frame 2 patches
        seq = torch.cat([patches1, revealed_patches], dim=1)

        # grab sequence length
        b, t = seq.size()

        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        # get number of patches in each image frame
        t_patches1 = patches1.size(1)
        img1_idxs = torch.arange(0, t_patches1, device=device).unsqueeze(0).expand(b, -1)
        img2_idxs = torch.arange(t_patches1, int(2 * t_patches1), device=device).unsqueeze(0).expand(b, -1)

        if reveal_patch_positions is None:
            selected_patch_idxs = 16 * patch_indices + int(2 * t_patches1)
        else:
            selected_patch_idxs = 16 * reveal_patch_positions.to(patches1.device) + int(2 * t_patches1)
        # Create a range tensor for extension
        extension_range = torch.arange(16, device=selected_patch_idxs.device).view(1, 1, -1)  # shape [1, 1, 16]
        # Expand the selected_patch_idxs tensor to match the extension range
        extended_idxs = selected_patch_idxs.unsqueeze(-1) + extension_range  # shape [2, 12, 16]
        # Reshape to the final desired shape
        extended_idxs = extended_idxs.view(b, selected_patch_idxs.size(1) * 16)

        pos = torch.cat((img1_idxs, extended_idxs, img2_idxs), dim=1)


        ### Grab KV Cache of img1 + unmasked patches

        k_list = []
        v_list = []

        # forward the GPT model itself
        tok_emb = self.transformer.token_embedding(seq) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.positional_embedding(pos[:, :seq.size(1)]) # position embeddings of shape (t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x, k, v = block(x, return_kv=True)
            k_list.append(k)
            v_list.append(v)
        # k_cache and v_cache have shape (n_layer, b, n_head, t, n_embd//n_head)
        k_cache = torch.stack(k_list, dim=0)
        v_cache = torch.stack(v_list, dim=0)

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x[:, [-1]])
        # sample
        next_pixel = self.sample_logits(logits, temp=temperature)
        # append to the sequence and continue
        seq = torch.cat((seq, next_pixel), dim=1)


        ### Rollout img2 autoregressively using kv caches

        for i in tqdm.tqdm(range(1, num_new_tokens), desc='rolling out frame 2'):

            # forward the GPT model itself
            tok_emb = self.transformer.token_embedding(seq[:, [-1]]) # token embeddings of shape (b, t, n_embd)
            pos_emb = self.transformer.positional_embedding(pos[:, [seq.size(1) - 1]]) # position embeddings of shape (t, n_embd)
            x = self.transformer.drop(tok_emb + pos_emb)
            k_list = []
            v_list = []
            for block_idx, block in enumerate(self.transformer.h):
                x, k, v = block(x, k_cache=k_cache[block_idx], v_cache=v_cache[block_idx])
                k_list.append(k)
                v_list.append(v)
            x = self.transformer.ln_f(x)
            # create the cache with the new embeddings
            k_cache = torch.stack(k_list, dim=0)
            v_cache = torch.stack(v_list, dim=0)

            logits = self.lm_head(x[:, [-1]])
            # sample
            next_pixel = self.sample_logits(logits, temp=temperature)
            # append to the sequence and continue
            seq = torch.cat((seq, next_pixel), dim=1)

        # Convert predicted frame from flat seq, to 16 bit color patches, to 24 bit rgb
        frame2_pred = seq[:, -num_new_tokens:]
        # Add red patches at the chosen indices of the predicted frame to display revealed patches
        # frame2_pred[:, extended_idxs - int(2 * t_patches1)] = 63488 # 65535
        frame2_pred = unpatchify(frame2_pred.view(b, -1, 16))
        frame2_pred = convert_from_16bit_color(frame2_pred.squeeze().cpu().numpy())

        # Convert gt frame 1 to 24 bit rgb
        frame1_gt = convert_from_16bit_color(img1.squeeze().cpu().numpy())

        return frame1_gt, frame2_pred

    def predict_campose(
            self, 
            x: torch.Tensor, 
            pos: torch.Tensor = None,
            temperature: int = 1.0,
        ) -> Tuple[np.array, np.array, np.array]:
        """
        Rollout the campose tokens given frames 1 and 2

        Parameters:
            x (torch.Tensor) of shape b, 2*img_seq_len: 
                The input image of shape b, 2*img_seq_len+6, where img_seq_len is the length of
                a sqeunce of patch indexes and pixel tokens representing an image
            pos (torch.Tensor) of shape b, t:
                The positional indices of the sequence
            random_masked_indices (torch.Tensor) of shape b, t:
                The indices of the patches to reveal
            temperature (int): 
                The temperature of the sampling
        
        Returns:
            Tuple of np.array: 
                The ground truth frame 1, ground truth frame 2, and predicted frame 2 each
                of which is a rgb image
        """

        # get the device
        device = x.device
        # get the dimensions
        b, t = x.size()
        t -= 6 # account for the 6 camera pose tokens

        # x is a sequence of frame 1 patches + frame 2 patches, so we split it into
        # the two component parts
        img1 = x[:, :t//2]
        img2 = x[:, t//2:-6]
        campose = x[:, -6:]

        # get the number of new tokens to predict
        num_new_tokens = 6

        # Get rid of the last 6 tokens
        no_campose_seq = x[:, :-6]

        # pass the sequence through the model to predict the next frame
        pred_campose = self.rollout_kv_cache(no_campose_seq, num_new_tokens=num_new_tokens, temperature=temperature, 
                                            pos=pos)[:, -6:]
        
        # Convert gt frame 1 to 24 bit rgb
        img1 = self.unpack_and_sort_img_seq(img1)
        img2 = self.unpack_and_sort_img_seq(img2)

        return img1, img2, campose, pred_campose

    def unpack_img_seq(self, img_seq: torch.Tensor) -> torch.Tensor:
        """
        Unpacks a sequence of indices and image toknes into a 2d image tensor

        Parameters:
            img_seq (torch.Tensor) of size b, t:
                The input image sequence, where t is 17 * the number of image patches if patch_size is 4
        
        Returns:
            torch.Tensor: The unpacked image of size b, 16 * sqrt(t/17), 16 * sqrt(t/17) if patch_size is 4
        """

        # reshape the image sequence into a sequence of patches and 
        # trim the first element in the 2nd dim (the index)
        n_tokens_per_patch = self.config.patch_size**2 + 1 # compute the number of tokens per patch
        img_seq = img_seq.view(img_seq.size(0), -1, n_tokens_per_patch)[:, :, 1:]

        # unpatchify the image sequence
        img = unpatchify(img_seq)

        return img

    def unpack_and_sort_img_seq(self, img_seq: torch.Tensor, num_revealed_patches: int = 0, mark_revealed_patches: int = None, 
                                logits: torch.Tensor = None, unpatchify_seq=True) -> torch.Tensor:
        """
        Unpacks a sequence of indices and image toknes into a 2d image tensor and sorts the patches
        according to the patch indices

        Parameters:
            img_seq (torch.Tensor) of size b, t:
                The input image sequence, where t is 17 * the number of image patches if patch_size is 4
            num_revealed_patches (int):
                The number of revealed patches to mark
            mark_revealed_patches (int):
                The value to mark the revealed patches with
            logits (torch.Tensor) of size b, t, n_tokens_per_patch, n_classes:
                The logits of the prediction
                
        Returns:
            torch.Tensor: The unpacked image of size b, 16 * sqrt(t/17), 16 * sqrt(t/17) if patch_size is 4
        """

        # reshape the image sequence into a sequence of patches and
        # trim the first element in the 2nd dim (the index)
        n_tokens_per_patch = self.config.patch_size**2 + 1 # compute the number of tokens per patch
        img_seq = img_seq.view(img_seq.size(0), -1, n_tokens_per_patch)
        img_idxs = img_seq[:, :, 0].long() - (65536 + 256) # shift the indices to the left by 65536 + 256
        reconstruct_indxs = torch.argsort(img_idxs, dim=1)
        rgb_seq = img_seq[:, :, 1:]

        # color the firs num_revealed_patches patches with the specified value if mark_revealed_patches is not None
        if mark_revealed_patches is not None:
            rgb_seq[:, :num_revealed_patches] = mark_revealed_patches

        # reorded the patches according to the patch indices
        rgb_seq = rgb_seq[torch.arange(img_seq.size(0)).unsqueeze(1).expand(*img_idxs.shape), reconstruct_indxs]

        # unpatchify the image sequence
        if unpatchify_seq:
            img = unpatchify(rgb_seq)
        else:
            h = w = int(math.sqrt(rgb_seq.size(1)))
            img = rgb_seq.reshape(rgb_seq.size(0), h, w, -1)

        if logits is not None:
            logits = logits.view(logits.size(0), -1, n_tokens_per_patch, logits.size(-1))
            logits = logits[:, :, 1:]
            logits = logits[torch.arange(img_seq.size(0)).unsqueeze(1).expand(*img_idxs.shape), reconstruct_indxs]
            logits = unpatchify_logits(logits)
            return img, logits
        
        return img

    def unpack_and_sort_flow_seq(
            self, 
            flow_seq_pred: torch.Tensor, 
            num_revealed_patches: int = 0,
            mark_revealed_patches: int = None,
            flow_size: int = 64
        ) -> torch.Tensor:
        
        """
        Unpacks a sequence of flow tokens into a sorted 2D flow tensor.

        Takes a sequence of flow tokens and patch indices and reconstructs the original 2D flow field,
        with optional marking of revealed patches.

        Parameters:
            flow_seq_pred: Predicted flow sequence tensor of shape (batch_size, num_tokens)
                Each patch contains patch_index followed by x,y flow values
            num_revealed_patches: Number of patches to mark as revealed (default: 0)
            mark_revealed_patches: Value to mark revealed patches with (default: None)
            flow_size: Size of flow field (default: 64)

        Returns:
            Tuple containing:
            - Flow tensor of shape (batch_size, 2, height, width) containing x,y flow components
            - Valid mask tensor of shape (batch_size, height, width) indicating which patches were predicted
        """
    
        batch_size = flow_seq_pred.shape[0]
        patch_size = 1 # self.config.patch_size
        num_patches = (flow_size // patch_size) ** 2
        tokens_per_patch = 3 # patch_size ** 2 * 2 + 1  # 1 patch index, 4 flow vectors per patch, 2 codes per vector (x,y)
        device = flow_seq_pred.device
        

        # Reshape sequence into patches and separate patch indices from flow values
        flow_seq_pred = flow_seq_pred.view(batch_size, -1, tokens_per_patch)
        patch_idx_pred = flow_seq_pred[:, :, 0] - self.config.flow_patch_idx_range[0]
        flow_code_pred = flow_seq_pred[:, :, 1:]

        # Mark revealed patches if specified
        if mark_revealed_patches is not None:
            flow_code_pred[:, :num_revealed_patches] = mark_revealed_patches

        # Reconstruct full flow codes of shape [B, num_patches, tokens_per_patch-1] by sorting patches
        batch_indices = torch.arange(batch_size).unsqueeze(1).expand(*patch_idx_pred.shape)
        flow_codes_full_shape = (batch_size, num_patches, tokens_per_patch - 1)

        if flow_code_pred.size(1) == num_patches:
            # If we have all patches, just sort them
            sort_indices = torch.argsort(patch_idx_pred, dim=1)
            flow_codes_full = flow_code_pred[batch_indices, sort_indices]
            valid_mask = torch.ones(flow_codes_full_shape).bool().to(device)
        else:
            # If some patches are missing, create a placeholder of shape [B, num_patches, tokens_per_patch-1] with zeros
            flow_codes_full = torch.zeros(flow_codes_full_shape)
            flow_codes_full = flow_codes_full.to(device).long()
            flow_codes_full[batch_indices, patch_idx_pred] = flow_code_pred
            valid_mask = torch.zeros(flow_codes_full_shape).bool().to(device)
            valid_mask[batch_indices, patch_idx_pred] = True

        # Separate and reshape x,y components
        flow_codes_full = flow_codes_full.view(batch_size, num_patches, patch_size ** 2, 2)
        flow_codes_full = flow_codes_full - self.config.flow_range[0]
        valid_mask = valid_mask.view(batch_size, num_patches, patch_size ** 2, 2)[..., 0]
        
        # Unpatchify x and y components separately
        flow_x = unpatchify(flow_codes_full[..., 0])
        flow_y = unpatchify(flow_codes_full[..., 1])
        valid_mask = unpatchify(valid_mask)

        # Stack into final flow tensor
        flow = torch.stack([flow_x, flow_y], dim=1)

        return flow, valid_mask
    
    def unpack_and_sort_campose_seq(self, campose_seq: torch.Tensor) -> torch.Tensor:
        """
        Unpacks a sequence of indices and campose tokens into a 2d tensor

        Parameters:
            campose_seq (torch.Tensor) of size b, t:
                The input campose sequence, where t is 6
        
        Returns:
            torch.Tensor: The unpacked campose of size b, 6
        """
        num_campose_pos_range = self.config.campose_pos_range[1] - self.config.campose_pos_range[0]
        # reshape the campose sequence into a sequence of campose tokens
        campose_seq_model_input = campose_seq.view(campose_seq.size(0), -1, num_campose_pos_range)[:, :, 1:]
        campose_seq_data = campose_seq_model_input - self.config.campose_range[0]

        return campose_seq_data

    # def unpack_and_sort_img_seq(self, img_seq: torch.Tensor, num_revealed_patches: int = 0, 
    #                             mark_revealed_patches: int = None) -> torch.Tensor:
    #     """
    #     Unpacks a sequence of indices and image toknes into a 2d image tensor and sorts the patches
    #     according to the patch indices

    #     Parameters:
    #         img_seq (torch.Tensor) of size b, t:
    #             The input image sequence, where t is 17 * the number of image patches if patch_size is 4
    #         num_revealed_patches (int):
    #             The number of revealed patches to mark
    #         mark_revealed_patches (int):
    #             The value to mark the revealed patches with
        
    #     Returns:
    #         torch.Tensor: The unpacked image of size b, 16 * sqrt(t/17), 16 * sqrt(t/17) if patch_size is 4
    #     """

    #     # reshape the image sequence into a sequence of patches and
    #     # trim the first element in the 2nd dim (the index)
    #     n_tokens_per_patch = self.config.patch_size**2 + 1 # compute the number of tokens per patch
    #     img_seq = img_seq.view(img_seq.size(0), -1, n_tokens_per_patch)
    #     img_idxs = img_seq[:, :, 0].long() - (65536 + 256) # shift the indices to the left by 65536 + 256
    #     reconstruct_indxs = torch.argsort(img_idxs, dim=1)
    #     rgb_seq = img_seq[:, :, 1:]

    #     # color the firs num_revealed_patches patches with the specified value if mark_revealed_patches is not None
    #     if mark_revealed_patches is not None:
    #         rgb_seq[:, :num_revealed_patches] = mark_revealed_patches

    #     # reorded the patches according to the patch indices
    #     rgb_seq = rgb_seq[torch.arange(img_seq.size(0)).unsqueeze(1).expand(*img_idxs.shape), reconstruct_indxs]

    #     # unpatchify the image sequence
    #     img = unpatchify(rgb_seq)

    #     return img

    def estimate_mfu(self, fwdbwd_per_iter, T, dt, gpu_type='A40'):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.unsharded_param_count # self.get_num_params()
        cfg = self.config
        L, H, Q = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head
        # L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second

        # grab promised flops based on GPU type
        if gpu_type == 'A40':
            flops_promised = 149.7e12 # A40 GPU bfloat16 peak flops is 149.7 TFLOPS
        elif gpu_type == 'A100':
            flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
        elif gpu_type == 'H100':
            flops_promised = 756e12 # H100 GPU bfloat16 peak flops is 756 TFLOPS
        elif gpu_type == 'TPUv4':
            flops_promised = 275e12
        elif gpu_type == 'TPUv5e':
            flops_promised = 197e12

        mfu = flops_achieved / flops_promised
        return mfu

    def forward_with_features(
            self, 
            seq: torch.Tensor, 
            pos: torch.Tensor = None, 
            tgt: torch.Tensor = None, 
            mask: torch.Tensor = None,
            feature_layers: List[int] = None,
        ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass of the model that returns both logits and features from specified layers

        Parameters:
            seq (torch.Tensor) of size b, t: The input sequence
            pos (torch.Tensor) of size b, t: The positional indices of the sequence
            tgt (torch.Tensor) of size b, t_tgt: The target sequence
            mask (torch.Tensor) of size b, t, t: The mask of the sequence
            feature_layers (List[int]): List of layer indices to extract features from
        
        Returns:
            Tuple[torch.Tensor, List[torch.Tensor]]: 
                - The logits of the model. Size b, t if tgt is None, else b, t_tgt
                - List of features from specified layers
        """

        # grab device to perform operations on
        device = seq.device
        # grab dimensions
        b, t = seq.size()
        if tgt is not None:
            b, t_tgt = tgt.size()

        assert t <= self.config.block_size, \
            f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        if tgt is not None:
            assert t_tgt <= t, \
                f"Target seqeunce length {t_tgt} must be shorter than or equal to sequence length {t}"

        # create a tensor of position indices, if not provided
        if pos is None:
            pos = torch.arange(t, device=device).unsqueeze(0).expand(b, -1)

        # forward the GPT model itself
        tok_emb = self.transformer.token_embedding(seq) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.positional_embedding(pos) # position embeddings of shape (t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)

        if self.spmd_mesh is not None:
            import torch_xla.distributed.spmd.xla_sharding as xs
            xs.mark_sharding(x, self.spmd_mesh,  (('dcn', 'data'), None, 'model'))

        # create attention mask
        with torch.no_grad():
            # Generate a mask template which will be used to mask out the attention
            # if len(mask.shape) == 2:
            mask = torch.triu(torch.ones(t, t, 
                    device=x.device, requires_grad=False) * -float('inf'), diagonal=1).view(1, 1, t, t)
                
        # Initialize list to store features
        features = []
        
        # Process through transformer blocks
        for block_idx, block in enumerate(self.transformer.h):
            x = block(x, spmd_mesh=self.spmd_mesh, mask=mask)
            # If this is a layer we want to extract features from, save it
            if feature_layers is not None and block_idx in feature_layers:
                features.append(x)
        
        x = self.transformer.ln_f(x)
        if "last" in feature_layers:
            features.append(x)

        # if tgt is not none, compute the logits for the entire sequence
        if tgt is None:
            logits = self.lm_head(x)
            return logits, features
        
        # if tgt is not none, compute the logits and the loss for the target sequence
        logits = self.lm_head(x[:, -tgt.size(1):])

        if self.spmd_mesh is not None:
            xs.mark_sharding(logits, self.spmd_mesh, (('dcn', 'data'), None, 'model'))

        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1), ignore_index=-1)
        return logits, loss, features

