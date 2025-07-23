import random
from pathlib import Path
from typing import Tuple, Union, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from PIL import Image

from spelke_net.utils.camera import transform_matrix_to_six_dof_axis_angle, quantize_6dof_campose
from spelke_net.utils.flow import decode_flow_code, sample_flow_values_and_positions
from spelke_net.utils.image_processing import patchify as patchify_func
from spelke_net.utils.model_wrapper import ModelFactory
from spelke_net.utils.sequence_construction import (
    add_patch_indexes, get_pos_idxs, shuffle_and_trim_values_and_positions
)


class LRASFlowPredictor:

    def __init__(self, model_name: str, quantizer_name: str, flow_quantizer_name: str = None, device: str = 'cpu', token_dict_path: str = None):
        
        # Load the model and quantizer
        try:
            self.model = ModelFactory().load_model(model_name).to(torch.bfloat16).to(device).eval()
        except Exception as e:
            if Path(model_name).exists():
                self.model = ModelFactory().load_model_from_checkpoint(model_name).to(torch.bfloat16).to(device).eval()
            else: raise e
        try:
            self.quantizer = ModelFactory().load_model(quantizer_name).to(device).to(torch.float32).eval()
        except Exception as e:
            if Path(quantizer_name).exists():
                self.quantizer = ModelFactory().load_model_from_checkpoint(quantizer_name).to(device).to(torch.float32).eval()
            else: raise e
        if flow_quantizer_name is not None:
            try:
                self.flow_quantizer = ModelFactory().load_model(flow_quantizer_name).to(device).to(torch.float32).eval()
            except:
                if Path(flow_quantizer_name).exists():
                    self.flow_quantizer = ModelFactory().load_model_from_checkpoint(flow_quantizer_name).to(device).to(torch.float32).eval()
                else: raise e
        else:
            self.flow_quantizer = None

        self.ctx = torch.amp.autocast(device_type='cuda' if 'cuda' in device else 'cpu', dtype=torch.bfloat16)

        # Set parameters
        self.device = device

        # Set transforms
        self.in_transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.inv_in_transform = torchvision.transforms.Compose([
            torchvision.transforms.Normalize(
                mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], std=[1/0.229, 1/0.224, 1/0.225]),
            torchvision.transforms.Lambda(lambda x: torch.clamp(x, 0, 1)), 
            torchvision.transforms.ToPILImage()
        ])

        self.flow_transform = torchvision.transforms.Compose([
            torchvision.transforms.Lambda(lambda x: torch.tensor(x).permute(2,0,1) if len(x.shape) == 3 else torch.tensor(x).permute(0,3,1,2)),
            torchvision.transforms.Normalize(mean=[0.0, 0.0], std=[20.0, 20.0]),
        ])
        self.inv_flow_transform = torchvision.transforms.Compose([
            torchvision.transforms.Normalize(mean=[0.0, 0.0], std=[1/20.0, 1/20.0]),
            torchvision.transforms.Lambda(lambda x: x.permute(1,2,0).cpu().numpy() if len(x.shape) == 3 else x.permute(0,2,3,1).numpy()),
        ])

        self.resize_crop_transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(256),
            torchvision.transforms.CenterCrop(256),
        ])

        self.token_to_flow_dict = ModelFactory().load_ckpt("flow_token_to_flow_vector_mapping.pt")


    @torch.no_grad()
    def flow_factual_prediction(
            self, 
            frame0: Union[Image.Image, np.ndarray, torch.Tensor], 
            frame1: Union[Image.Image, np.ndarray, torch.Tensor], 
            flow: torch.FloatTensor = None,
            unmask_indices: List[int] = None,
            mode: str = 'sequential',
            temperature: float = 1.0,
            top_p: float = 0.9,
            top_k: int = 1000,
            seed : int = 0,
            decoding_order: List[int] = None,
            unmask_indices_img1: List[int] = None,
        ) -> Tuple[Image.Image, Image.Image, Image.Image]:
        
        """
        Perform a forward pass through the model using all of frame0 and part of frame1.

        Parameters:
            frame0: Image.Image, the first frame
            frame1: Image.Image, the second frame
            campose: torch.FloatTensor, the camera pose represented as a 4x4 transformation matrix
            mask_ratio: float, the ratio of patches to mask out
            mask_indices: List[int], the indices of the patches to mask out
            mode: str, the mode to use for the forward pass (sequential or parallel)
            seed: int, the random seed to use
            mask_out: bool, whether to black out the unmasked patches in the predicted frame
        
        Returns:
            frame0: Image.Image, the first frame
            frame1_pred: Image.Image, the predicted second frame
            frame1: Image.Image, the actual second frame
            rgb_logits: torch.FloatTensor, the logits for the predicted image codes
        """

        assert mode in ['sequential', 'parallel'], "Mode must be one of ['sequential', 'parallel']"

        self._set_seed(seed)
        
        # Transform the input frames and quantize them if they are provided as PIL images
        if isinstance(frame0, Image.Image) or isinstance(frame0, np.ndarray):
            frame0_codes = self.quantizer.quantize(self.in_transform(frame0).unsqueeze(0).to(self.device))
        else:
            frame0_codes = frame0
        if isinstance(frame1, Image.Image) or isinstance(frame1, np.ndarray):
            frame1_codes = self.quantizer.quantize(self.in_transform(frame1).unsqueeze(0).to(self.device))
        else:
            frame1_codes = frame1

        if unmask_indices is None:
            unmask_idxs = [x for x in range(1024)]
            random.shuffle(unmask_idxs)
            unmask_idxs = unmask_idxs[:100]
        else:
            unmask_idxs = unmask_indices

        if mode == 'sequential':
            frame1_pred_codes, rgb_logits, _ = self.frame0_flow_frame1_sequential_forward(
                frame0_codes.clone(),
                flow,
                frame1_codes.clone(),
                unmask_idxs=unmask_idxs,
                unmask_idxs_img1=unmask_indices_img1,
                temperature=temperature, top_p=top_p, top_k=top_k, decoding_order=decoding_order
            )
        else:
            raise NotImplementedError("Parallel mode not implemented for flow prediction")
            # frame1_pred_codes, rgb_logits = self.two_frame_patchwise_parallel_forward(
            #     frame0_codes.clone(), frame1_codes.clone(), unmask_idxs,
            #     temperature=temperature, top_p=top_p, top_k=top_k
            # )

        # Compute grid entropy and varentropy
        if unmask_indices_img1 is None:
            rgb_grid_entropy = self._compute_rgb_grid_entropy(rgb_logits, unmask_idxs).detach().cpu().float()
            rgb_grid_varentropy = self._compute_rgb_grid_varentropy(rgb_logits, unmask_idxs).detach().cpu().float()
            ce_error = self._compute_rgb_grid_ce_error(rgb_logits, frame1_codes).detach().cpu().float()
        else:
            rgb_grid_entropy = None
            rgb_grid_varentropy = None
            ce_error = None

        # Decode the predicted frame
        frame0 = self.quantizer.decode(frame0_codes.to(self.device))
        frame1_pred = self.quantizer.decode(frame1_pred_codes)
        frame1 = self.quantizer.decode(frame1_codes.to(self.device))

        # Un-normalize and convert to PIL
        frame0_pil = self.inv_in_transform(frame0[0])
        frame1_pred_pil = self.inv_in_transform(frame1_pred[0])
        frame1_pil = self.inv_in_transform(frame1[0])

        # Compute CE and MSE errors

        l1_error = self._compute_rgb_grid_l1_error(frame1_pred, frame1).detach().cpu().float()
        mse_error = self._compute_rgb_grid_mse_error(frame1_pred, frame1).detach().cpu().float()

    
        return {
            "frame0_rgb": frame0,
            "frame0_pil": frame0_pil,
            "frame0_codes": frame0_codes[0],
            "frame1_pred_rgb": frame1_pred,
            "frame1_pred_pil": frame1_pred_pil,
            "frame1_pred_codes": frame1_pred_codes[0],
            "frame1_rgb": frame1,
            "frame1_pil": frame1_pil,
            "frame1_codes": frame1_codes[0],
            "rgb_logits": rgb_logits,
            "rgb_grid_entropy": rgb_grid_entropy,
            "rgb_grid_varentropy": rgb_grid_varentropy,
            "ce_grid_error": ce_error,
            "l1_grid_error": l1_error,
            "mse_grid_error": mse_error,
        }
    
    @torch.no_grad()
    def quantized_flow_prediction(
            self, 
            frame0: Union[Image.Image, np.ndarray, torch.Tensor], 
            flow_cond: Union[np.ndarray, torch.Tensor] = None, 
            motion_indices: List[int] = None,
            campose: torch.FloatTensor = None,
            mask_ratio: float = 1.0,
            unmask_indices: List[int] = None,
            mode: str = 'sequential',
            temperature: float = 1.0,
            top_p: float = 0.9,
            top_k: int = 1000,
            seed : int = 0,
            num_seq_patches: int = 32,
            mask_out: bool = True,
            cfg_ratio: float = None,
            segment_map: torch.Tensor = None,
            allowed_tokens: List[int] = None,
            rmi=None,
        ) -> Tuple[Image.Image, Image.Image, Image.Image]:
        
        """
        Perform a forward pass through the model using all of frame0 and part of frame1.

        Parameters:
            frame0: Image.Image or np.ndarray or torch.Tensor, the first frame
            flow_cond: first value is horizontal index, second value is vertical index
            flow: torch.Tensor or np.ndarray, the flow represented as a 2-channel image
            campose: torch.FloatTensor, the camera pose represented as a 4x4 transformation matrix
            mask_ratio: float, the ratio of patches to mask out
            mask_indices: List[int], the indices of the patches to mask out
            mode: str, the mode to use for the forward pass (sequential or parallel)
            seed: int, the random seed to use
            mask_out: bool, whether to black out the unmasked patches in the predicted frame
        
        Returns:
            frame0: Image.Image, the first frame
            frame1_pred: Image.Image, the predicted second frame
            frame1: Image.Image, the actual second frame
            rgb_logits: torch.FloatTensor, the logits for the predicted image codes
        """

        assert mode in ['sequential', 'patchwise_parallel', 'parallel', 'seq2par'], "Mode must be one of ['sequential', 'patchwise_parallel', 'parallel']"
        # assert mask_ratio is not None or unmask_indices is not None, "Either mask_ratio or mask_indices must be provided"
        # assert mask_ratio is None or unmask_indices is None, "Only one of mask_ratio or mask_indices can be provided"
        
        # Transform the input frames and quantize them if they are provided as PIL images
        if isinstance(frame0, Image.Image) or isinstance(frame0, np.ndarray):
            frame0_codes = self.quantizer.quantize(self.in_transform(frame0).unsqueeze(0).to(self.device))
        else:
            frame0_codes = frame0


        flow = None
        if flow_cond is not None:
            # make a flow map depending on the flow conditioning
            flow = np.zeros((256, 256, 2), dtype=np.float32)
            # make a list of unmask indices
            unmask_indices = []

            for x1, y1, x2, y2 in flow_cond:
                dx = (x2-x1)
                dy = (y2-y1)
                # round x and y indices down to nearest multiple of 8
                x = int(x1 // 8)
                y = int(y1 // 8)
                # set the part of the flow map to the dx and dy values
                flow[8*y:8*(y+1), 8*x:8*(x+1), 0] = dx
                flow[8*y:8*(y+1), 8*x:8*(x+1), 1] = dy
                # convert x and y to patch index on a 32x32 grid
                patch_idx = x + y * 32
                unmask_indices.append(patch_idx)

        if segment_map is not None:
            flow = flow * segment_map[:, :, None]

        # print("unmask_indices", unmask_indices)
                
        # If flow is None, make it a tensor of zeros like frame0_codes
        if flow is None:
            flow_codes = torch.zeros_like(frame0_codes)
        # If flow is a numpy array, convert it to a tensor with the flow quantizer
        if isinstance(flow, np.ndarray):
            flow_codes = self.flow_quantizer.quantize(self.flow_transform(torch.tensor(flow).unsqueeze(0).to(self.device)))
        # If flow is a tensor, assume it is already quantized codes
        if isinstance(flow, torch.Tensor):
            flow_codes = flow

        flow_codes = flow_codes + self.model.config.flow_range[0]
        # Transform the 4x4 campose matrix to a 6-DOF quantized vector
        if campose is not None:
            six_dof_campose = transform_matrix_to_six_dof_axis_angle(campose.cpu().numpy(), scale=True)
            campose_codes = torch.tensor(quantize_6dof_campose(six_dof_campose), dtype=torch.long)

        # Generate random list of unmasked indexes
        if unmask_indices is None:
            unmask_indices = random.sample(range(self.model.config.flow_patch_idx_range[1] - self.model.config.flow_patch_idx_range[0]), 
                int((self.model.config.flow_patch_idx_range[1] - self.model.config.flow_patch_idx_range[0]) * (1.0 - mask_ratio)))

        self._set_seed(seed)

        if unmask_indices is None:
            unmask_indices = []

        if mode == 'sequential':
            flow_pred_codes, flow_logits, decoding_order = self.two_frame_sequential_forward(
                frame0_codes.clone(), flow_codes.clone(), unmask_indices,
                frame0_patch_offset=None, frame0_seq_offset=None,
                frame1_patch_offset=self.model.config.flow_patch_idx_range[0],
                frame1_seq_offset=self.model.config.flow_pos_range[0],
                campose_codes=campose_codes if campose is not None else None,
                # flow_codes=flow if flow is not None else None,
                temperature=temperature, top_p=top_p, top_k=top_k,
                cfg_ratio=cfg_ratio
            )
        elif mode == 'seq2par':
            flow_pred_codes, flow_logits, decoding_order = self.two_frame_seq2par_forward(
                frame0_codes.clone(), flow_codes.clone(), unmask_indices,
                frame0_patch_offset=None, frame0_seq_offset=None,
                frame1_patch_offset=self.model.config.flow_patch_idx_range[0],
                frame1_seq_offset=self.model.config.flow_pos_range[0],
                motion_indices=motion_indices,
                rmi=rmi,
                campose_codes=campose_codes if campose is not None else None,
                num_seq_patches=num_seq_patches, temperature=temperature, top_p=top_p, top_k=top_k, allowed_tokens=allowed_tokens,
            )

        # Compute grid entropy and varentropy
        flow_logits = flow_logits[-1024:]
        flow_grid_entropy = self._compute_flow_grid_entropy(flow_logits.cpu(), unmask_indices).detach().cpu().float()
        prob_no_motion = self._compute_flow_grid_cumulative_probability(
            flow_logits.cpu(),
            unmask_indices,
            [self.model.config.flow_range[0] + 11646, self.model.config.flow_range[0] + 11582],
            self.model.config.flow_range
        ).detach().cpu().float()
        # flow_rgb_grid_varentropy = self._compute_rgb_grid_varentropy(flow_logits.cpu(), unmask_indices).detach().cpu().float()

        # Decode the predicted frame
        flow_pred_codes = flow_pred_codes - self.model.config.flow_range[0]
        flow_codes = flow_codes - self.model.config.flow_range[0]
        frame0 = self.quantizer.decode(frame0_codes.to(self.device))
        flow_pred = self.flow_quantizer.decode(flow_pred_codes.to(self.device))
        flow = self.flow_quantizer.decode(flow_codes.to(self.device))

        # Un-normalize and convert to PIL
        frame0_pil = self.inv_in_transform(frame0[0])
        flow_pred_np = self.inv_flow_transform(flow_pred[0])
        flow_np = self.inv_flow_transform(flow[0])

        # Compute CE and MSE errors
        # ce_error = self._compute_rgb_grid_ce_error(rgb_logits.cpu(), frame1_codes.cpu()).detach().cpu().float()
        # l1_error = self._compute_rgb_grid_l1_error(frame1_pred.cpu(), frame1.cpu()).detach().cpu().float()
        # mse_error = self._compute_rgb_grid_mse_error(frame1_pred.cpu(), frame1.cpu()).detach().cpu().float()

        # Black out the unmasked patches if mask_out is True
        # if mask_out:
        #     flow_pred_np = mask_out_image(flow_pred_np, unmask_indices, color=200, patch_size=self.model.config.patch_size*4)

        return {
            "frame0_rgb": frame0,
            "frame0_pil": frame0_pil,
            "frame0_codes": frame0_codes[0],
            "flow_pred_rgb": flow_pred,
            "flow_pred_np": flow_pred_np,
            "flow_pred_codes": flow_pred_codes[0],
            "flow_rgb": flow,
            "flow_np": flow_np,
            "flow_logits": flow_logits.cpu(),
            # "frame1_codes": frame1_codes[0],
            # "rgb_logits": rgb_logits,
            "flow_grid_entropy": flow_grid_entropy.cpu(),
            "prob_no_motion": prob_no_motion.cpu(),
            # "flow_grid_varentropy": flow_rgb_grid_varentropy,
            # "ce_grid_error": ce_error,
            # "l1_grid_error": l1_error,
            # "mse_grid_error": mse_error,
            "decoding_order": decoding_order if mode == 'sequential' else []
        }

    def quantized_flow_prediction_biased_cond(
            self,
            frame0: Union[Image.Image, np.ndarray, torch.Tensor],
            flow_cond: Union[np.ndarray, torch.Tensor] = None,
            motion_indices: List[int] = None,
            campose: torch.FloatTensor = None,
            mask_ratio: float = 1.0,
            unmask_indices: List[int] = None,
            mode: str = 'sequential',
            temperature: float = 1.0,
            top_p: float = 0.9,
            top_k: int = 1000,
            seed: int = 0,
            num_seq_patches: int = 32,
            mask_out: bool = True,
            cfg_ratio: float = None,
            segment_map: torch.Tensor = None,
            probe_point: Optional[torch.Tensor] = None,
            allowed_tokens: List[int] = None,
            min_mag = 20,
            max_mag = 30,
    ) -> Tuple[Image.Image, Image.Image, Image.Image]:

        """
        Perform a forward pass through the model using all of frame0 and part of frame1.

        Parameters:
            frame0: Image.Image or np.ndarray or torch.Tensor, the first frame
            flow_cond: first value is horizontal index, second value is vertical index
            flow: torch.Tensor or np.ndarray, the flow represented as a 2-channel image
            campose: torch.FloatTensor, the camera pose represented as a 4x4 transformation matrix
            mask_ratio: float, the ratio of patches to mask out
            mask_indices: List[int], the indices of the patches to mask out
            mode: str, the mode to use for the forward pass (sequential or parallel)
            seed: int, the random seed to use
            mask_out: bool, whether to black out the unmasked patches in the predicted frame

        Returns:
            frame0: Image.Image, the first frame
            frame1_pred: Image.Image, the predicted second frame
            frame1: Image.Image, the actual second frame
            rgb_logits: torch.FloatTensor, the logits for the predicted image codes
        """

        assert mode in ['sequential', 'patchwise_parallel', 'parallel',
                        'seq2par'], "Mode must be one of ['sequential', 'patchwise_parallel', 'parallel']"
        # assert mask_ratio is not None or unmask_indices is not None, "Either mask_ratio or mask_indices must be provided"
        # assert mask_ratio is None or unmask_indices is None, "Only one of mask_ratio or mask_indices can be provided"

        # Transform the input frames and quantize them if they are provided as PIL images
        if isinstance(frame0, Image.Image) or isinstance(frame0, np.ndarray):
            frame0_codes = self.quantizer.quantize(self.in_transform(frame0).unsqueeze(0).to(self.device))
        else:
            frame0_codes = frame0


        flow = None
        if flow_cond is not None:
            # make a flow map depending on the flow conditioning
            flow = np.zeros((256, 256, 2), dtype=np.float32)
            # make a list of unmask indices
            unmask_indices = []

            for x1, y1, x2, y2 in flow_cond:
                dx = (x2 - x1)
                dy = (y2 - y1)
                # round x and y indices down to nearest multiple of 8
                x = int(x1 // 8)
                y = int(y1 // 8)
                # set the part of the flow map to the dx and dy values
                flow[8 * y:8 * (y + 1), 8 * x:8 * (x + 1), 0] = dx
                flow[8 * y:8 * (y + 1), 8 * x:8 * (x + 1), 1] = dy
                # convert x and y to patch index on a 32x32 grid
                patch_idx = x + y * 32
                unmask_indices.append(patch_idx)

        x_probe, y_probe = probe_point // 8
        # get the patch index for the probe point
        probe_patch_idx = x_probe + y_probe * 32
        # create a random shuffling of patch indices where the probe point comes right after the number of unmask indices length
        # if unmask_indices is not None:
        #     rmi = np.arange(1024)
        #     np.random.shuffle(rmi)
        #
        #     # Remove probe_patch_idx if it exists (to avoid duplication)
        #     rmi = rmi[rmi != probe_patch_idx]
        #
        #     # Insert probe_patch_idx right after the unmask_indices
        #     insertion_index = len(unmask_indices)
        #     rmi = np.insert(rmi, insertion_index, probe_patch_idx)
        #
        # rmi = torch.tensor(rmi, dtype=torch.long)
        rmi = None
        # print(rmi[4])
        start, end = self.model.config.flow_range
        # Start with your given token ids and flow map
        token_ids = np.array(list(self.token_to_flow_dict.keys()))
        flow_mapp = np.stack([self.token_to_flow_dict[x] for x in token_ids], axis=0)  # shape: (N, 4, 4, 2)

        # Step 1: Compute flow magnitude (sqrt(dx^2 + dy^2)) at each location
        flow_magnitude = np.linalg.norm(flow_mapp, axis=-1)  # shape: (N, 4, 4)

        # Step 2: Compute mean magnitude for each token
        mean_magnitude = flow_magnitude.max(axis=(1, 2))  # shape: (N,)

        # Step 3: Filter token_ids with mean magnitude > 10
        selected_token_ids = token_ids[(mean_magnitude > min_mag) & (mean_magnitude < max_mag)]  # + start

        sampling_blacklist = np.array([x for x in np.arange(0, end - start) if x not in selected_token_ids]) + start

        if segment_map is not None:
            flow = flow * segment_map[:, :, None]

        # print("unmask_indices", unmask_indices)

        # If flow is None, make it a tensor of zeros like frame0_codes
        if flow is None:
            flow_codes = torch.zeros_like(frame0_codes)
        # If flow is a numpy array, convert it to a tensor with the flow quantizer
        if isinstance(flow, np.ndarray):
            flow_codes = self.flow_quantizer.quantize(
                self.flow_transform(torch.tensor(flow).unsqueeze(0).to(self.device)))
        # If flow is a tensor, assume it is already quantized codes
        if isinstance(flow, torch.Tensor):
            flow_codes = flow

        flow_codes = flow_codes + self.model.config.flow_range[0]
        # Transform the 4x4 campose matrix to a 6-DOF quantized vector
        if campose is not None:
            six_dof_campose = transform_matrix_to_six_dof_axis_angle(campose.cpu().numpy(), scale=True)
            campose_codes = torch.tensor(quantize_6dof_campose(six_dof_campose), dtype=torch.long)

        # Generate random list of unmasked indexes
        if unmask_indices is None:
            unmask_indices = random.sample(
                range(self.model.config.flow_patch_idx_range[1] - self.model.config.flow_patch_idx_range[0]),
                int((self.model.config.flow_patch_idx_range[1] - self.model.config.flow_patch_idx_range[0]) * (
                            1.0 - mask_ratio)))

        self._set_seed(seed)

        if unmask_indices is None:
            unmask_indices = []

        if mode == 'sequential':
            flow_pred_codes, flow_logits, decoding_order = self.two_frame_sequential_forward(
                frame0_codes.clone(), flow_codes.clone(), unmask_indices,
                frame0_patch_offset=None, frame0_seq_offset=None,
                frame1_patch_offset=self.model.config.flow_patch_idx_range[0],
                frame1_seq_offset=self.model.config.flow_pos_range[0],
                campose_codes=campose_codes if campose is not None else None,
                # flow_codes=flow if flow is not None else None,
                temperature=temperature, top_p=top_p, top_k=top_k,
                cfg_ratio=cfg_ratio
            )
        elif mode == 'seq2par':
            flow_pred_codes, flow_logits, decoding_order = self.two_frame_seq2par_forward(
                frame0_codes.clone(), flow_codes.clone(), unmask_indices,
                frame0_patch_offset=None, frame0_seq_offset=None,
                frame1_patch_offset=self.model.config.flow_patch_idx_range[0],
                frame1_seq_offset=self.model.config.flow_pos_range[0],
                motion_indices=motion_indices,
                rmi=rmi,
                campose_codes=campose_codes if campose is not None else None,
                num_seq_patches=num_seq_patches, allowed_tokens=allowed_tokens, temperature=temperature, top_p=top_p, top_k=top_k,
                sampling_blacklist=sampling_blacklist, probe_idx=probe_patch_idx
            )

        # Compute grid entropy and varentropy
        flow_logits = flow_logits[-1024:]
        flow_grid_entropy = self._compute_flow_grid_entropy(flow_logits.cpu(), unmask_indices).detach().cpu().float()
        prob_no_motion = self._compute_flow_grid_cumulative_probability(
            flow_logits.cpu(),
            unmask_indices,
            [self.model.config.flow_range[0] + 11646, self.model.config.flow_range[0] + 11582],
            self.model.config.flow_range
        ).detach().cpu().float()
        # flow_rgb_grid_varentropy = self._compute_rgb_grid_varentropy(flow_logits.cpu(), unmask_indices).detach().cpu().float()

        # Decode the predicted frame
        flow_pred_codes = flow_pred_codes - self.model.config.flow_range[0]
        flow_codes = flow_codes - self.model.config.flow_range[0]
        frame0 = self.quantizer.decode(frame0_codes.to(self.device))
        flow_pred = self.flow_quantizer.decode(flow_pred_codes.to(self.device))
        flow = self.flow_quantizer.decode(flow_codes.to(self.device))

        # Un-normalize and convert to PIL
        frame0_pil = self.inv_in_transform(frame0[0])
        flow_pred_np = self.inv_flow_transform(flow_pred[0])
        flow_np = self.inv_flow_transform(flow[0])

        # Compute CE and MSE errors
        # ce_error = self._compute_rgb_grid_ce_error(rgb_logits.cpu(), frame1_codes.cpu()).detach().cpu().float()
        # l1_error = self._compute_rgb_grid_l1_error(frame1_pred.cpu(), frame1.cpu()).detach().cpu().float()
        # mse_error = self._compute_rgb_grid_mse_error(frame1_pred.cpu(), frame1.cpu()).detach().cpu().float()

        # Black out the unmasked patches if mask_out is True
        # if mask_out:
        #     flow_pred_np = mask_out_image(flow_pred_np, unmask_indices, color=200, patch_size=self.model.config.patch_size*4)


        return {
            "frame0_rgb": frame0,
            "frame0_pil": frame0_pil,
            "frame0_codes": frame0_codes[0],
            "flow_pred_rgb": flow_pred,
            "flow_pred_np": flow_pred_np,
            "flow_pred_codes": flow_pred_codes[0],
            "flow_rgb": flow,
            "flow_np": flow_np,
            "flow_logits": flow_logits.cpu(),
            # "frame1_codes": frame1_codes[0],
            # "rgb_logits": rgb_logits,
            "flow_grid_entropy": flow_grid_entropy.cpu(),
            "prob_no_motion": prob_no_motion.cpu(),
            # "flow_grid_varentropy": flow_rgb_grid_varentropy,
            # "ce_grid_error": ce_error,
            # "l1_grid_error": l1_error,
            # "mse_grid_error": mse_error,
            "decoding_order": decoding_order if mode == 'sequential' else []
        }

    @torch.no_grad()
    def two_frame_patchwise_parallel_forward(
        self, frame0_codes: torch.LongTensor, 
        frame1_codes: torch.LongTensor, 
        unmask_idxs: List[int],
        campose_codes: torch.LongTensor = None,
        flow_codes: torch.LongTensor = None,
        flow_unmask_idxs: List[int] = None,
        temperature: float = 1.0,
        top_p: float = 0.9,
        top_k: int = 1000,
    ) -> Tuple[torch.LongTensor, torch.FloatTensor]:
        """
        Perform a forward pass through the model using the "patchwise parallel" method.

        Parameters:
            frame0_codes: torch.LongTensor, shape (B, H, W), the quantized image codes for frame 0
            frame1_codes: torch.LongTensor, shape (B, H, W)), the quantized image codes for frame 1
            unmask_idxs: List[int], the indexes of the patches to reveal
            temperature: float, the temperature value for sampling
            top_p: float, the top_p value for sampling
            top_k: int, the top_k value for sampling

        Returns:
            frame1_pred: torch.LongTensor, shape (B, H, W), the predicted image codes for frame 1
            logits: torch.FloatTensor, shape (B, H, W, C), the logits for the predicted image codes
        """

        # Pack the images into sequences
        im0_seq, im0_pos = self._pack_image_codes_into_sequence(
            frame0_codes.cpu(), mask=0.0,
            patch_offset=self.model.config.rgb_patch_0_idx_range[0], 
            seq_offset=self.model.config.frame_0_rgb_pos_range[0])
        im1_seq, im1_pos = self._pack_image_codes_into_sequence(
            frame1_codes.cpu(), mask=0.0,
            patch_offset=self.model.config.rgb_patch_1_idx_range[0], 
            seq_offset=self.model.config.frame_1_rgb_pos_range[0])
        
        # Bring the revealed patches to the front
        im1_seq, im1_pos = self._bring_patches_to_front(im1_seq, im1_pos, unmask_idxs)

        # Create the sequence
        seq = im0_seq.view(-1)
        pos = im0_pos.view(-1)
        seq_delim_idx = im0_seq.numel()
        
        # If flow is provided, pack it into a sequence
        if flow_codes is not None:
            flow_seq, flow_pos = self._pack_flow_codes_into_sequence(
                flow_codes.cpu(), mask=0.0,
                patch_offset=self.model.config.flow_patch_idx_range[0],
                seq_offset=self.model.config.flow_pos_range[0])
            flow_seq, flow_pos = self._bring_flow_patches_to_front(flow_seq, flow_pos, flow_unmask_idxs, discard=True)
            seq = torch.cat([seq, flow_seq.view(-1)])
            pos = torch.cat([pos, flow_pos.view(-1)])
            seq_delim_idx += flow_seq.numel()

        # Concatenate the two image sequences into a single sequence
        seq = torch.cat([seq, im1_seq.view(-1)])
        pos = torch.cat([pos, im1_pos.view(-1)])
        seq_delim_idx += len(unmask_idxs) * im1_seq.shape[2]

        # Mask out part of frame 1
        im0_delim_idx = im0_seq.numel()

        cond_seq = seq[:seq_delim_idx].clone()
        cond_pos = pos[:seq_delim_idx].clone()

        pred_seq = seq[seq_delim_idx:].clone().view(-1, im1_seq.shape[2])
        pred_pos = pos[seq_delim_idx:].clone().view(-1, im1_pos.shape[2])

        step_seq = torch.cat([cond_seq, pred_seq[:, 0].view(-1)])
        step_pos = cond_pos.clone()

        all_logits = []

        # Iterate over the patches once for each patch in the second frame
        for it in range(im1_seq.shape[2] - 1):

            step_pos = torch.cat([step_pos, pred_pos[:, it].view(-1)])

            step_mask = torch.zeros(step_pos.shape[0], step_pos.shape[0]).to(step_pos.device)


            # attention mask for frame 0 + unmasked part of frame 1
            step_mask[:, :seq_delim_idx] = 1
            step_mask[:seq_delim_idx, :seq_delim_idx].tril_()

            # attention mask for rest of frame 1
            step_mask[seq_delim_idx:, seq_delim_idx:] = 1


            # # attention mask for frame 0
            # step_mask[:, :im0_delim_idx] = 1
            # step_mask[:im0_delim_idx, :im0_delim_idx].tril_()

            # # attention mask for unmask idx
            # if len(unmask_idxs) > 0:
            #     step_mask[im0_delim_idx:seq_delim_idx, im0_delim_idx:seq_delim_idx] = 1
            #     step_mask[im0_delim_idx:seq_delim_idx, im0_delim_idx:seq_delim_idx].tril_()
            #     step_mask[seq_delim_idx:, im0_delim_idx:seq_delim_idx] = 1

            # # attention mask for predicted
            # for k in range(it + 1):
            #     pred_len = pred_pos.shape[0] * (it + 1 - k)
            #     row_start_idx = seq_delim_idx + pred_pos.shape[0] * k
            #     col_start_idx = seq_delim_idx
            #     row_end_idx = row_start_idx + pred_len
            #     col_end_idx = col_start_idx + pred_len
            #     step_mask[row_start_idx:row_end_idx, col_start_idx:col_end_idx].fill_diagonal_(1)


            step_mask = step_mask.unsqueeze(0) # Add a batch dimension to the mask
            step_mask = 1 - step_mask # the forward function assumes 0 to participate in attention, 1 otherwise

            step_tgt = pred_seq[:, it+1].view(-1)

            # Perform the prediction
            with self.ctx:
                logits, loss = self.model(
                    step_seq.unsqueeze(0).to(self.device).long(), 
                    pos=step_pos.unsqueeze(0).to(self.device).long(),
                    mask=step_mask.unsqueeze(0).to(self.device).bool(),
                    tgt=step_tgt.unsqueeze(0).to(self.device).long(),
                )
            # sampled_tokens = self.model.sample_logits(logits, temp=temperature, top_k=top_k, top_p=top_p)[0]
            sampled_tokens = self.model.sample_logits(logits, temp=0.0)[0]
            step_seq = torch.cat([step_seq, sampled_tokens.cpu()])

            all_logits.append(logits[0].clone())

        all_logits = torch.stack(all_logits, dim=1)
        sort_order = im1_pos[:, len(unmask_idxs):, 0].argsort()
        rgb_logits = all_logits[sort_order][0]

        # insert a dummy patch at any missing patch indexes:
        missing_patch_indexes = [i for i in range(im1_seq.shape[1]) if i not in 
            (im1_seq[:, len(unmask_idxs):, 0] - self.model.config.rgb_patch_1_idx_range[0]).tolist()[0]]
        for i in missing_patch_indexes:
            rgb_logits = torch.cat([rgb_logits[:i], torch.zeros_like(rgb_logits[[0]]), rgb_logits[i:]], dim=0)

        pred_seq = step_seq[cond_seq.numel():].reshape(im1_seq.shape[2], -1).permute(1, 0)
        image1_in_cond_seq = cond_seq[-len(unmask_idxs)*im1_seq.shape[2]:].reshape(-1) if len(unmask_idxs) > 0 else None
        if image1_in_cond_seq is not None:
            frame1_seq = torch.cat([image1_in_cond_seq, pred_seq.reshape(-1)])
        else:
            frame1_seq = pred_seq.reshape(-1)

        # frame1_pred = step_seq[im0_seq.numel():].reshape(im1_seq.shape[2], im1_seq.shape[1]).permute(1, 0)
        frame1_pred =  self.model.unpack_and_sort_img_seq(frame1_seq.reshape(1, -1))

        return frame1_pred, rgb_logits

    @torch.no_grad()
    def two_frame_seq2par_forward(
        self,
        frame0_codes: torch.LongTensor,
        frame1_codes: torch.LongTensor,
        unmask_idxs: List[int],
        flow_unmask_indices: List[int] = None,
        motion_indices: List[int] = None,
        frame0_patch_offset: int = None,
        frame0_seq_offset: int = None,
        frame1_patch_offset: int = None,
        frame1_seq_offset: int = None,
        campose_codes: torch.LongTensor = None,
        flow_codes: torch.LongTensor = None,
        top_p: Union[float, List[float]] = 0.9,
        top_k: Union[int, List[int]] = 1000,
        temperature: Union[float, List[float]] = 1.0,
        cfg_ratio: float = None,
        num_seq_patches: int = 32,
        allowed_tokens: List[int] = None,
        rmi=None,
        frame0_shuffle_order=None,
        sampling_blacklist: List[int] = None,
        probe_idx: int = None,
        first_logits_only=False
    ) -> Tuple[torch.LongTensor, torch.FloatTensor]:
        """
        Perform a forward pass through the model using the "sequential" method.

        Parameters:
            frame0_codes: torch.LongTensor, shape (B, H, W), the quantized image codes for frame 0
            frame1_codes: torch.LongTensor, shape (B, H, W)), the quantized image codes for frame 1
            campose_codes: torch.LongTensor, shape (B, 8), the quantized camera pose codes
            unmask_idxs: List[int], the indexes of the patches to reveal
            top_p: Union[float, List[float]], the top_p value for sampling (optionally per token)
            top_k: Union[int, List[int]], the top_k value for sampling (optionally per token)
            temperature: Union[float, List[float]], the temperature value for sampling (optionally per token)
        
        Returns:
            frame1_pred: torch.LongTensor, shape (B, H, W), the predicted image codes for frame 1
            logits: torch.FloatTensor, shape (B, H, W, C), the logits for the predicted image codes
            decoding_order: torch.LongTensor, shape (B, H), the order in which the patches were decoded
        """

        # Grab default RGB pos and patch idx ranges
        if frame0_patch_offset is None:
            frame0_patch_offset = self.model.config.rgb_patch_0_idx_range[0]
        if frame0_seq_offset is None:
            frame0_seq_offset = self.model.config.frame_0_rgb_pos_range[0]
        if frame1_patch_offset is None:
            frame1_patch_offset = self.model.config.rgb_patch_1_idx_range[0]
        if frame1_seq_offset is None:
            frame1_seq_offset = self.model.config.frame_1_rgb_pos_range[0]
        # breakpoint()
        # Pack the images into sequences
        im0_seq, im0_pos = self._pack_image_codes_into_sequence(
            frame0_codes.cpu(), mask=0.0, shuffle=True,
            patch_offset=frame0_patch_offset, seq_offset=frame0_seq_offset,
            shuffle_order=frame0_shuffle_order)
        im1_seq, im1_pos = self._pack_image_codes_into_sequence(
            frame1_codes.cpu(), mask=0.0, shuffle=True,
            # shuffle_order=rmi,
            patch_offset=frame1_patch_offset, seq_offset=frame1_seq_offset)

        # Pack the camera pose into a sequence if provided
        if campose_codes is not None:
            campose_seq, campose_pos = self._pack_camera_pose_codes_into_sequence(
                campose_codes.cpu(),
                campose_offse=self.model.config.campose_range[0],
                patch_idx_offset=self.model.config.campose_patch_idx_range[0],
                seq_offset=self.model.config.campose_pos_range[0])
        
        # Bring the revealed patches to the front
        if unmask_idxs is not None and len(unmask_idxs) > 0:

            im1_seq, im1_pos = self._bring_patches_to_front(
                im1_seq, im1_pos, unmask_idxs, patch_idx_offset=frame1_patch_offset, additional_idx=probe_idx)

        if flow_codes is not None:
            flows = self.flow_quantizer.decode(flow_codes.to(self.device))
            flow_norm = torch.norm(flows[0], dim=0, keepdim=False, p=2)
            # apply 2d 8x8 average pooling to the flow with stride 8
            pooled_flow = F.avg_pool2d(flow_norm.unsqueeze(0).unsqueeze(0), kernel_size=8, stride=8)[0,0]
            values, indices = torch.sort(pooled_flow.view(-1), descending=True)
            # indices = indices[:int((1.0-0.75)*indices.numel())]

            # breakpoint()
            flow_seq, flow_pos = self._pack_image_codes_into_sequence(
                flow_codes.cpu(), mask=0.0, shuffle=True,
                # shuffle_order=indices.cpu(),
                patch_offset=self.model.config.flow_patch_idx_range[0], 
                seq_offset=self.model.config.flow_pos_range[0])

            # Bring the revealed patches to the front
            if flow_unmask_indices is not None and len(flow_unmask_indices) > 0:
                flow_seq, flow_pos = self._bring_flow_patches_to_front(
                    flow_seq, flow_pos, flow_unmask_indices, discard=True)
            
            print(f"FLOW SEQ SHAPE {flow_seq.shape}")
            
            if campose_codes is not None:
                seq = torch.cat([im0_seq.view(-1), flow_seq.view(-1), campose_seq.view(-1), im1_seq.view(-1)])
                pos = torch.cat([im0_pos.view(-1), flow_pos.view(-1), campose_pos.view(-1), im1_pos.view(-1)])
            else:
                seq = torch.cat([im0_seq.view(-1), flow_seq.view(-1), im1_seq.view(-1)])
                pos = torch.cat([im0_pos.view(-1), flow_pos.view(-1), im1_pos.view(-1)])
        else:
            if campose_codes is not None:
                seq = torch.cat([im0_seq.view(-1), campose_seq.view(-1), im1_seq.view(-1)])
                pos = torch.cat([im0_pos.view(-1), campose_pos.view(-1), im1_pos.view(-1)])
            else:
                seq = torch.cat([im0_seq.view(-1), im1_seq.view(-1)])
                pos = torch.cat([im0_pos.view(-1), im1_pos.view(-1)])

        # Mask out part of frame 1
        seq_delim_idx = im0_seq.numel()

        if unmask_idxs is not None and len(unmask_idxs) > 0:
            seq_delim_idx += len(unmask_idxs) * im1_seq.shape[2]

        if flow_codes is not None:
            seq_delim_idx += flow_seq.numel()
        if campose_codes is not None:
            seq_delim_idx += campose_seq.numel()

        cond_seq = seq[:seq_delim_idx]
        cond_pos = pos[:seq_delim_idx]

        # Grab the number of tokens to generate by subtracting the number of 
        # tokens in the conditional sequence from the total number of tokens
        num_total_tokens = seq.numel() - cond_seq.numel()

        num_seq_tokens = num_seq_patches * im1_seq.shape[2]
        num_par_tokens = num_total_tokens - num_seq_tokens

        # Move motion indixes of seq1 tokens to the front
        if motion_indices is not None:
            im1_seq, im1_pos = self._bring_patches_to_front(
                im1_seq, im1_pos, motion_indices, patch_idx_offset=frame1_patch_offset) 

        # # Grab the indexes of the patches of frame 1 to reveal
        # if rmi is None:
        #     rmi = im1_seq[:, :, 0].view(-1)[len(unmask_idxs):]
        # else:
        #     # if rmi is list convert it to tensor
        #     if isinstance(rmi, list):
        #         rmi = torch.tensor(rmi, device=im1_seq.device, dtype=torch.long)
        # if rmi is None:
        # print(rmi[:10])
        rmi = im1_seq[:, :, 0].view(-1)[len(unmask_idxs):]

        # print(rmi[:10])

        # Make sampling blacklist
        # if motion_indices is not None:
        #     sampling_blacklist = [[self.model.config.flow_range[0] + 11646,
        #                                  self.model.config.flow_range[0] + 11582
        #                                  ]]*(im1_seq.shape[2] * 1)
        # else:
        #     sampling_blacklist = []



        if num_seq_patches > 0:

            # Perform the sequential prediction
            with self.ctx:
                frame1_pred, logits = self.model.rollout(
                    cond_seq.unsqueeze(0).to(self.device).long().clone(), 
                    temperature=temperature,
                    random_masked_indices=rmi.unsqueeze(0).to(self.device).long(),
                    sampling_blacklist=sampling_blacklist,
                    pos=pos.unsqueeze(0).to(self.device).long(),
                    num_new_tokens=num_seq_tokens,
                    top_k=top_k,
                    top_p=top_p,
                    # causal_mask_length=(cond_seq.numel()-im0_seq.numel()),
                    causal_mask_length=cond_seq.numel(),
                    cfg_ratio=cfg_ratio,
                    allowed_tokens=allowed_tokens,
                )

            seq_delim_idx += num_seq_tokens

            cond_seq = frame1_pred.cpu().clone().reshape(-1)
            cond_pos = pos[:seq_delim_idx].clone()

            pred_seq = seq[seq_delim_idx:].clone().view(-1, im1_seq.shape[2])
            pred_pos = pos[seq_delim_idx:].clone().view(-1, im1_pos.shape[2])

            step_seq = torch.cat([cond_seq, pred_seq[:, 0].view(-1)])
            step_pos = cond_pos.clone()
        else:

            frame1_pred = cond_seq.clone()

            seq_delim_idx += num_seq_tokens

            cond_seq = frame1_pred.cpu().clone().reshape(-1)
            cond_pos = pos[:seq_delim_idx].clone()

            pred_seq = seq[seq_delim_idx:].clone().view(-1, im1_seq.shape[2])
            pred_pos = pos[seq_delim_idx:].clone().view(-1, im1_pos.shape[2])

            step_seq = torch.cat([cond_seq, pred_seq[:, 0].view(-1)])
            step_pos = cond_pos.clone()

        all_logits = []

        # Iterate over the patches once for each patch in the second frame
        for it in range(im1_seq.shape[2] - 1):
            step_pos = torch.cat([step_pos, pred_pos[:, it].view(-1)])
            step_mask = torch.zeros(step_pos.shape[0], step_pos.shape[0]).to(step_pos.device)

            # attention mask for frame 0 + unmasked part of frame 1
            step_mask[:, :seq_delim_idx] = 1
            step_mask[:seq_delim_idx, :seq_delim_idx].tril_()

            # attention mask for rest of frame 1
            step_mask[seq_delim_idx:, seq_delim_idx:] = 1

            step_mask = step_mask.unsqueeze(0) # Add a batch dimension to the mask
            step_mask = 1 - step_mask # the forward function assumes 0 to participate in attention, 1 otherwise

            step_tgt = pred_seq[:, it+1].view(-1)

            # Perform the prediction
            with self.ctx:
                logits, loss = self.model(
                    step_seq.unsqueeze(0).to(self.device).long(), 
                    pos=step_pos.unsqueeze(0).to(self.device).long(),
                    mask=step_mask.unsqueeze(0).to(self.device).bool(),
                    tgt=step_tgt.unsqueeze(0).to(self.device).long(),
                )
            # sampled_tokens = self.model.sample_logits(logits, temp=temperature, top_k=top_k, top_p=top_p)[0]

            if first_logits_only: 
                break 
                
            sampled_tokens = self.model.sample_logits(logits.cpu(), temp=0.0)[0]
            step_seq = torch.cat([step_seq, sampled_tokens])

            all_logits.append(logits[0].clone())
        

        sort_order = im1_pos[:, len(unmask_idxs) + num_seq_patches:, 0].argsort()

        if first_logits_only: 
            # all_logits = repeat(all_logits[-1], "n c d -> n 4 c d")
            rgb_logits = logits[0][:, None][sort_order][0]

            # device   = rgb_logits_clone.device                      # logits is (922,4,D)
            N        = im1_seq.shape[1]                  # 1024
            
            # 1) which rows are ALREADY present?  (922, ) int64 on GPU
            used_idx = (pred_seq[..., 0] - frame1_patch_offset).long()

            # 2) build a boolean mask of valid (922) vs missing (102) rows
            mask = torch.zeros(N, dtype=torch.bool, device=self.device)  # kernel 1: fill
            mask[used_idx] = True                                 # kernel 2: scatter

            # 3) allocate the final tensor already filled with dummy-logits
            out = torch.full((N, 1, rgb_logits.shape[-1]), 0,
                            dtype=rgb_logits.dtype, device=self.device)    # kernel 3: fill

            # 4) copy the real logits in a single coalesced write
            out[mask] = rgb_logits
            return None, out, None  

        all_logits = torch.stack(all_logits, dim=1)
        rgb_logits = all_logits[sort_order][0]

        # insert a dummy patch at any missing patch indexes:
        missing_patch_indexes = [i for i in range(im1_seq.shape[1]) if i not in 
            (pred_seq.unsqueeze(0)[:, :, 0] - frame1_patch_offset).tolist()[0]]
        for i in missing_patch_indexes:
            rgb_logits = torch.cat([rgb_logits[:i], torch.zeros_like(rgb_logits[[0]]), rgb_logits[i:]], dim=0)

        pred_seq = step_seq[cond_seq.numel():].reshape(im1_seq.shape[2], -1).permute(1, 0)

        frame1_delim_idx = im1_seq.numel()
        if campose_codes is not None:
            frame1_delim_idx += campose_seq.numel()
        frame1_seq = torch.cat([cond_seq, pred_seq.reshape(-1)])[-im1_seq.numel():]

        # frame1_pred = step_seq[im0_seq.numel():].reshape(im1_seq.shape[2], im1_seq.shape[1]).permute(1, 0)
        frame1_pred = self.model.unpack_and_sort_img_seq(frame1_seq.reshape(1, -1))

        return frame1_pred, rgb_logits, rmi - frame1_patch_offset

    @torch.no_grad()
    def two_frame_sequential_forward(
        self, 
        frame0_codes: torch.LongTensor, 
        frame1_codes: torch.LongTensor, 
        unmask_idxs: List[int],
        frame0_patch_offset: int = None,
        frame0_seq_offset: int = None,
        frame1_patch_offset: int = None,
        frame1_seq_offset: int = None,
        campose_codes: torch.LongTensor = None,
        flow_codes: torch.LongTensor = None,
        top_p: Union[float, List[float]] = 0.9,
        top_k: Union[int, List[int]] = 1000,
        temperature: Union[float, List[float]] = 1.0,
        cfg_ratio: float = None,
    ) -> Tuple[torch.LongTensor, torch.FloatTensor]:
        """
        Perform a forward pass through the model using the "sequential" method.

        Parameters:
            frame0_codes: torch.LongTensor, shape (B, H, W), the quantized image codes for frame 0
            frame1_codes: torch.LongTensor, shape (B, H, W)), the quantized image codes for frame 1
            campose_codes: torch.LongTensor, shape (B, 8), the quantized camera pose codes
            unmask_idxs: List[int], the indexes of the patches to reveal
            top_p: Union[float, List[float]], the top_p value for sampling (optionally per token)
            top_k: Union[int, List[int]], the top_k value for sampling (optionally per token)
            temperature: Union[float, List[float]], the temperature value for sampling (optionally per token)
        
        Returns:
            frame1_pred: torch.LongTensor, shape (B, H, W), the predicted image codes for frame 1
            logits: torch.FloatTensor, shape (B, H, W, C), the logits for the predicted image codes
            decoding_order: torch.LongTensor, shape (B, H), the order in which the patches were decoded
        """

        # Grab default RGB pos and patch idx ranges
        if frame0_patch_offset is None:
            frame0_patch_offset = self.model.config.rgb_patch_0_idx_range[0]
        if frame0_seq_offset is None:
            frame0_seq_offset = self.model.config.frame_0_rgb_pos_range[0]
        if frame1_patch_offset is None:
            frame1_patch_offset = self.model.config.rgb_patch_1_idx_range[0]
        if frame1_seq_offset is None:
            frame1_seq_offset = self.model.config.frame_1_rgb_pos_range[0]
        
        # Pack the images into sequences
        im0_seq, im0_pos = self._pack_image_codes_into_sequence(
            frame0_codes.cpu(), mask=0.0,
            patch_offset=frame0_patch_offset, seq_offset=frame0_seq_offset)
        im1_seq, im1_pos = self._pack_image_codes_into_sequence(
            frame1_codes.cpu(), mask=0.0,
            patch_offset=frame1_patch_offset, seq_offset=frame1_seq_offset)
        
        # Pack the camera pose into a sequence if provided
        if campose_codes is not None:
            campose_seq, campose_pos = self._pack_camera_pose_codes_into_sequence(
                campose_codes.cpu(),
                campose_offse=self.model.config.campose_range[0],
                patch_idx_offset=self.model.config.campose_patch_idx_range[0],
                seq_offset=self.model.config.campose_pos_range[0])
        
        # Bring the revealed patches to the front
        im1_seq, im1_pos = self._bring_patches_to_front(
            im1_seq, im1_pos, unmask_idxs, patch_idx_offset=frame1_patch_offset)

        # Concatenate the two image sequences (and additional conditioning) into a single sequence
        if campose_codes is not None:
            seq = torch.cat([im0_seq.view(-1), campose_seq.view(-1), im1_seq.view(-1)])
            pos = torch.cat([im0_pos.view(-1), campose_pos.view(-1), im1_pos.view(-1)])
        else:
            seq = torch.cat([im0_seq.view(-1), im1_seq.view(-1)])
            pos = torch.cat([im0_pos.view(-1), im1_pos.view(-1)])
        
        if flow_codes is not None:
            flow_seq, flow_pos = self._pack_flow_codes_into_sequence(
                flow_codes.cpu(), mask=0.0,
                patch_offset=self.model.config.flow_patch_idx_range[0],
                seq_offset=self.model.config.flow_pos_range[0])
            seq = torch.cat([seq, flow_seq.view(-1)])
            pos = torch.cat([pos, flow_pos.view(-1)])

        # Mask out part of frame 1
        seq_delim_idx = im0_seq.numel() + len(unmask_idxs) * im1_seq.shape[2]
        # Add camppose sequence length to seq delim index if campose is provided
        if campose_codes is not None:
            seq_delim_idx += campose_seq.numel()

        cond_seq = seq[:seq_delim_idx]
        cond_pos = pos[:seq_delim_idx]

        # Grab the number of tokens to generate by subtracting the number of 
        # tokens in the conditional sequence from the total number of tokens
        num_seq_tokens = seq.numel() - cond_seq.numel()

        # Grab the indexes of the patches of frame 1 to reveal
        rmi = im1_seq[:, :, 0].view(-1)[len(unmask_idxs):]

        # Perform the sequential prediction
        with self.ctx:
            frame1_pred, logits = self.model.rollout(
                cond_seq.unsqueeze(0).to(self.device).long(), 
                temperature=temperature,
                random_masked_indices=rmi.unsqueeze(0).to(self.device).long(),
                pos=pos.unsqueeze(0).to(self.device).long(),
                num_new_tokens=num_seq_tokens,
                top_k=top_k,
                top_p=top_p,
                # causal_mask_length=(cond_seq.numel()-im0_seq.numel()),
                causal_mask_length=cond_seq.numel(),
                cfg_ratio=cfg_ratio,
            )
        
        frame1_pred = frame1_pred[0, -im1_seq.numel():]
        logits = logits[0].reshape(im1_seq.shape[1] - len(unmask_idxs), im1_seq.shape[2], -1)

        sort_order = im1_pos[:, len(unmask_idxs):, 0].argsort()
        # TODO: check if this is correct !!!!
        # Should it be rgb_logits = logits[sort_order][0, :, 1:] instead ????
        rgb_logits = logits[sort_order][0, :, :-1]

        # insert a dummy patch at any missing patch indexes:
        missing_patch_indexes = [i for i in range(im1_seq.shape[1]) if i not in 
            (im1_seq[:, len(unmask_idxs):, 0] - self.model.config.rgb_patch_1_idx_range[0]).tolist()[0]]
        for i in missing_patch_indexes:
            rgb_logits = torch.cat([rgb_logits[:i], torch.zeros_like(rgb_logits[[0]]), rgb_logits[i:]], dim=0)

        frame1_pred = self.model.unpack_and_sort_img_seq(frame1_pred.reshape(1, -1))
             
        return frame1_pred, rgb_logits, rmi - self.model.config.rgb_patch_1_idx_range[0]


    def frame0_flow_frame1_sequential_forward(
            self, frame0_codes: torch.LongTensor,
            flow_codes: torch.LongTensor,
            frame1_codes: torch.LongTensor,
            unmask_idxs: List[int],
            top_p: Union[float, List[float]] = 0.9,
            top_k: Union[int, List[int]] = 1000,
            temperature: Union[float, List[float]] = 1.0,
            decoding_order: torch.LongTensor = None,
            unmask_idxs_img1: List[int] = None,
    ) -> Tuple[torch.LongTensor, torch.FloatTensor]:
        """
        Perform a forward pass through the model using the "sequential" method.

        Parameters:
            frame0_codes: torch.LongTensor, shape (B, H, W), the quantized image codes for frame 0
            flow_codes: torch.LongTensor, shape (B, 2, H, W), the quantized xy codes for flow
            unmask_idxs: List[int], the indexes of the patches to reveal
            top_p: Union[float, List[float]], the top_p value for sampling (optionally per token)
            top_k: Union[int, List[int]], the top_k value for sampling (optionally per token)
            temperature: Union[float, List[float]], the temperature value for sampling (optionally per token)
            decoding_order: torch.LongTensor, shape (B, H/patch_size*W/patch_size), the order in which to decode the patches

        Returns:
            frame1_pred: torch.LongTensor, shape (B, H, W), the predicted image codes for frame 1
            logits: torch.FloatTensor, shape (B, H, W, C), the logits for the predicted image codes
        """

        # Pack the images into sequences

        im0_seq, im0_pos = self._pack_image_codes_into_sequence(
            frame0_codes.cpu(), mask=0.25,
            patch_offset=self.model.config.rgb_patch_0_idx_range[0],
            seq_offset=self.model.config.frame_0_rgb_pos_range[0])
        flow_seq, flow_pos = self._pack_flow_codes_into_sequence(
            flow_codes.cpu(), mask=0.0,
            patch_offset=self.model.config.flow_patch_idx_range[0],
            seq_offset=self.model.config.flow_pos_range[0])

        im1_seq, im1_pos = self._pack_image_codes_into_sequence(
            frame1_codes.cpu(), mask=0.0,
            patch_offset=self.model.config.rgb_patch_1_idx_range[0],
            seq_offset=self.model.config.frame_1_rgb_pos_range[0], shuffle_order=decoding_order)

        # Bring the revealed patches to the front
        flow_seq, flow_pos = self._bring_flow_patches_to_front(flow_seq, flow_pos, unmask_idxs, discard=True)

        if unmask_idxs_img1 is not None:
            im1_seq, im1_pos = self._bring_patches_to_front(im1_seq, im1_pos, unmask_idxs_img1)

        # Concatenate the two image sequences into a single sequence
        seq = torch.cat([im0_seq.view(-1), flow_seq.view(-1), im1_seq.view(-1)])
        pos = torch.cat([im0_pos.view(-1), flow_pos.view(-1), im1_pos.view(-1)])
        tgt = seq.clone()

        # Mask out part of frame 1
        if unmask_idxs_img1 is not None:
            num_unmasked_tokens =  len(unmask_idxs_img1) * im1_seq.shape[2]
        else:
            num_unmasked_tokens = 0

        seq_delim_idx = im0_seq.numel() + len(unmask_idxs) * flow_seq.shape[2]

        cond_seq = seq[:seq_delim_idx]
        cond_pos = pos[:seq_delim_idx]

        # Grab the number of tokens to generate by subtracting the number of
        # tokens in the conditional sequence from the total number of tokens
        num_seq_tokens = seq.numel() - cond_seq.numel()

        # Grab the indexes of the patches of frame 1 to reveal
        rmi = flow_seq[:, :, 0].view(-1)[len(unmask_idxs):]
        # Perform the prediction
        rmi_cutoff = 0 #len(unmask_idxs_img1) if unmask_idxs_img1 is not None else 0
        with self.ctx:
            rmi = im1_seq[:, rmi_cutoff:, 0].view(-1)

            frame1_pred, logits = self.model.rollout(
                cond_seq.unsqueeze(0).to(self.device).long(),
                temperature=temperature,
                random_masked_indices=rmi.unsqueeze(0).to(self.device).long(),
                pos=pos.unsqueeze(0).to(self.device).long(),
                num_new_tokens=num_seq_tokens,
                top_k=top_k,
                top_p=top_p,
                # causal_mask_length=(cond_seq.numel()-im0_seq.numel()),
                causal_mask_length=cond_seq.numel(), #im0_seq.numel() + len(unmask_idxs) * flow_seq.shape[2]
                num_unmasked_tokens=num_unmasked_tokens,
                remaining_seq=seq[seq_delim_idx:].to(self.device).long()
            )


        unmask_idxs = []

        frame1_pred = frame1_pred[0, -im1_seq.numel():]

        # if unmask_idxs_img1 is not None:
        #     frame1_pred = self.model.unpack_and_sort_img_seq(frame1_pred.reshape(1, -1))
        #     return frame1_pred, None, tgt

        logits = logits[0].reshape(im1_seq.shape[1], im1_seq.shape[2], -1)

        sort_order = im1_pos[:, len(unmask_idxs):, 0].argsort()
        rgb_logits = logits[sort_order][0, :, :-1]

        # insert a dummy patch at any missing patch indexes:
        missing_patch_indexes = [i for i in range(im1_seq.shape[1]) if i not in
                                 (im1_seq[:, len(unmask_idxs):, 0] - self.model.config.rgb_patch_1_idx_range[
                                     0]).tolist()[0]]
        for i in missing_patch_indexes:
            rgb_logits = torch.cat([rgb_logits[:i], torch.zeros_like(rgb_logits[[0]]), rgb_logits[i:]], dim=0)

        frame1_pred = self.model.unpack_and_sort_img_seq(frame1_pred.reshape(1, -1))

        return frame1_pred, rgb_logits, tgt

    def _compute_rgb_grid_entropy(self, logits: torch.Tensor, unmasked_idxs = []) -> torch.Tensor:
        """
        Compute the entropy of the RGB grid of the logits.

        Parameters:
            logits: torch.Tensor, shape (B, H * W, C), the logits of the RGB grid
            unmasked_idxs: List[int], the indexes of the unmasked patches
        
        Returns:
            rgb_grid_entropy: torch.Tensor, shape (H, W), the entropy of the RGB grid
        """
        # Extract RGB logits based on the range
        rgb_logits = logits[:, :, self.model.config.rgb_range[0]:self.model.config.rgb_range[1]]
        # Compute the per-patch entropy
        rgb_entropy = F.softmax(rgb_logits, dim=-1) * F.log_softmax(rgb_logits, dim=-1)
        rgb_entropy = -rgb_entropy.sum(dim=-1)
        rgb_patch_entropy = rgb_entropy.mean(dim=1)
        # rgb_patch_entropy = rgb_entropy[:, 0]#.mean(dim=1)
        # Set rgb patch entropy of the unmasked patches to 0
        rgb_patch_entropy[unmasked_idxs] = 0.0
        im_size = int(rgb_entropy.shape[0] ** 0.5)
        rgb_grid_entropy = rgb_entropy.mean(dim=1).view(im_size, im_size)

        return rgb_grid_entropy
    
    def _compute_flow_grid_entropy(self, logits: torch.Tensor, unmasked_idxs = []) -> torch.Tensor:
        """
        Compute the entropy of the RGB grid of the logits.

        Parameters:
            logits: torch.Tensor, shape (B, H * W, C), the logits of the RGB grid
            unmasked_idxs: List[int], the indexes of the unmasked patches
        
        Returns:
            rgb_grid_entropy: torch.Tensor, shape (H, W), the entropy of the RGB grid
        """
        # Extract RGB logits based on the range
        flow_logits = logits[:, :, self.model.config.flow_range[0]:self.model.config.flow_range[1]]
        # Compute the per-patch entropy
        flow_entropy = F.softmax(flow_logits, dim=-1) * F.log_softmax(flow_logits, dim=-1)
        flow_entropy = -flow_entropy.sum(dim=-1)
        flow_patch_entropy = flow_entropy[:, 0]#.mean(dim=1)
        # rgb_patch_entropy = rgb_entropy[:, 0]#.mean(dim=1)
        # Set rgb patch entropy of the unmasked patches to 0
        flow_patch_entropy[unmasked_idxs] = 0.0
        im_size = int(flow_patch_entropy.shape[0] ** 0.5)
        flow_grid_entropy = flow_patch_entropy.view(im_size, im_size)

        return flow_grid_entropy

    def _compute_rgb_grid_varentropy(self, logits: torch.Tensor, unmasked_idxs = []) -> torch.Tensor:
        """
        Compute the varentropy of the RGB grid of the logits.

        Parameters:
            logits: torch.Tensor, shape (B, H * W, C), the logits of the RGB grid
            unmasked_idxs: List[int], the indexes of the unmasked patches
        
        Returns:
            rgb_grid_varentropy: torch.Tensor, shape (H, W), the varentropy of the RGB grid
        """
        # Extract RGB logits based on the range
        rgb_logits = logits[:, :, self.model.config.rgb_range[0]:self.model.config.rgb_range[1]]
        # Compute softmax probabilities
        rgb_probs = F.softmax(rgb_logits, dim=-1)
        # Compute log-probabilities (log-softmax)
        rgb_log_probs = F.log_softmax(rgb_logits, dim=-1)
        # Compute the surprisal (negative log probabilities)
        surprisal = -rgb_log_probs  # shape (B, H*W, C)
        # Compute the first moment (entropy), which is the expected surprisal
        rgb_entropy = (rgb_probs * surprisal).sum(dim=-1)  # shape (B, H*W)
        # Compute the second moment (expected value of the squared surprisal)
        rgb_second_moment = (rgb_probs * surprisal**2).sum(dim=-1)  # shape (B, H*W)
        # Compute the varentropy: second moment - (entropy)^2
        rgb_varentropy = rgb_second_moment - rgb_entropy**2  # shape (B, H*W)
        # Mean across patches to get the per-image varentropy
        rgb_patch_varentropy = rgb_varentropy.mean(dim=1)
        # Set the varentropy of the unmasked patches to 0
        rgb_patch_varentropy[unmasked_idxs] = 0.0
        # Reshape the result into the grid format (H, W)
        im_size = int(rgb_varentropy.shape[0] ** 0.5)
        rgb_grid_varentropy = rgb_patch_varentropy.view(im_size, im_size)
        
        return rgb_grid_varentropy
    
    def _compute_flow_grid_cumulative_probability(
        self,
        logits: torch.Tensor,
        unmasked_idxs,
        target_classes,
        total_range
    ) -> torch.Tensor:
        """
        Compute the cumulative probability of the given target_classes within a 
        specified range of channels (total_range).

        Parameters:
        -----------
        logits : torch.Tensor
            The logits for each patch, shape (B, H*W, C).
        unmasked_idxs : List[int]
            The list of patch indices (0 to H*W-1) for which you want the final 
            probabilities to be set to 0 (e.g., unmasked patches).
        target_classes : List[int]
            The global class indices whose probabilities should be summed.
        total_range : Tuple[int, int]
            The (start, end) slice of the channel dimension to consider when 
            computing probabilities.
        
        Returns:
        --------
        flow_grid_cumulative : torch.Tensor
            A 2D tensor of shape (H, W) containing the summed probability of 
            target_classes at each patch, with unmasked patches set to 0.
        """

        # 1) Slice out the relevant channel range
        #    logits shape: (B, H*W, C) -> (B, H*W, total_range_size)
        start, end = total_range
        subset_logits = logits[:, :, start:end]

        # 2) Compute probabilities with softmax over the sliced channels
        subset_probs = F.softmax(subset_logits, dim=-1)  # shape: (B, H*W, end - start)

        # 3) Adjust target_classes to local indices relative to total_range
        #    (only keep classes that fall within [start, end) )
        local_target_classes = [c - start for c in target_classes if start <= c < end]

        # 4) Sum the probabilities over the target classes
        #    shape: (B, H*W)
        cumulative_prob = subset_probs[:, :, local_target_classes].sum(dim=-1)

        # 5) We want a single patch-level probability grid, so remove batch dim (assuming B=1)
        flow_patch_cumulative = cumulative_prob.squeeze(0)  # shape: (H*W,)

        # 6) Zero out the probability for the unmasked patches
        flow_patch_cumulative[unmasked_idxs] = 1.0

        # 7) Average the probabilities across all tokens in each patch
        flow_patch_cumulative = flow_patch_cumulative[:, 0]

        # 8) Reshape to (H, W)
        im_size = int(flow_patch_cumulative.shape[0] ** 0.5)
        flow_grid_cumulative = flow_patch_cumulative.view(im_size, im_size)

        return flow_grid_cumulative

    def _compute_rgb_grid_ce_error(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute the CE error of the RGB grid of the logits.

        Parameters:
            logits: torch.Tensor, shape (B, H * W, C), the logits of the RGB grid
            targets: torch.Tensor, shape (B, H * W), the target image codes
        
        Returns:
            rgb_grid_ce_error: torch.Tensor, shape (H, W), the CE error of the RGB grid
        """
        # Patchify targets
        patch_targets = patchify_func(targets, patch_size=self.model.config.patch_size)
        # Compute the CE error
        rgb_ce_error = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), patch_targets.reshape(-1), reduce=False)
        # Reshape the result into the grid format (H, W)
        rgb_ce_error = rgb_ce_error.view(1, logits.shape[0], logits.shape[1]).mean(dim=-1)
        # rgb_ce_error_grid = unpatchify(rgb_ce_error)
        im_size = int(logits.shape[0] ** 0.5)
        rgb_grid_ce_error = rgb_ce_error.view(im_size, im_size)

        return rgb_grid_ce_error

    def _compute_rgb_grid_l1_error(self, pred_frame: torch.Tensor, gt_frame: torch.Tensor) -> torch.Tensor:
        """
        Compute the MSE error of the RGB frame.

        Parameters:
            pred_frame: 
        
        Returns:
            rgb_grid_ce_error: torch.Tensor, shape (H, W), the CE error of the RGB grid
        """
        # Compute the L1 error
        rgb_l1_error = F.l1_loss(pred_frame, gt_frame, reduction='none').mean(dim=1)
        # Patchify the L1 error (use config token patch size * 2 to match the patch size of the RGB grid)
        rgb_l1_error = patchify_func(rgb_l1_error, patch_size=self.model.config.patch_size*2).mean(dim=-1)
        # Reshape the result into the grid format (H, W)
        im_size = int(rgb_l1_error.shape[1] ** 0.5)
        rgb_l1_error = rgb_l1_error.view(im_size, im_size)

        return rgb_l1_error

    def _compute_rgb_grid_mse_error(self, pred_frame, gt_frame) -> torch.Tensor:
        """
        Compute the MSE error of the RGB frame.

        Parameters:
            pred_frame: 
        
        Returns:
            rgb_grid_ce_error: torch.Tensor, shape (H, W), the CE error of the RGB grid
        """
        # Compute the MSE error
        rgb_mse_error = F.mse_loss(pred_frame, gt_frame, reduction='none').mean(dim=1)
        # Patchify the L1 error (use config token patch size * 2 to match the patch size of the RGB grid)
        rgb_mse_error = patchify_func(rgb_mse_error, patch_size=self.model.config.patch_size*2).mean(dim=-1)
        # Reshape the result into the grid format (H, W)
        im_size = int(rgb_mse_error.shape[1] ** 0.5)
        rgb_mse_error = rgb_mse_error.view(im_size, im_size)

        return rgb_mse_error

    def _pack_image_codes_into_sequence(
            self, frame_codes: torch.Tensor, mask: float = 0.0, shuffle: bool = True, 
            patch_offset: int = 0, seq_offset: int = 0, shuffle_order: List[int] = None,
            patchify: bool = True,
        ):
        if patchify:
            frame_patches = patchify_func(frame_codes, patch_size=self.model.config.patch_size)
        else:
            frame_patches = frame_codes
        frame_with_idxs = add_patch_indexes(frame_patches, patch_offset)
        frame_pos_idxs = get_pos_idxs(frame_with_idxs, seq_offset)
        shuffled_im0_patches, shuffled_img0_pos_idxs = shuffle_and_trim_values_and_positions(
            frame_with_idxs, frame_pos_idxs, mask=mask, shuffle=shuffle, shuffle_order=shuffle_order)
        
        return shuffled_im0_patches, shuffled_img0_pos_idxs
    
    def _pack_camera_pose_codes_into_sequence(
            self, campose_codes: torch.Tensor, campose_offse: int = 0, 
            patch_idx_offset: int = 0, seq_offset: int = 0
    ):
        campose_with_idxs = torch.cat([torch.tensor([patch_idx_offset], dtype=campose_codes.dtype), 
                                       campose_codes + campose_offse])
        campose_pos_idxs = get_pos_idxs(campose_with_idxs, seq_offset)

        return campose_with_idxs, campose_pos_idxs

    def _pack_flow_codes_into_sequence(
            self, flow_codes: torch.Tensor, mask: float = 0.0, shuffle: bool = True,
            patch_offset: int = 0, seq_offset: int = 0
    ):
        if flow_codes.shape[1] == 2:
            flow_patches_x = patchify_func(flow_codes[:, 0], patch_size=self.model.config.patch_size)
            flow_patches_y = patchify_func(flow_codes[:, 1], patch_size=self.model.config.patch_size)
            flow_patches = torch.stack([flow_patches_x, flow_patches_y], dim=-1).flatten(-2, -1)
            flow_patches += self.model.config.flow_range[0]
            flow_with_idxs = add_patch_indexes(flow_patches, patch_offset)
            flow_pos_idxs = get_pos_idxs(flow_with_idxs, seq_offset)

            flow_decode = decode_flow_code(flow_codes, input_size=256, num_bins=512)
            flow_decode = F.interpolate(flow_decode, scale_factor=1 / self.model.config.patch_size, mode='nearest')[0]
            assert mask == 0, "assume mask==0 for num_flow_patches=shuffled_patches_flow.shape[1], alpha=0"
            # with num_flow_patches=shuffled_patches_flow.shape[1], alpha=0, it will shuffle the flow patches randomly
            shuffled_patches_flow, shuffled_flow_pos_idx = sample_flow_values_and_positions(
                flow_with_idxs, flow_pos_idxs, flow_decode, num_flow_patches=flow_with_idxs.shape[1], alpha=0)
        else:
            flow_patches = patchify_func(flow_codes, patch_size=self.model.config.patch_size)
            flow_patches += self.model.config.flow_range[0]
            flow_with_idxs = add_patch_indexes(flow_patches, patch_offset)
            flow_pos_idxs = get_pos_idxs(flow_with_idxs, seq_offset)
            shuffled_patches_flow, shuffled_flow_pos_idx = shuffle_and_trim_values_and_positions(
                flow_with_idxs, flow_pos_idxs, mask=mask, shuffle=shuffle)

        return shuffled_patches_flow, shuffled_flow_pos_idx
    
    def _set_seed(self, seed: int):
        """
        Set the seed for reproducibility.

        Parameters:
            seed: int, the seed to set
        """
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)

    def _bring_patches_to_front(self, seq: torch.Tensor, pos: torch.Tensor, idxs: List[int],
                                patch_idx_offset: int = None, additional_idx=None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Bring the patches of the given indexes to the front of the sequence and its corresponding positions.

        Parameters:
            seq: torch.Tensor, shape (B, H, W), the sequence of patches
            pos: torch.Tensor, shape (B, H, W), the positions of the patches
            idxs: List[int], the indexes of the patches to bring to the front
        
        Returns:
            reordered_seq: torch.Tensor, shape (B, H, W), the reordered sequence
            reordered_pos: torch.Tensor, shape (B, H, W), the reordered positions
        """
        if patch_idx_offset is None:
            # check if self.model.config has flow_patch_idx_range attribute (rgb model does not)
            if hasattr(self.model.config, 'flow_patch_idx_range'):
                patch_idx_offset = self.model.config.flow_patch_idx_range[0]
            else:
                patch_idx_offset = self.model.config.rgb_patch_1_idx_range[0]
        # Find locations of specified patches within the sequence based on patch indexes in the sequence
        patch_idxs = seq[:, :, 0].view(-1) - patch_idx_offset # self.model.config.flow_patch_idx_range[0] # self.model.config.rgb_patch_1_idx_range[0]
        # patch_idxs = seq[:, :, 0].view(-1) - self.model.config.rgb_patch_1_idx_range[0]
        bring_to_front_idxs = [i for i in range(len(patch_idxs)) if patch_idxs[i] in idxs]
        if additional_idx is not None:
            bring_to_front_idxs.extend([i for i in range(len(patch_idxs)) if patch_idxs[i] == additional_idx])
        # Reorder the sequence and positions
        reordered_seq = torch.cat([seq[:, bring_to_front_idxs, :], 
            seq[:, [i for i in range(seq.shape[1]) if i not in bring_to_front_idxs], :]], dim=1)
        reordered_pos = torch.cat([pos[:, bring_to_front_idxs, :],
            pos[:, [i for i in range(pos.shape[1]) if i not in bring_to_front_idxs], :]], dim=1)
        return reordered_seq, reordered_pos

    def _bring_flow_patches_to_front(self, seq: torch.Tensor, pos: torch.Tensor, idxs: List[int], discard: bool = False,
                                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Bring the patches of the given indexes to the front of the sequence and its corresponding positions.

        Parameters:
            seq: torch.Tensor, shape (B, H, W), the sequence of patches
            pos: torch.Tensor, shape (B, H, W), the positions of the patches
            idxs: List[int], the indexes of the patches to bring to the front

        Returns:
            reordered_seq: torch.Tensor, shape (B, H, W), the reordered sequence
            reordered_pos: torch.Tensor, shape (B, H, W), the reordered positions
        """
        # Find locations of specified patches within the sequence based on patch indexes in the sequence
        patch_idxs = seq[:, :, 0].view(-1) - self.model.config.flow_patch_idx_range[0]
        bring_to_front_idxs = [i for i in range(len(patch_idxs)) if patch_idxs[i] in idxs]
        # Reorder the sequence and positions
        if discard:
            # print('discard remaining')
            reordered_seq = seq[:, bring_to_front_idxs, :]
            reordered_pos = pos[:, bring_to_front_idxs, :]
        else:
            reordered_seq = torch.cat([seq[:, bring_to_front_idxs, :],
                                       seq[:, [i for i in range(seq.shape[1]) if i not in bring_to_front_idxs], :]], dim=1)
            reordered_pos = torch.cat([pos[:, bring_to_front_idxs, :],
                                       pos[:, [i for i in range(pos.shape[1]) if i not in bring_to_front_idxs], :]], dim=1)
        return reordered_seq, reordered_pos
