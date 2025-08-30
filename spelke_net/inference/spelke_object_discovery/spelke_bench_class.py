import torch
from spelke_net.predictor.flow_predictor import LRASFlowPredictor

from spelke_net.utils.segment_zoom import square_crop_with_padding, convert_iterative_bboxes_to_absolute
from spelke_net.utils.camera import pose_list_to_matrix
from spelke_net.utils.segment import compute_avg_flow_from_logits, offset_multiple_centroids, threshold_heatmap
from spelke_net.utils.segment_zoom import get_dot_product_map, sample_distant_point_on_segment, resize_segment_to_original
import numpy as np
import cv2
from spelke_net.utils.model_wrapper import ModelFactory

from abc import ABC, abstractmethod
import numpy as np

class SpelkeBenchModel(ABC):
    """
    Abstract base class for SpelkeBench evaluation models.
    
    All models evaluated on SpelkeBench must inherit from this class
    and implement the required methods.
    """

    def __init__(self):
        """args to initialize the model"""
        return

    @abstractmethod
    def run_inference(self, input_image, poke_point):
        '''
        Run inference on the input image and poke point.
        :param input_image: numpy array of shape [H, W, 3] in [0, 255] range
        :param poke_point: (x, y) tuple representing the poke point in the image, x horizontal, y vertical
        :return: H, W numpy array representing the segment mask
        '''
        pass

    def get_all_segmemts(self, input_image, poke_point_list):
        """
        Get all segments from the input image based on the poke point.
        :param input_image: numpy array of shape [H, W, 3] in [0, 255] range
        :param poke_point_list: [N, 2] list of poke points, where each poke point is a tuple (x, y)
        :return: list of segments
        """

        all_segments = []

        for poke_point in poke_point_list:
            segment = self.run_inference(input_image, poke_point)
            all_segments.append(segment)

        return all_segments


class SpelkeNetInference(SpelkeBenchModel):
    """
    SpelkeNetInference class for running inference on SpelkeNet models.
    Inherits from SpelkeBenchModel.
    """

    def __init__(self, num_zoom_iters=0, num_seq_patches=1, num_seeds=1, num_dirs=1, model_name=None, num_zoom_dirs=1, min_mag_zoom=10, max_mag_zoom=25, topp=None, topk=None):

        super().__init__()

        model_factory = ModelFactory()

        self.token_to_flow_dict = model_factory.load_ckpt("flow_token_to_flow_vector_mapping.pt")

        self.predictor = LRASFlowPredictor(
            model_name,  'rgb_quantizer.pt', 'flow_quantizer.pt', device='cuda:0'
        )

        self.num_zoom_iters = num_zoom_iters
        self.num_seq_patches = num_seq_patches
        self.num_seeds = num_seeds
        self.num_dirs = num_dirs
        self.num_zoom_dirs = num_zoom_dirs
        self.min_mag_zoom = min_mag_zoom
        self.max_mag_zoom = max_mag_zoom
        self.topp = topp
        self.topk = topk

    def get_segment(self, im, probe_point, flow_cond, start, end, token_to_flow_dict, num_seq=0,
                    use_flow_pred=False, num_seeds=1, num_dirs=5, min_mag=10.0, max_mag=25.0, topk=None, topp=None,
                    uncond=False, initial_segment=None, sample_predictor_pokes=False):

        all_dot_prods = []

        all_flows = []

        all_probe_points = []

        probe_point_orig = probe_point.clone()

        for seed in range(num_seeds):

            dx, dy = offset_multiple_centroids(probe_point, num_dirs, min_mag=min_mag, max_mag=max_mag)  # [num_dirs, ]

            for ct in range(num_dirs):

                if initial_segment is not None:

                    second_point = sample_distant_point_on_segment(initial_segment, probe_point_orig, min_dist=9,
                                                                   max_dist=50)

                    if second_point is False:
                        print("No second point found in 100 tries")
                        probe_point = probe_point_orig
                    else:
                        probe_point = torch.stack([probe_point_orig, second_point], 0)  # .float() [2, 2]
                else:
                    probe_point = probe_point_orig[None]  # [1, 2]

                # set random seed to get random point each time
                torch.manual_seed(seed + ct)
                # print(dx.shape)
                x_expand = probe_point[:, 0]  # N,
                y_expand = probe_point[:, 1]
                x_new = x_expand + dx[ct]
                y_new = y_expand + dy[ct]

                pt = torch.stack([x_expand, y_expand, x_new, y_new], 1).tolist()

                if not uncond:

                    flow_cond_with_obj = flow_cond + pt
                else:
                    flow_cond_with_obj = flow_cond

                campose = pose_list_to_matrix([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                if not sample_predictor_pokes:
                    par_predictions = self.predictor.quantized_flow_prediction(
                        im,
                        campose=campose,
                        flow_cond=flow_cond_with_obj,
                        num_seq_patches=num_seq,
                        mode="seq2par",
                        seed=1000 + seed + ct,
                        mask_out=True,
                        temperature=1.0, top_k=topk, top_p=topp,
                    )
                else:

                    # if coordinates is None:
                    #     coordinates = []
                    flowcond = [[0, 0, 0, 0], [248, 248, 248, 248], [248, 0, 248, 0], [0, 248, 0, 248]]

                    par_predictions = self.predictor.quantized_flow_prediction_biased_cond(
                        im,
                        campose=campose,
                        flow_cond=flowcond,
                        num_seq_patches=num_seq,
                        mode="seq2par",
                        seed=np.random.randint(0, 10000),
                        mask_out=True,
                        temperature=1.0, top_k=topk, top_p=topp,
                        probe_point=np.array([x_expand[0], y_expand[0]]),
                        min_mag=20,
                        max_mag=30
                    )

                logits = par_predictions['flow_logits']

                if use_flow_pred:
                    avg_flow = torch.tensor(par_predictions["flow_pred_np"]).to(torch.float32)
                else:
                    avg_flow, _ = compute_avg_flow_from_logits(logits, start, end, token_to_flow_dict)

                dot_product_map = get_dot_product_map(avg_flow, flow_cond_with_obj)

                all_dot_prods.append(dot_product_map)

                all_flows.append(avg_flow)

                # if there is only one point, then we need to repead it and make it 2
                if len(pt) == 1:
                    pt = torch.cat([torch.tensor(pt)] * 2, 0)
                else:
                    pt = torch.tensor(pt)

                all_probe_points.append(pt)

        all_dot_prods = torch.stack(all_dot_prods, 0)
        mean_dot_prod = all_dot_prods.mean(0)

        mean_dot_prod = mean_dot_prod.cpu().numpy()

        all_probe_points = torch.stack(all_probe_points, 0).tolist()

        # Threshold and resize
        segment = threshold_heatmap(mean_dot_prod)
        segment_resized = cv2.resize(segment.astype(np.uint8), (256, 256), interpolation=cv2.INTER_NEAREST)

        all_flows = np.stack(all_flows, 0)

        return segment_resized, all_flows, all_probe_points
    def zoom_into_object(self, im, probe_point, flow_cond, start, end, token_to_flow_dict, num_iters=2,
                         num_seq=0, \
                         use_flow_pred=False, num_seeds=1, num_dirs=5, min_mag=10.0, max_mag=25.0, topk=None, topp=None,
                         sample_predictor_pokes=False):
        all_bbox = [np.array([0, 0, im.shape[1], im.shape[0]])]
        crop = im.copy()
        all_crops = [crop]
        all_probe_points = []
        all_segments = []
        all_flows = []
        all_ratios = [1.0]

        for iters in range(num_iters):
            segment, flows, probe_points_doubled = self.get_segment(crop, probe_point, flow_cond, start, end,
                                                               token_to_flow_dict, num_seq=num_seq, \
                                                               use_flow_pred=use_flow_pred, num_seeds=num_seeds,
                                                               num_dirs=num_dirs,
                                                               min_mag=min_mag, max_mag=max_mag, topk=topk, topp=topp,
                                                               sample_predictor_pokes=sample_predictor_pokes)
            all_segments.append(segment)
            all_probe_points.append(probe_points_doubled)
            all_flows.append(flows)

            # TODO: Here we can play around to see how much belief we want to give on the predicted segment (need to modify the code inside the function)
            crop, crop_mask, x_new, y_new, x_start, y_start, x_end, y_end, ratio = square_crop_with_padding(crop,
                                                                                                            segment,
                                                                                                            probe_point)

            all_ratios.append(ratio)

            # update probe point for the next iter
            probe_point = torch.tensor([x_new, y_new])

            all_crops.append(crop)

            # make bbox and save
            bbox = np.array([x_start, y_start, x_end, y_end])
            all_bbox.append(bbox)

        # get final crop, probe point and estimated parallel segment (for comparison) for doing fully sequential prediction
        final_probe_point = probe_point
        final_crop = crop
        final_segment, flows, probe_points_doubled = self.get_segment(final_crop, probe_point, flow_cond,
                                                                 start, end, token_to_flow_dict, num_seq=num_seq, \
                                                                 use_flow_pred=use_flow_pred, num_seeds=num_seeds,
                                                                 num_dirs=num_dirs,
                                                                 min_mag=min_mag, max_mag=max_mag,
                                                                 sample_predictor_pokes=sample_predictor_pokes)
        all_probe_points.append(probe_points_doubled)
        all_segments.append(final_segment)
        all_flows.append(flows)

        # convert to original image coordinates
        all_bbox = np.stack(all_bbox, 0)
        all_bbox_in_orig = convert_iterative_bboxes_to_absolute(all_bbox, all_ratios)

        # get last crop in original image coordinates
        final_bbox_in_orig = all_bbox_in_orig[-1].astype(np.int32)
        all_probe_points = np.array(all_probe_points)  # [num_iters+1, num_seeds*num_dirs, 2]
        # print(final_bbox_in_orig.shape, final_bbox_in_orig)
        #
        # get full segment in original image coords for visualization -- again to compare to parallel prediction version
        full_segment_in_orig = resize_segment_to_original(final_segment, final_bbox_in_orig, im.shape[:2])

        return {
            "all_flows": all_flows,
            "all_bboxes": all_bbox,
            "all_crops": all_crops,
            "all_probe_points": all_probe_points,
            "all_segments": all_segments,
            "all_ratios": all_ratios,
            "full_segment_in_orig": full_segment_in_orig,
            "final_probe_point": final_probe_point,
            "final_crop": final_crop,
            "final_segment": final_segment,
            "final_bbox_in_orig": final_bbox_in_orig
        }

    def run_inference(self, input_image, poke_point):
        '''
        Run inference on the input image and poke point.
        :param input_image: numpy array of shape [H, W, 3] in [0, 255] range
        :param poke_point: (x, y) tuple representing the poke point in the image, x horizontal, y vertical
        :return:
        '''

        probe_point = torch.tensor(poke_point)

        start, end = self.predictor.model.config.flow_range

        flow_cond_base = [[248, 0, 248, 0], [0, 0, 0, 0], [0, 248, 0, 248], [248, 248, 248, 248]]

        if self.num_zoom_iters > 0:
            zoom_result = self.zoom_into_object(input_image, probe_point, flow_cond_base, start, end,
                                           self.token_to_flow_dict, num_iters=self.num_zoom_iters, num_seq=0,
                                           use_flow_pred=False, num_seeds=1, num_dirs=self.num_zoom_dirs,
                                           min_mag=self.min_mag_zoom, max_mag=self.max_mag_zoom)

            initial_segment = zoom_result["final_segment"]
            # unconditional method
            final_crop = zoom_result["final_crop"]
            final_probe_point = zoom_result["final_probe_point"]
        else:
            initial_segment = None
            final_crop = input_image
            final_probe_point = probe_point

        final_segment, flows, probe_points = self.get_segment(final_crop, final_probe_point, flow_cond_base,
                                                         start, end, self.token_to_flow_dict,
                                                         num_seq=self.num_seq_patches,
                                                         use_flow_pred=True, num_seeds=self.num_seeds,
                                                         num_dirs=self.num_dirs,
                                                         min_mag=self.min_mag_zoom, max_mag=self.max_mag_zoom,
                                                         topk=self.topk, topp=self.topp, uncond=False,
                                                         initial_segment=initial_segment)
        if self.num_zoom_iters > 0:
            final_bbox_in_orig = zoom_result["final_bbox_in_orig"]
            final_segment = resize_segment_to_original(final_segment, final_bbox_in_orig, input_image.shape[:2])

        return final_segment

class SpelkeNetModel1B(SpelkeBenchModel):
    """
    SpelkeBench model for SpelkeNet inference.
    Inherits from SpelkeBenchModel.
    """

    def __init__(self):
        super().__init__()

        num_zoom_iters = 2
        num_seq_patches = 256
        num_seeds = 3
        num_dirs = 8
        model_name = "SpelkeNet1B.pt"
        num_zoom_dirs = 5

        self.inference = SpelkeNetInference(num_zoom_iters=num_zoom_iters, num_seq_patches=num_seq_patches,
                                            num_seeds=num_seeds, num_dirs=num_dirs,
                                            model_name=model_name, num_zoom_dirs=num_zoom_dirs
                                            )

    def run_inference(self, input_image, poke_point):
        return self.inference.run_inference(input_image, poke_point)

class SpelkeNetModel1BFast(SpelkeBenchModel):
    """
    SpelkeBench model for SpelkeNet inference.
    Inherits from SpelkeBenchModel.
    """

    def __init__(self):
        super().__init__()

        num_zoom_iters = 1
        num_seq_patches = 2
        num_seeds = 1
        num_dirs = 4
        model_name = "SpelkeNet1B.pt"
        num_zoom_dirs = 1

        self.inference = SpelkeNetInference(num_zoom_iters=num_zoom_iters, num_seq_patches=num_seq_patches,
                                            num_seeds=num_seeds, num_dirs=num_dirs,
                                            model_name=model_name, num_zoom_dirs=num_zoom_dirs
                                            )

    def run_inference(self, input_image, poke_point):
        return self.inference.run_inference(input_image, poke_point)

class SpelkeNetModel7B(SpelkeBenchModel):
    """
    SpelkeBench model for SpelkeNet inference.
    Inherits from SpelkeBenchModel.
    """

    def __init__(self):
        super().__init__()

        num_zoom_iters = 2
        num_seq_patches = 256
        num_seeds = 3
        num_dirs = 8
        model_name = "SpelkeNet7B.pt"
        num_zoom_dirs = 5

        self.inference = SpelkeNetInference(num_zoom_iters=num_zoom_iters, num_seq_patches=num_seq_patches,
                                            num_seeds=num_seeds, num_dirs=num_dirs,
                                            model_name=model_name, num_zoom_dirs=num_zoom_dirs
                                           )
        def run_inference(self, input_image, poke_point):
            return self.inference.run_inference(input_image, poke_point)