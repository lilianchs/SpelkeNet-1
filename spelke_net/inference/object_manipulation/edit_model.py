import cv2
import h5py
import numpy as np
import torch
from edit_bench.models.object_edit_class import ObjectEditingModel

from spelke_net.inference.object_manipulation.obj_motion_utils import get_dense_flow_from_segment_depth_RT, \
    downsample_and_scale_flow, \
    combine_dilated_bounding_boxes, get_true_pixel_coords, get_flattened_index_from_2d_index, \
    get_unmask_indices_from_flow_map, to_tensor_segment, convert_segment_map_to_3d_coords, project_pixels
from spelke_net.predictor.flow_predictor import LRASFlowPredictor
from spelke_net.utils.camera import get_camera_orientation_dict_from_threepoints_depth_intrinsics
from spelke_net.utils.flow import compute_quantize_flow
from spelke_net.utils.model_wrapper import ModelFactory


class LRAS3D():

    def __init__(self, precomputed_3dedit_bench_segments_path=None, segment_type='GT'):

        super().__init__()

        rollout_config = {"temperature": 0.9, "top_k": 1000, "top_p": 0.9, "rollout_mode": "sequential", "seed": 48}

        # load depth model
        from spelke_net.external.depth_anything_v2.metric_depth.depth_anything_v2.dpt import DepthAnythingV2

        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]}
        }
        encoder = 'vitl'  # or 'vits', 'vitb'
        max_depth = 20  # 20 for indoor model, 80 for outdoor model
        depth_model = DepthAnythingV2(**{**model_configs[encoder], 'max_depth': max_depth})

        model_factory = ModelFactory()

        depth_model.load_state_dict(model_factory.load_ckpt("depth_anything_v2.pth"))

        depth_model.eval()

        self.depth_model = depth_model.to("cuda:0")

        self.ccwm_rgb_predictor  = LRASFlowPredictor(
        'LRAS_3D_7B.pt', 'rgb_quantizer.pt', device='cuda:0')


        self.set_rollout_config(rollout_config)

        self.viz = False

        self.special_decode_order = False

        self.new_segment_sampling = True

        self.condition_from_nvs = True

        self.condition_rgb = True

        self.precomputed_3dedit_bench_segments_path = precomputed_3dedit_bench_segments_path

        self.segment_type = segment_type



    def run_forward(self, image, image_id, point_prompt, R, T, K, gt_segment):

        if self.segment_type == 'GT':
            gt_segment = gt_segment
        elif self.segment_type == 'SAM':
            with h5py.File(self.precomputed_3dedit_bench_segments_path + image_id, 'r') as f:
                gt_segment = f['SAM_segment'][:]
                gt_segment = cv2.resize(gt_segment.astype('uint8'), (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        elif self.segment_type == 'SpelkeNet':
            with h5py.File(self.precomputed_3dedit_bench_segments_path + image_id, 'r') as f:
                gt_segment = f['SpelkeNet_segment'][:]
                gt_segment = cv2.resize(gt_segment.astype('uint8'), (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

        image0_downsampled = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)

        depth_img0 = self.predict_depth0_from_rgb0(image / 255)['depth0_numpy']

        image_1_gt = None

        unmask_indices_rgb1 = None

        flow_map, segment_map, seg_map_frame1, unmask_indices = self.get_inputs_for_forward(
            image, depth_img0, K, R, T, gt_segment, self.new_segment_sampling,
            use_full_segmentation=False, num_fg_flows=140, num_bg_flows=60)


        if self.condition_from_nvs:

            flow_map_nvs, _, seg_map_frame1_nvs, unmask_indices_nvs = self.get_inputs_for_forward(
                image, depth_img0, K, R, T, gt_segment, self.new_segment_sampling,
                use_full_segmentation=True, num_fg_flows=200, num_bg_flows=0)

            if self.viz:
                rgb1_pred_nvs = image0_downsampled
            else:
                rgb1_pred_nvs = self.predict_rgb1_from_flow(image0_downsampled, flow_map_nvs[:, [1, 0]],
                                                               unmask_indices_nvs)

            segment_map_32 = self.downsamle_segment_map(seg_map_frame1, kernel_size=32)

            unmask_indices_rgb1 = self.get_unmask_inds_fromseg_map(segment_map_32)

            unmask_indices_rgb1 = unmask_indices_rgb1[:2]

            image_1_gt = np.array(rgb1_pred_nvs)

        if self.condition_rgb:

            combined_seg_map = combine_dilated_bounding_boxes(segment_map.astype('uint8') * 255,
                                                              seg_map_frame1.astype('uint8') * 255, kernel_size=40)

            combined_seg_map_256 = cv2.resize(combined_seg_map, (256, 256), interpolation=cv2.INTER_AREA)
            combined_seg_map_256 = combined_seg_map_256.astype('bool')

            combined_seg_map = cv2.resize(combined_seg_map, (32, 32), interpolation=cv2.INTER_AREA)
            combined_seg_map = combined_seg_map.astype('bool')
            unmask_indices_rgb1_combined = get_flattened_index_from_2d_index(get_true_pixel_coords(~combined_seg_map),
                                                                             combined_seg_map.shape[0])
            unmask_indices_rgb1_combined = unmask_indices_rgb1_combined.tolist()
            np.random.shuffle(unmask_indices_rgb1_combined)

            unmask_indices_rgb1_combined = unmask_indices_rgb1_combined[:2]

            if unmask_indices_rgb1 is not None:
                unmask_indices_rgb1 = unmask_indices_rgb1 + unmask_indices_rgb1_combined

            if image_1_gt is not None:
                image_1_gt[~combined_seg_map_256] = image0_downsampled[~combined_seg_map_256]
            else:
                image_1_gt = image0_downsampled


        rgb1_pred = self.predict_rgb1_from_flow(image0_downsampled, flow_map[:, [1, 0]],
                                                                  unmask_indices, None, image_1_gt,
                                                                  unmask_indices_rgb1)

        return np.array(rgb1_pred)

    def set_rollout_config(self, rollout_config):

        self.temperature = rollout_config["temperature"]
        self.top_k = rollout_config["top_k"]
        self.top_p = rollout_config["top_p"]
        self.rollout_mode = rollout_config["rollout_mode"]
        self.seed = rollout_config["seed"]

    @torch.no_grad()
    def predict_depth0_from_rgb0(self, rgb0_numpy_0to1):
        rgb0_numpy = (rgb0_numpy_0to1 * 255).astype(np.uint8)
        rgb0_numpy_bgr = cv2.cvtColor(rgb0_numpy, cv2.COLOR_RGB2BGR)
        depth0_numpy = self.depth_model.infer_image(rgb0_numpy_bgr) #, device="cuda")
        depth0_tensor = torch.tensor(depth0_numpy).unsqueeze(0).unsqueeze(0)  # 1x1xHxW
        return {"depth0_tensor": depth0_tensor, "depth0_numpy": depth0_numpy}

    def get_dense_flow_from_sparse_correspondences(self, image, depth_map0, K, R, T, gt_segment, use_full_segmentation=False):

        segment_map = gt_segment
        #resize segment map to image resolution
        segment_map = cv2.resize(segment_map.astype(np.uint8), (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        segment_map = segment_map.astype(bool)

        if use_full_segmentation:
            segment_map = np.ones_like(segment_map).astype(np.uint8)

        # segment the object and densify flow by warping the depth with R and T
        flow_map, segment_coords_in_pixels_img0, segment_coords_in_3d_img1, segment_map_frame1  = get_dense_flow_from_segment_depth_RT(segment_map, depth_map0, R, T, K)

        colors = image[segment_coords_in_pixels_img0[:, 0], segment_coords_in_pixels_img0[:, 1]]

        return flow_map, R, T, segment_map, segment_coords_in_3d_img1, colors, segment_map_frame1

    def predict_rgb1_from_flow(self, rgb0_numpy, flow_tensor_cuda, unmask_indices, decode_order=None, rgb1_numpy=None, unmask_indices_img1=None):
        '''
        rgb0_numpy: [H, W, 3]: (0, 255)
        flow_tensor_cuda: max pooled flow: [1, 2, 64, 64]
        unmask_indices: List of indices of where to reveal flow
        '''

        rgb0_numpy_0to1 = rgb0_numpy.astype(np.float32) / 255

        if rgb1_numpy is not None:
            rgb1_numpy_0to1 = rgb1_numpy.astype(np.float32) / 255
        else:
            rgb1_numpy_0to1 = rgb0_numpy_0to1

        flow_codes = compute_quantize_flow(flow_tensor_cuda, input_size=256, num_bins=512)

        # Quantize rgb
        rgb_prediction = self.ccwm_rgb_predictor.flow_factual_prediction(
            rgb0_numpy_0to1, rgb1_numpy_0to1, flow=flow_codes, unmask_indices=unmask_indices,
            mode=self.rollout_mode, seed=self.seed,
            temperature=self.temperature, top_k=self.top_k, top_p=self.top_p, decoding_order=decode_order, unmask_indices_img1=unmask_indices_img1
        )

        return rgb_prediction['frame1_pred_pil']


    def get_inputs_for_forward(self, image, depth_img0, K, R, T, gt_segment, new_segment_sampling=False, use_full_segmentation=False, num_fg_flows=80, num_bg_flows=20):


        flow_map, R, T, seg_map_frame0, coords_3d, colors, frame1_estimated_segment_map = self.get_dense_flow_from_sparse_correspondences(
            image, depth_img0, K, R, T, gt_segment, use_full_segmentation=use_full_segmentation)

        flow_map, indices_flow_in_256 = downsample_and_scale_flow(flow_map)

        unmask_indices = get_unmask_indices_from_flow_map(flow_map, num_fg_flows=num_fg_flows, num_bg_flows=num_bg_flows,
                                                          new_sampling_method=new_segment_sampling)

        return flow_map, seg_map_frame0, frame1_estimated_segment_map, unmask_indices

    def downsamle_segment_map(self, segment_map, kernel_size=32):

        segment_map_tensor = to_tensor_segment(segment_map)

        segment_map_tensor_fg = -torch.nn.functional.max_pool2d(-segment_map_tensor.float(), kernel_size=kernel_size, stride=kernel_size)[
            0, 0]
        segment_map_tensor_fg = segment_map_tensor_fg.bool().cpu().numpy()

        return segment_map_tensor_fg

    def get_unmask_inds_fromseg_map(self, segment_map):

        unmask_indices_rgb1 = get_flattened_index_from_2d_index(get_true_pixel_coords(~segment_map),
                                                                segment_map.shape[0])
        unmask_indices_rgb1 = unmask_indices_rgb1.tolist()
        np.random.shuffle(unmask_indices_rgb1)

        return unmask_indices_rgb1

    def prepare_inputs_world_method(self, threepoints_on_ground, image, K, depth_img0, R_world, T_world, gt_segment, new_segment_sampling=False, full_segment_map=False):

        cam_orientation = get_camera_orientation_dict_from_threepoints_depth_intrinsics(threepoints_on_ground, K[None],
                                                                                        depth_img0)
        cam_to_world = np.array(cam_orientation['transform_world_from_camera'][0][:3, :3])
        world_to_cam = np.array(cam_orientation['transform_camera_from_world'][:3, :3])

        # get centroid of object in camera coordinate system
        segment_map = gt_segment #self.get_segment_from_points(image, start_points)
        #resize segment map to image resolution
        segment_map = cv2.resize(segment_map.astype(np.uint8), (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        object_coords_in_3d_img0, _ = convert_segment_map_to_3d_coords(segment_map,
                                                                                                    depth_img0, K)
        centroid = np.mean(object_coords_in_3d_img0, axis=0)

        # make point cloud from segment map
        if full_segment_map:
            segment_map_for_flow = np.ones_like(segment_map)
        else:
            segment_map_for_flow = segment_map

        segment_coords_in_cam_3d_img0, segment_coords_in_pixels_img0 = convert_segment_map_to_3d_coords(
            segment_map_for_flow,
            depth_img0, K)

        # covert point cloud to new coordinate system centered at the centroid of the object

        segment_coords_in_cam_3d_img0 = segment_coords_in_cam_3d_img0 - centroid[None, :]
        segment_coords_in_world_3d_img0 = np.matmul(cam_to_world, segment_coords_in_cam_3d_img0.T).T

        # rotate the scene about the new coordinate system and then translate
        segment_coords_in_world_3d_img0 = np.matmul(R_world, segment_coords_in_world_3d_img0.T).T + T_world

        # bring back to camera system and untranslate
        segment_coords_in_cam_3d_img1 = np.matmul(world_to_cam, segment_coords_in_world_3d_img0.T).T + centroid[None, :]

        # project points to get flow
        segment_coords_in_pixels_img1 = project_pixels(segment_coords_in_cam_3d_img1, K)

        # get segment map: TODO need to make it from the segmented image
        segment_coords_in_pixels_img1 = segment_coords_in_pixels_img1.astype(int)
        segment_coords_in_pixels_img1 = np.clip(segment_coords_in_pixels_img1, 0, segment_map.shape[0] - 1)
        segment_map_img1_estimated = np.zeros([segment_map.shape[0], segment_map.shape[1]])
        segment_map_img1_estimated[segment_coords_in_pixels_img1[:, 0], segment_coords_in_pixels_img1[:, 1]] = 1

        # compute flow vectors
        flow_ = segment_coords_in_pixels_img1[:, :2] - segment_coords_in_pixels_img0

        # make flow map
        flow_map = np.zeros([segment_map.shape[0], segment_map.shape[1], 2])
        flow_map[segment_coords_in_pixels_img0[:, 0], segment_coords_in_pixels_img0[:, 1]] = flow_

        flow_map, indices_flow_in_256 = downsample_and_scale_flow(flow_map)

        if full_segment_map:
            num_fg_flows = 200
            num_bg_flows = 0
        else:
            num_fg_flows = 140
            num_bg_flows = 60

        unmask_indices_nvs = get_unmask_indices_from_flow_map(flow_map, num_fg_flows=num_fg_flows, num_bg_flows=num_bg_flows,
                                                              new_sampling_method=new_segment_sampling)

        return flow_map, unmask_indices_nvs, indices_flow_in_256, segment_map, segment_coords_in_pixels_img0, segment_coords_in_pixels_img1, segment_map_img1_estimated

    def run_forward_with_RT(self, image, threepoints_on_ground, R_world, T_world, K, gt_segment, condition_rgb=False, new_segment_sampling=False, condition_from_nvs=False, full_segment_map=False):

        '''
        :param image: [H, W, 3] in [0, 1] range
        :param start_points: [N, 2] a list of K points for each mask, K >=3
        :param end_points: [N, 2] a list of K points for each mask, K >=3
        :return:
            counterfactual_image: [H, W, 3]
        '''

        image0_downsampled = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)

        depth_img0 = self.predict_depth0_from_rgb0(image / 255)['depth0_numpy']

        flow_map, unmask_indices, indices_flow_in_256, segment_map, _, _, _ = \
            self.prepare_inputs_world_method(threepoints_on_ground, image, K, depth_img0, R_world, T_world, gt_segment, new_segment_sampling, full_segment_map)

        if self.viz:
            rgb1_pred = image0_downsampled
            cum_log_prob = 0
        else:
            rgb1_pred = self.predict_rgb1_from_flow(image0_downsampled, flow_map[:, [1, 0]], unmask_indices)

        if full_segment_map or (not(condition_from_nvs) and not(condition_rgb)):
            return rgb1_pred
        else:
            flow_map, unmask_indices_obj_motion, indices_flow_in_256, _, _, _, segment_map_img1_estimated = \
                self.prepare_inputs_world_method(threepoints_on_ground, image, K, depth_img0, R_world,
                                                 T_world, gt_segment, new_segment_sampling, full_segment_map=False)

            image_1_gt = None

            unmask_indices_rgb1 = None

            if condition_from_nvs:

                segment_map_32 = self.downsamle_segment_map(segment_map_img1_estimated, kernel_size=32)
                unmask_indices_rgb1 = self.get_unmask_inds_fromseg_map(segment_map_32)
                unmask_indices_rgb1 = unmask_indices_rgb1[:2]
                image_1_gt = np.array(rgb1_pred)

            if condition_rgb:
                combined_seg_map = combine_dilated_bounding_boxes(segment_map.astype('uint8') * 255,
                                                                  segment_map_img1_estimated.astype('uint8') * 255, kernel_size=40)

                combined_seg_map_256 = cv2.resize(combined_seg_map, (256, 256), interpolation=cv2.INTER_AREA)
                combined_seg_map_256 = combined_seg_map_256.astype('bool')

                combined_seg_map = cv2.resize(combined_seg_map, (32, 32), interpolation=cv2.INTER_AREA)
                combined_seg_map = combined_seg_map.astype('bool')
                unmask_indices_rgb1_combined = get_flattened_index_from_2d_index(
                    get_true_pixel_coords(~combined_seg_map),
                    combined_seg_map.shape[0])
                unmask_indices_rgb1_combined = unmask_indices_rgb1_combined.tolist()
                np.random.shuffle(unmask_indices_rgb1_combined)

                unmask_indices_rgb1_combined = unmask_indices_rgb1_combined[:2]

                if unmask_indices_rgb1 is not None:
                    unmask_indices_rgb1 = unmask_indices_rgb1 + unmask_indices_rgb1_combined

                if image_1_gt is not None:
                    image_1_gt[~combined_seg_map_256] = image0_downsampled[~combined_seg_map_256]
                else:
                    image_1_gt = image0_downsampled

            if self.viz:
                rgb1_pred = image0_downsampled
                cum_log_prob = 0
            else:
                rgb1_pred = self.predict_rgb1_from_flow(image0_downsampled, flow_map[:, [1, 0]],
                                                                  unmask_indices_obj_motion, None, image_1_gt,
                                                                  unmask_indices_rgb1)

            return rgb1_pred


class ObjectEditModelGT(ObjectEditingModel):
    def __init__(self):
        super().__init__()
        self.model_object_edit = LRAS3D(segment_type='GT')

    def run_forward(self, image, image_id, point_prompt, R, T, K, gt_segment):
        return self.model_object_edit.run_forward(image, image_id, point_prompt, R, T, K, gt_segment)

class ObjectEditModelSAM(ObjectEditingModel):
    def __init__(self):
        super().__init__()
        precomputed_3dedit_bench_segments_path = './datasets/precomputed_segments/'
        self.model_object_edit = LRAS3D(precomputed_3dedit_bench_segments_path, 'SAM')

    def run_forward(self, image, image_id, point_prompt, R, T, K, gt_segment):
        return self.model_object_edit.run_forward(image, image_id, point_prompt, R, T, K, gt_segment)

class ObjectEditModelSpelkeNet(ObjectEditingModel):
    def __init__(self):
        super().__init__()
        precomputed_3dedit_bench_segments_path = './datasets/precomputed_segments/'
        self.model_object_edit = LRAS3D(precomputed_3dedit_bench_segments_path, 'SpelkeNet')

    def run_forward(self, image, image_id, point_prompt, R, T, K, gt_segment):
        return self.model_object_edit.run_forward(image, image_id, point_prompt, R, T, K, gt_segment)