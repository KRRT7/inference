import numpy as np
import pytest
from inference.core.utils.postprocess import (
    cosine_similarity,
    masks2poly,
    masks2multipoly,
    mask2poly,
    mask2multipoly,
    post_process_bboxes,
    stretch_bboxes,
    undo_image_padding_for_predicted_boxes,
    clip_boxes_coordinates,
    shift_bboxes,
    process_mask_accurate,
    process_mask_tradeoff,
    process_mask_fast,
    preprocess_segmentation_masks,
    scale_bboxes,  # noqa: F401
    crop_mask,
    post_process_polygons,  # noqa: F401
    scale_polygons,
    undo_image_padding_for_predicted_polygons,
    get_static_crop_dimensions,  # noqa: F401
    standardise_static_crop,
    post_process_keypoints,
    stretch_keypoints,
    undo_image_padding_for_predicted_keypoints,
    clip_keypoints_coordinates,
    shift_keypoints,
    sigmoid,
)
from inference.core.utils.preprocess import STATIC_CROP_KEY


def test_cosine_similarity(benchmark):
    benchmark(
        cosine_similarity,
        np.random.rand(10000),
        np.random.rand(10000),
    )


def test_masks2poly_single_case(benchmark):
    num_masks, height, width = 1000, 256, 256
    masks = np.random.randint(0, 2, (num_masks, height, width)).astype(np.uint8)
    benchmark(masks2poly, masks)


def test_masks2multipoly_single_case(benchmark):
    num_masks, height, width = 1000, 256, 256
    masks = np.random.randint(0, 2, (num_masks, height, width)).astype(np.uint8)
    benchmark(masks2multipoly, masks)


def test_mask2poly_single_case(benchmark):
    height, width = 1024, 1024
    mask = np.random.randint(0, 2, (height, width)).astype(np.uint8)
    benchmark(mask2poly, mask)


def test_mask2multipoly_single_case(benchmark):
    height, width = 1024, 1024
    mask = np.random.randint(0, 2, (height, width)).astype(np.uint8)
    benchmark(mask2multipoly, mask)


def test_stretch_bboxes_single_case(benchmark):
    num_predictions = 50
    infer_shape = (640, 480)
    origin_shape = (1920, 1080)
    predicted_bboxes = np.random.rand(num_predictions, 4)
    predicted_bboxes[:, 0] *= infer_shape[1]
    predicted_bboxes[:, 1] *= infer_shape[0]
    predicted_bboxes[:, 2] *= infer_shape[1]
    predicted_bboxes[:, 3] *= infer_shape[0]
    predicted_bboxes[:, 2] = np.maximum(
        predicted_bboxes[:, 0] + 1, predicted_bboxes[:, 2]
    )
    predicted_bboxes[:, 3] = np.maximum(
        predicted_bboxes[:, 1] + 1, predicted_bboxes[:, 3]
    )

    benchmark(
        stretch_bboxes,
        predicted_bboxes=predicted_bboxes,
        infer_shape=infer_shape,
        origin_shape=origin_shape,
    )


def test_post_process_bboxes(benchmark):
    batch_size = 4
    num_predictions = 50
    infer_shape = (640, 480)
    img_dims_list = [(1920, 1080), (1280, 720), (800, 600), (640, 480)]
    preproc = {}
    resize_method = "Fit (black edges) in"
    disable_static_crop = False

    predictions = [
        np.random.rand(num_predictions, 6).tolist() for _ in range(batch_size)
    ]
    predictions_np = np.array(predictions)
    predictions_np[..., 0] *= infer_shape[1]
    predictions_np[..., 1] *= infer_shape[0]
    predictions_np[..., 2] *= infer_shape[1]
    predictions_np[..., 3] *= infer_shape[0]
    predictions_np[..., 2] = np.maximum(
        predictions_np[..., 0] + 1, predictions_np[..., 2]
    )
    predictions_np[..., 3] = np.maximum(
        predictions_np[..., 1] + 1, predictions_np[..., 3]
    )
    predictions = predictions_np.tolist()

    benchmark(
        post_process_bboxes,
        predictions=predictions,
        infer_shape=infer_shape,
        img_dims=img_dims_list,
        preproc=preproc,
        disable_preproc_static_crop=disable_static_crop,
        resize_method=resize_method,
    )


def test_undo_image_padding_for_predicted_boxes(benchmark):
    num_predictions = 50
    infer_shape = (640, 480)
    origin_shape = (1920, 1080)
    predicted_bboxes = np.random.rand(num_predictions, 4)
    predicted_bboxes[:, 0] *= infer_shape[1]
    predicted_bboxes[:, 1] *= infer_shape[0]
    predicted_bboxes[:, 2] *= infer_shape[1]
    predicted_bboxes[:, 3] *= infer_shape[0]
    predicted_bboxes[:, 2] = np.maximum(
        predicted_bboxes[:, 0] + 1, predicted_bboxes[:, 2]
    )
    predicted_bboxes[:, 3] = np.maximum(
        predicted_bboxes[:, 1] + 1, predicted_bboxes[:, 3]
    )

    benchmark(
        undo_image_padding_for_predicted_boxes,
        predicted_bboxes=predicted_bboxes,
        infer_shape=infer_shape,
        origin_shape=origin_shape,
    )


def test_clip_boxes_coordinates(benchmark):
    num_predictions = 50
    origin_shape = (1920, 1080)
    predicted_bboxes = np.random.rand(num_predictions, 4)
    predicted_bboxes[:, 0] = (predicted_bboxes[:, 0] * 1.2 - 0.1) * origin_shape[1]
    predicted_bboxes[:, 1] = (predicted_bboxes[:, 1] * 1.2 - 0.1) * origin_shape[0]
    predicted_bboxes[:, 2] = (predicted_bboxes[:, 2] * 1.2 - 0.1) * origin_shape[1]
    predicted_bboxes[:, 3] = (predicted_bboxes[:, 3] * 1.2 - 0.1) * origin_shape[0]
    predicted_bboxes[:, 2] = np.maximum(
        predicted_bboxes[:, 0] + 1, predicted_bboxes[:, 2]
    )
    predicted_bboxes[:, 3] = np.maximum(
        predicted_bboxes[:, 1] + 1, predicted_bboxes[:, 3]
    )

    benchmark(
        clip_boxes_coordinates,
        predicted_bboxes=predicted_bboxes,
        origin_shape=origin_shape,
    )


def test_shift_bboxes(benchmark):
    num_predictions = 50
    shift_x = -20
    shift_y = 30
    bboxes = np.random.rand(num_predictions, 4) * 640
    bboxes[:, 2] = np.maximum(bboxes[:, 0] + 1, bboxes[:, 2])
    bboxes[:, 3] = np.maximum(bboxes[:, 1] + 1, bboxes[:, 3])

    benchmark(shift_bboxes, bboxes=bboxes, shift_x=shift_x, shift_y=shift_y)


def test_preprocess_segmentation_masks(benchmark):
    num_masks, proto_channels, proto_h, proto_w, shape = 50, 64, 80, 80, (1920, 1080)
    protos = np.random.rand(proto_channels, proto_h, proto_w).astype(np.float32)
    masks_in = np.random.rand(num_masks, proto_channels).astype(np.float32)

    benchmark(
        preprocess_segmentation_masks,
        protos=protos,
        masks_in=masks_in,
        shape=shape,
    )


def test_process_mask_accurate(benchmark):
    num_masks, proto_channels, proto_h, proto_w, shape = 50, 64, 80, 80, (1920, 1080)
    protos = np.random.rand(proto_channels, proto_h, proto_w).astype(np.float32)
    masks_in = np.random.rand(num_masks, proto_channels).astype(np.float32)
    bboxes = np.random.rand(num_masks, 4)
    bboxes[:, 0] *= shape[1]
    bboxes[:, 1] *= shape[0]
    bboxes[:, 2] *= shape[1]
    bboxes[:, 3] *= shape[0]
    bboxes[:, 2] = np.maximum(bboxes[:, 0] + 1, bboxes[:, 2])
    bboxes[:, 3] = np.maximum(bboxes[:, 1] + 1, bboxes[:, 3])

    benchmark(
        process_mask_accurate,
        protos=protos,
        masks_in=masks_in,
        bboxes=bboxes,
        shape=shape,
    )


def test_process_mask_tradeoff(benchmark):
    num_masks, proto_channels, proto_h, proto_w, shape, tradeoff_factor = (
        50,
        64,
        80,
        80,
        (1920, 1080),
        0.2,
    )
    protos = np.random.rand(proto_channels, proto_h, proto_w).astype(np.float32)
    masks_in = np.random.rand(num_masks, proto_channels).astype(np.float32)
    bboxes = np.random.rand(num_masks, 4)
    bboxes[:, 0] *= shape[1]
    bboxes[:, 1] *= shape[0]
    bboxes[:, 2] *= shape[1]
    bboxes[:, 3] *= shape[0]
    bboxes[:, 2] = np.maximum(bboxes[:, 0] + 1, bboxes[:, 2])
    bboxes[:, 3] = np.maximum(bboxes[:, 1] + 1, bboxes[:, 3])

    benchmark(
        process_mask_tradeoff,
        protos=protos,
        masks_in=masks_in,
        bboxes=bboxes,
        shape=shape,
        tradeoff_factor=tradeoff_factor,
    )


def test_process_mask_fast(benchmark):
    num_masks, proto_channels, proto_h, proto_w, shape = 50, 64, 80, 80, (1920, 1080)
    protos = np.random.rand(proto_channels, proto_h, proto_w).astype(np.float32)
    masks_in = np.random.rand(num_masks, proto_channels).astype(np.float32)
    bboxes = np.random.rand(num_masks, 4)
    bboxes[:, 0] *= shape[1]
    bboxes[:, 1] *= shape[0]
    bboxes[:, 2] *= shape[1]
    bboxes[:, 3] *= shape[0]
    bboxes[:, 2] = np.maximum(bboxes[:, 0] + 1, bboxes[:, 2])
    bboxes[:, 3] = np.maximum(bboxes[:, 1] + 1, bboxes[:, 3])

    benchmark(
        process_mask_fast,
        protos=protos,
        masks_in=masks_in,
        bboxes=bboxes,
        shape=shape,
    )


def test_scale_bboxes(benchmark): ...


def test_crop_mask(benchmark):
    num_masks, height, width = 50, 80, 80
    masks = np.random.rand(num_masks, height, width).astype(np.float32)
    boxes = np.random.rand(num_masks, 4)
    boxes[:, 0] *= width
    boxes[:, 1] *= height
    boxes[:, 2] *= width
    boxes[:, 3] *= height
    boxes[:, 0] = np.clip(boxes[:, 0], 0, width - 1)
    boxes[:, 1] = np.clip(boxes[:, 1], 0, height - 1)
    boxes[:, 2] = np.clip(np.maximum(boxes[:, 0] + 1, boxes[:, 2]), 0, width)
    boxes[:, 3] = np.clip(np.maximum(boxes[:, 1] + 1, boxes[:, 3]), 0, height)

    benchmark(crop_mask, masks=masks, boxes=boxes)


def test_post_process_polygons(): ...  # TODO implement benchmarks


def test_scale_polygons(benchmark):
    num_polygons = 500
    num_points = 8
    x_scale = 2.0
    y_scale = 1.0
    polygons = [
        [(np.random.rand() * 640, np.random.rand() * 480) for _ in range(num_points)]
        for _ in range(num_polygons)
    ]
    benchmark(scale_polygons, polygons=polygons, x_scale=x_scale, y_scale=y_scale)


def test_undo_image_padding_for_predicted_polygons(benchmark):
    num_polygons = 500
    num_points = 8
    origin_shape = (800, 800)
    infer_shape = (400, 400)
    polygons = [
        [
            (np.random.rand() * infer_shape[1], np.random.rand() * infer_shape[0])
            for _ in range(num_points)
        ]
        for _ in range(num_polygons)
    ]
    benchmark(
        undo_image_padding_for_predicted_polygons,
        polygons=polygons,
        origin_shape=origin_shape,
        infer_shape=infer_shape,
    )


def test_get_static_crop_dimensions(benchmark): ...  # TODO: implement benchmarks


def test_standardise_static_crop(benchmark):
    static_crop_config = {"x_min": 0, "y_min": 20, "x_max": 80, "y_max": 100}
    benchmark(standardise_static_crop, static_crop_config=static_crop_config)


def test_post_process_keypoints(benchmark):
    batch_size = 4
    num_predictions = 50
    num_keypoints = 5
    keypoints_start_index = 4
    infer_shape = (640, 480)
    img_dims_list = [(1920, 1080), (1280, 720), (800, 600), (640, 480)]
    preproc = {}
    resize_method = "Fit (black edges) in"
    disable_static_crop = False

    total_keypoint_coords = num_keypoints * 3
    prediction_length = keypoints_start_index + total_keypoint_coords
    predictions = [
        np.random.rand(num_predictions, prediction_length).tolist()
        for _ in range(batch_size)
    ]
    predictions_np = np.array(predictions)
    for kp_idx in range(num_keypoints):
        predictions_np[..., keypoints_start_index + kp_idx * 3] *= infer_shape[1]
        predictions_np[..., keypoints_start_index + kp_idx * 3 + 1] *= infer_shape[0]

    predictions = predictions_np.tolist()

    benchmark(
        post_process_keypoints,
        predictions=predictions,
        keypoints_start_index=keypoints_start_index,
        infer_shape=infer_shape,
        img_dims=img_dims_list,
        preproc=preproc,
        disable_preproc_static_crop=disable_static_crop,
        resize_method=resize_method,
    )


def test_stretch_keypoints_most_intensive_case(benchmark):
    num_predictions = 50
    num_keypoints = 5
    infer_shape = (640, 480)
    origin_shape = (1920, 1080)
    total_keypoint_coords = num_keypoints * 3
    keypoints = np.random.rand(num_predictions, total_keypoint_coords)
    for kp_idx in range(num_keypoints):
        keypoints[:, kp_idx * 3] *= infer_shape[1]
        keypoints[:, kp_idx * 3 + 1] *= infer_shape[0]

    benchmark(
        stretch_keypoints,
        keypoints=keypoints.copy(),
        infer_shape=infer_shape,
        origin_shape=origin_shape,
    )


def test_undo_image_padding_for_predicted_keypoints_most_intensive_case(benchmark):
    num_predictions = 50
    num_keypoints = 5
    infer_shape = (640, 480)
    origin_shape = (1920, 1080)
    total_keypoint_coords = num_keypoints * 3
    keypoints = np.random.rand(num_predictions, total_keypoint_coords)
    for kp_idx in range(num_keypoints):
        keypoints[:, kp_idx * 3] *= infer_shape[1]
        keypoints[:, kp_idx * 3 + 1] *= infer_shape[0]

    benchmark(
        undo_image_padding_for_predicted_keypoints,
        keypoints=keypoints.copy(),
        infer_shape=infer_shape,
        origin_shape=origin_shape,
    )


def test_clip_keypoints_coordinates_most_intensive_case(benchmark):
    num_predictions = 50
    num_keypoints = 5
    origin_shape = (1920, 1080)
    total_keypoint_coords = num_keypoints * 3
    keypoints = np.random.rand(num_predictions, total_keypoint_coords)
    for kp_idx in range(num_keypoints):
        keypoints[:, kp_idx * 3] = (
            keypoints[:, kp_idx * 3] * 1.2 - 0.1
        ) * origin_shape[1]
        keypoints[:, kp_idx * 3 + 1] = (
            keypoints[:, kp_idx * 3 + 1] * 1.2 - 0.1
        ) * origin_shape[0]

    benchmark(
        clip_keypoints_coordinates,
        keypoints=keypoints.copy(),
        origin_shape=origin_shape,
    )


def test_shift_keypoints_most_intensive_case(benchmark):
    num_predictions = 50
    num_keypoints = 5
    shift_x = -20
    shift_y = 30
    total_keypoint_coords = num_keypoints * 3
    keypoints = np.random.rand(num_predictions, total_keypoint_coords) * 640

    benchmark(
        shift_keypoints,
        keypoints=keypoints.copy(),
        shift_x=shift_x,
        shift_y=shift_y,
    )


def test_sigmoid_most_intensive_case(benchmark):
    input_data = np.random.rand(10000) * 20 - 10
    benchmark(sigmoid, x=input_data)
