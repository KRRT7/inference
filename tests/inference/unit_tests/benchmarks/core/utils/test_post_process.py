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


@pytest.mark.parametrize(
    "vec1, vec2",
    [
        (np.array([1, 2, 3]), np.array([1, 2, 3])),
        (np.random.rand(10000), np.random.rand(10000)),
        (np.random.rand(1000000), np.random.rand(1000000)),
    ],
)
def test_cosine_similarity(benchmark, vec1, vec2):
    benchmark(
        cosine_similarity,
        vec1,
        vec2,
    )


@pytest.mark.parametrize(
    "num_masks, height, width",
    [
        (100, 100, 100),
        (1000, 256, 256),
        (10, 1024, 1024),
    ],
)
def test_masks2poly(benchmark, num_masks, height, width):
    masks = np.random.randint(0, 2, (num_masks, height, width)).astype(np.uint8)
    benchmark(masks2poly, masks)


@pytest.mark.parametrize(
    "num_masks, height, width",
    [
        (100, 100, 100),
        (1000, 256, 256),
        (10, 1024, 1024),
    ],
)
def test_masks2multipoly(benchmark, num_masks, height, width):
    masks = np.random.randint(0, 2, (num_masks, height, width)).astype(np.uint8)
    benchmark(masks2multipoly, masks)


@pytest.mark.parametrize(
    "height, width",
    [
        (100, 100),
        (256, 256),
        (1024, 1024),
    ],
)
def test_mask2poly(benchmark, height, width):
    mask = np.random.randint(0, 2, (height, width)).astype(np.uint8)
    benchmark(mask2poly, mask)


@pytest.mark.parametrize(
    "height, width",
    [
        (100, 100),
        (256, 256),
        (1024, 1024),
    ],
)
def test_mask2multipoly(benchmark, height, width):
    mask = np.random.randint(0, 2, (height, width)).astype(np.uint8)
    benchmark(mask2multipoly, mask)


@pytest.mark.parametrize(
    "num_predictions, infer_shape, origin_shape",
    [
        (10, (640, 640), (1280, 1280)),
        (50, (640, 480), (1920, 1080)),
        (20, (320, 320), (800, 800)),
    ],
)
def test_stretch_bboxes(benchmark, num_predictions, infer_shape, origin_shape):
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


@pytest.mark.parametrize(
    "batch_size, num_predictions, infer_shape, img_dims_list, preproc, resize_method, disable_static_crop",
    [
        (
            1,
            10,
            (640, 640),
            [(1280, 1280)],
            {},
            "Stretch to",
            False,
        ),
        (
            4,
            50,
            (640, 480),
            [(1920, 1080), (1280, 720), (800, 600), (640, 480)],
            {},
            "Fit (black edges) in",
            False,
        ),
        (
            2,
            20,
            (320, 320),
            [(640, 640), (800, 800)],
            {STATIC_CROP_KEY: {"x_min": 0, "y_min": 0, "x_max": 100, "y_max": 100}},
            "Fit (white edges) in",
            True,
        ),
    ],
)
def test_post_process_bboxes(
    benchmark,
    batch_size,
    num_predictions,
    infer_shape,
    img_dims_list,
    preproc,
    resize_method,
    disable_static_crop,
):
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


@pytest.mark.parametrize(
    "num_predictions, infer_shape, origin_shape",
    [
        (10, (640, 640), (1280, 1280)),
        (50, (640, 480), (1920, 1080)),
        (20, (320, 320), (800, 800)),
    ],
)
def test_undo_image_padding_for_predicted_boxes(
    benchmark, num_predictions, infer_shape, origin_shape
):
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


@pytest.mark.parametrize(
    "num_predictions, origin_shape",
    [
        (10, (1280, 1280)),
        (50, (1920, 1080)),
        (20, (800, 800)),
    ],
)
def test_clip_boxes_coordinates(benchmark, num_predictions, origin_shape):
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


@pytest.mark.parametrize(
    "num_predictions, shift_x, shift_y",
    [
        (10, 100, 50),
        (50, -20, 30),
        (20, 0, 0),
    ],
)
def test_shift_bboxes(benchmark, num_predictions, shift_x, shift_y):
    bboxes = np.random.rand(num_predictions, 4) * 640
    bboxes[:, 2] = np.maximum(bboxes[:, 0] + 1, bboxes[:, 2])
    bboxes[:, 3] = np.maximum(bboxes[:, 1] + 1, bboxes[:, 3])

    benchmark(shift_bboxes, bboxes=bboxes, shift_x=shift_x, shift_y=shift_y)


@pytest.mark.parametrize(
    "num_masks, proto_channels, proto_h, proto_w, shape",
    [
        (10, 32, 160, 160, (640, 640)),
        (50, 64, 80, 80, (1920, 1080)),
        (20, 16, 40, 40, (800, 800)),
    ],
)
def test_preprocess_segmentation_masks(
    benchmark, num_masks, proto_channels, proto_h, proto_w, shape
):
    protos = np.random.rand(proto_channels, proto_h, proto_w).astype(np.float32)
    masks_in = np.random.rand(num_masks, proto_channels).astype(np.float32)

    benchmark(
        preprocess_segmentation_masks,
        protos=protos,
        masks_in=masks_in,
        shape=shape,
    )


@pytest.mark.parametrize(
    "num_masks, proto_channels, proto_h, proto_w, shape",
    [
        (10, 32, 160, 160, (640, 640)),
        (50, 64, 80, 80, (1920, 1080)),
        (20, 16, 40, 40, (800, 800)),
    ],
)
def test_process_mask_accurate(
    benchmark, num_masks, proto_channels, proto_h, proto_w, shape
):
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


@pytest.mark.parametrize(
    "num_masks, proto_channels, proto_h, proto_w, shape, tradeoff_factor",
    [
        (10, 32, 160, 160, (640, 640), 0.5),
        (50, 64, 80, 80, (1920, 1080), 0.2),
        (20, 16, 40, 40, (800, 800), 0.8),
    ],
)
def test_process_mask_tradeoff(
    benchmark, num_masks, proto_channels, proto_h, proto_w, shape, tradeoff_factor
):
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


@pytest.mark.parametrize(
    "num_masks, proto_channels, proto_h, proto_w, shape",
    [
        (10, 32, 160, 160, (640, 640)),
        (50, 64, 80, 80, (1920, 1080)),
        (20, 16, 40, 40, (800, 800)),
    ],
)
def test_process_mask_fast(
    benchmark, num_masks, proto_channels, proto_h, proto_w, shape
):
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


@pytest.mark.parametrize(
    "num_masks, height, width",
    [
        (10, 160, 160),
        (50, 80, 80),
        (20, 40, 40),
    ],
)
def test_crop_mask(benchmark, num_masks, height, width):
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


@pytest.mark.parametrize(
    "num_polygons, num_points, x_scale, y_scale",
    [
        (10, 5, 1.5, 1.5),
        (100, 10, 0.5, 0.8),
        (500, 8, 2.0, 1.0),
    ],
)
def test_scale_polygons(benchmark, num_polygons, num_points, x_scale, y_scale):
    polygons = [
        [(np.random.rand() * 640, np.random.rand() * 480) for _ in range(num_points)]
        for _ in range(num_polygons)
    ]
    benchmark(scale_polygons, polygons=polygons, x_scale=x_scale, y_scale=y_scale)


@pytest.mark.parametrize(
    "num_polygons, num_points, origin_shape, infer_shape",
    [
        (10, 5, (640, 640), (320, 320)),
        (100, 10, (1920, 1080), (640, 480)),
        (500, 8, (800, 800), (400, 400)),
    ],
)
def test_undo_image_padding_for_predicted_polygons(
    benchmark, num_polygons, num_points, origin_shape, infer_shape
):
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


@pytest.mark.parametrize(
    "static_crop_config",
    [
        ({"x_min": 10, "y_min": 10, "x_max": 90, "y_max": 90}),
        ({"x_min": 0, "y_min": 20, "x_max": 80, "y_max": 100}),
        ({"x_min": 25, "y_min": 25, "x_max": 75, "y_max": 75}),
    ],
)
def test_standardise_static_crop(benchmark, static_crop_config):
    benchmark(standardise_static_crop, static_crop_config=static_crop_config)


@pytest.mark.parametrize(
    "batch_size, num_predictions, num_keypoints, keypoints_start_index, infer_shape, img_dims_list, preproc, resize_method, disable_static_crop",
    [
        (
            1,
            10,
            17,
            4,
            (640, 640),
            [(1280, 1280)],
            {},
            "Stretch to",
            False,
        ),
        (
            4,
            50,
            5,
            4,
            (640, 480),
            [(1920, 1080), (1280, 720), (800, 600), (640, 480)],
            {},
            "Fit (black edges) in",
            False,
        ),
        (
            2,
            20,
            10,
            4,
            (320, 320),
            [(640, 640), (800, 800)],
            {STATIC_CROP_KEY: {"x_min": 0, "y_min": 0, "x_max": 100, "y_max": 100}},
            "Fit (white edges) in",
            True,
        ),
    ],
)
def test_post_process_keypoints(
    benchmark,
    batch_size,
    num_predictions,
    num_keypoints,
    keypoints_start_index,
    infer_shape,
    img_dims_list,
    preproc,
    resize_method,
    disable_static_crop,
):
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


@pytest.mark.parametrize(
    "num_predictions, num_keypoints, infer_shape, origin_shape",
    [
        (10, 17, (640, 640), (1280, 1280)),
        (50, 5, (640, 480), (1920, 1080)),
        (20, 10, (320, 320), (800, 800)),
    ],
)
def test_stretch_keypoints(
    benchmark, num_predictions, num_keypoints, infer_shape, origin_shape
):
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


@pytest.mark.parametrize(
    "num_predictions, num_keypoints, infer_shape, origin_shape",
    [
        (10, 17, (640, 640), (1280, 1280)),
        (50, 5, (640, 480), (1920, 1080)),
        (20, 10, (320, 320), (800, 800)),
    ],
)
def test_undo_image_padding_for_predicted_keypoints(
    benchmark, num_predictions, num_keypoints, infer_shape, origin_shape
):
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


@pytest.mark.parametrize(
    "num_predictions, num_keypoints, origin_shape",
    [
        (10, 17, (1280, 1280)),
        (50, 5, (1920, 1080)),
        (20, 10, (800, 800)),
    ],
)
def test_clip_keypoints_coordinates(
    benchmark, num_predictions, num_keypoints, origin_shape
):
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


@pytest.mark.parametrize(
    "num_predictions, num_keypoints, shift_x, shift_y",
    [
        (10, 17, 100, 50),
        (50, 5, -20, 30),
        (20, 10, 0, 0),
    ],
)
def test_shift_keypoints(benchmark, num_predictions, num_keypoints, shift_x, shift_y):
    total_keypoint_coords = num_keypoints * 3
    keypoints = np.random.rand(num_predictions, total_keypoint_coords) * 640

    benchmark(
        shift_keypoints,
        keypoints=keypoints.copy(),
        shift_x=shift_x,
        shift_y=shift_y,
    )


@pytest.mark.parametrize(
    "input_data",
    [
        (0.5),
        (-10.0),
        (np.random.rand(100) * 20 - 10),
        (np.random.rand(10000) * 20 - 10),
        (np.random.rand(100, 100) * 20 - 10),
    ],
)
def test_sigmoid(benchmark, input_data):
    benchmark(sigmoid, x=input_data)
