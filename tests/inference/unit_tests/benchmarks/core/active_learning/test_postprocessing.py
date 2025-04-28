from inference.core.active_learning.post_processing import (
    adjust_prediction_to_client_scaling_factor,
    predictions_should_not_be_post_processed,
    adjust_object_detection_predictions_to_client_scaling_factor,
    adjust_prediction_with_bbox_and_points_to_client_scaling_factor,
    adjust_bbox_coordinates_to_client_scaling_factor,
    adjust_points_coordinates_to_client_scaling_factor,
    encode_prediction,
)


from inference.core.constants import (
    CLASSIFICATION_TASK,
    INSTANCE_SEGMENTATION_TASK,
    OBJECT_DETECTION_TASK,
)
from copy import deepcopy


from inference_cli.lib.benchmark.dataset import load_dataset_images
import pytest
import numpy as np


@pytest.fixture
def dataset_reference() -> tuple[list[np.ndarray], set[tuple[int, int]]]:
    dataset_images = load_dataset_images(
        dataset_reference="coco",
    )
    return dataset_images, {i.shape[:2] for i in dataset_images}


@pytest.mark.parametrize(
    "prediction_key,scaling_factor",
    [
        (prediction_key, scaling_factor)
        for prediction_key in [
            OBJECT_DETECTION_TASK,
            INSTANCE_SEGMENTATION_TASK,
            CLASSIFICATION_TASK,
            "stub",
            "empty",
        ]
        for scaling_factor in [0.5, 1.0, 2.0]
    ],
)
def test_adjust_prediction_to_client_scaling_factor(
    benchmark, prediction_key, scaling_factor
):
    predictions_base = {
        OBJECT_DETECTION_TASK: {
            "image": {"width": 200, "height": 150},
            "predictions": [
                {
                    "x": 50,
                    "y": 75,
                    "width": 20,
                    "height": 30,
                    "class": "A",
                    "confidence": 0.9,
                },
                {
                    "x": 100,
                    "y": 50,
                    "width": 10,
                    "height": 15,
                    "class": "B",
                    "confidence": 0.8,
                },
            ],
        },
        INSTANCE_SEGMENTATION_TASK: {
            "image": {"width": 200, "height": 150},
            "predictions": [
                {
                    "x": 50,
                    "y": 75,
                    "width": 20,
                    "height": 30,
                    "class": "A",
                    "confidence": 0.9,
                    "points": [
                        {"x": 50, "y": 75},
                        {"x": 70, "y": 75},
                        {"x": 70, "y": 105},
                        {"x": 50, "y": 105},
                    ],
                },
            ],
        },
        CLASSIFICATION_TASK: {
            "image": {"width": 200, "height": 150},
            "predictions": {"cat": {"confidence": 0.9}, "dog": {"confidence": 0.1}},
            "top": "cat",
            "confidence": 0.9,
        },
        "stub": {
            "is_stub": True,
            "image": {"width": 200, "height": 150},
        },
        "empty": {
            "image": {"width": 200, "height": 150},
            "predictions": [],
        },
    }

    prediction_input = deepcopy(predictions_base[prediction_key])

    if prediction_key == "stub" or prediction_key == "empty":
        actual_prediction_type = OBJECT_DETECTION_TASK
    else:
        actual_prediction_type = prediction_key

    benchmark(
        adjust_prediction_to_client_scaling_factor,
        prediction=prediction_input,
        scaling_factor=scaling_factor,
        prediction_type=actual_prediction_type,
    )


@pytest.mark.parametrize(
    "prediction_list",
    [
        [
            {
                "x": 50,
                "y": 75,
                "width": 20,
                "height": 30,
                "class": "A",
                "confidence": 0.9,
            },
            {
                "x": 100,
                "y": 50,
                "width": 10,
                "height": 15,
                "class": "B",
                "confidence": 0.8,
            },
        ],
        [],
        [
            {
                "x": 10,
                "y": 20,
                "width": 5,
                "height": 8,
                "class": "C",
                "confidence": 0.7,
            }
        ],
    ],
)
@pytest.mark.parametrize("scaling_factor", [0.5, 1.0, 2.0])
def test_adjust_object_detection_predictions_to_client_scaling_factor(
    benchmark, prediction_list, scaling_factor
):
    prediction_input = deepcopy(prediction_list)
    benchmark(
        adjust_object_detection_predictions_to_client_scaling_factor,
        predictions=prediction_input,
        scaling_factor=scaling_factor,
    )


@pytest.mark.parametrize(
    "prediction_list",
    [
        [
            {
                "x": 50,
                "y": 75,
                "width": 20,
                "height": 30,
                "class": "A",
                "confidence": 0.9,
                "points": [
                    {"x": 50, "y": 75},
                    {"x": 70, "y": 75},
                    {"x": 70, "y": 105},
                    {"x": 50, "y": 105},
                ],
                "keypoints": [
                    {"x": 55, "y": 80, "confidence": 0.95, "class_id": 0},
                    {"x": 65, "y": 80, "confidence": 0.92, "class_id": 1},
                ],
            },
            {
                "x": 100,
                "y": 50,
                "width": 10,
                "height": 15,
                "class": "B",
                "confidence": 0.8,
                "points": [
                    {"x": 100, "y": 50},
                    {"x": 110, "y": 50},
                    {"x": 110, "y": 65},
                    {"x": 100, "y": 65},
                ],
                "keypoints": [
                    {"x": 105, "y": 55, "confidence": 0.88, "class_id": 0},
                ],
            },
        ],
        [],
        [
            {
                "x": 10,
                "y": 20,
                "width": 5,
                "height": 8,
                "class": "C",
                "confidence": 0.7,
                "points": [{"x": 10, "y": 20}, {"x": 15, "y": 28}],
                "keypoints": [{"x": 12, "y": 24, "confidence": 0.75, "class_id": 0}],
            }
        ],
    ],
)
@pytest.mark.parametrize("scaling_factor", [0.5, 1.0, 2.0])
@pytest.mark.parametrize("points_key", ["points", "keypoints"])
def test_adjust_prediction_with_bbox_and_points_to_client_scaling_factor(
    benchmark, prediction_list, scaling_factor, points_key
):
    prediction_input = deepcopy(prediction_list)
    # Ensure the correct points key exists for the test case
    for pred in prediction_input:
        if points_key not in pred:
            pred[points_key] = []  # Add empty list if key missing

    benchmark(
        adjust_prediction_with_bbox_and_points_to_client_scaling_factor,
        predictions=prediction_input,
        scaling_factor=scaling_factor,
        points_key=points_key,
    )


@pytest.mark.parametrize(
    "bbox",
    [
        {"x": 50, "y": 75, "width": 20, "height": 30},
        {"x": 100, "y": 50, "width": 10, "height": 15},
        {"x": 0, "y": 0, "width": 1, "height": 1},
        {"x": 199, "y": 149, "width": 1, "height": 1},
    ],
)
@pytest.mark.parametrize("scaling_factor", [0.5, 1.0, 2.0, 3.5])
def test_adjust_bbox_coordinates_to_client_scaling_factor(
    benchmark, bbox, scaling_factor
):
    bbox_input = deepcopy(bbox)
    benchmark(
        adjust_bbox_coordinates_to_client_scaling_factor,
        bbox=bbox_input,
        scaling_factor=scaling_factor,
    )


@pytest.mark.parametrize(
    "points",
    [
        [
            {"x": 50, "y": 75},
            {"x": 70, "y": 75},
            {"x": 70, "y": 105},
            {"x": 50, "y": 105},
        ],
        [{"x": 100, "y": 50}],
        [],
        [{"x": 0, "y": 0}, {"x": 199, "y": 149}],
    ],
)
@pytest.mark.parametrize("scaling_factor", [0.5, 1.0, 2.0, 3.5])
def test_adjust_points_coordinates_to_client_scaling_factor(
    benchmark, points, scaling_factor
):
    points_input = deepcopy(points)
    benchmark(
        adjust_points_coordinates_to_client_scaling_factor,
        points=points_input,
        scaling_factor=scaling_factor,
    )


@pytest.mark.parametrize(
    "prediction_type,prediction_input",
    [
        (
            "object_detection",
            {
                "image": {"width": 200, "height": 150},
                "predictions": [
                    {
                        "x": 50,
                        "y": 75,
                        "width": 20,
                        "height": 30,
                        "class": "A",
                        "confidence": 0.9,
                    },
                ],
            },
        ),
        (
            "instance_segmentation",
            {
                "image": {"width": 200, "height": 150},
                "predictions": [
                    {
                        "x": 50,
                        "y": 75,
                        "width": 20,
                        "height": 30,
                        "class": "A",
                        "confidence": 0.9,
                        "points": [{"x": 50, "y": 75}, {"x": 70, "y": 105}],
                    },
                ],
            },
        ),
        (
            "classification",
            {
                "image": {"width": 200, "height": 150},
                "predictions": {"cat": {"confidence": 0.9}, "dog": {"confidence": 0.1}},
                "top": "cat",
                "confidence": 0.9,
            },
        ),
        (
            "stub",
            {"is_stub": True, "image": {"width": 200, "height": 150}},
        ),
        (
            "empty",
            {"image": {"width": 200, "height": 150}, "predictions": []},
        ),
    ],
)
def test_encode_prediction(benchmark, prediction_type, prediction_input):
    prediction_input_copy = deepcopy(prediction_input)
    benchmark(
        encode_prediction,
        prediction=prediction_input_copy,
        prediction_type=prediction_type,
    )


@pytest.mark.parametrize(
    "prediction_input",
    [
        {"predictions": [{"x": 10}]},
        {"predictions": []},
        {"top": "cat"},
        {"is_stub": True},
        {},
    ],
)
@pytest.mark.parametrize(
    "prediction_type",
    [
        OBJECT_DETECTION_TASK,
        INSTANCE_SEGMENTATION_TASK,
        CLASSIFICATION_TASK,
        "some_other_task",
    ],
)
def test_predictions_should_not_be_post_processed(
    benchmark, prediction_input, prediction_type
):
    prediction_input_copy = deepcopy(prediction_input)
    benchmark(
        predictions_should_not_be_post_processed,
        prediction=prediction_input_copy,
        prediction_type=prediction_type,
    )
