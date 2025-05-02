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


def test_benchmark_adjust_prediction_to_client_scaling_factor_single_case(benchmark):
    prediction_input = {
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
    }
    scaling_factor = 2.0
    prediction_type = OBJECT_DETECTION_TASK
    benchmark(
        adjust_prediction_to_client_scaling_factor,
        prediction=deepcopy(prediction_input),
        scaling_factor=scaling_factor,
        prediction_type=prediction_type,
    )


def test_benchmark_adjust_object_detection_predictions_to_client_scaling_factor_single_case(
    benchmark,
):
    prediction_input = [
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
    ]
    scaling_factor = 2.0
    benchmark(
        adjust_object_detection_predictions_to_client_scaling_factor,
        predictions=deepcopy(prediction_input),
        scaling_factor=scaling_factor,
    )


def test_benchmark_adjust_prediction_with_bbox_and_points_to_client_scaling_factor_single_case(
    benchmark,
):
    prediction_list = [
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
    ]
    scaling_factor = 2.0
    points_key = "points"
    prediction_input = deepcopy(prediction_list)
    for pred in prediction_input:
        if points_key not in pred:
            pred[points_key] = []
    benchmark(
        adjust_prediction_with_bbox_and_points_to_client_scaling_factor,
        predictions=prediction_input,
        scaling_factor=scaling_factor,
        points_key=points_key,
    )


def test_benchmark_adjust_bbox_coordinates_to_client_scaling_factor_single_case(
    benchmark,
):
    bbox_input = {"x": 50, "y": 75, "width": 20, "height": 30}
    scaling_factor = 2.0
    benchmark(
        adjust_bbox_coordinates_to_client_scaling_factor,
        bbox=deepcopy(bbox_input),
        scaling_factor=scaling_factor,
    )


def test_benchmark_adjust_points_coordinates_to_client_scaling_factor_single_case(
    benchmark,
):
    points_input = [
        {"x": 50, "y": 75},
        {"x": 70, "y": 75},
        {"x": 70, "y": 105},
        {"x": 50, "y": 105},
    ]
    scaling_factor = 2.0
    benchmark(
        adjust_points_coordinates_to_client_scaling_factor,
        points=deepcopy(points_input),
        scaling_factor=scaling_factor,
    )


def test_benchmark_encode_prediction_object_detection_single_case(benchmark):
    prediction_input = {
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
    }
    prediction_type = "object_detection"
    benchmark(
        encode_prediction,
        prediction=deepcopy(prediction_input),
        prediction_type=prediction_type,
    )


def test_benchmark_predictions_should_not_be_post_processed_single_case(benchmark):
    prediction_input = {"predictions": [{"x": 10}]}
    prediction_type = OBJECT_DETECTION_TASK
    benchmark(
        predictions_should_not_be_post_processed,
        prediction=prediction_input,
        prediction_type=prediction_type,
    )
