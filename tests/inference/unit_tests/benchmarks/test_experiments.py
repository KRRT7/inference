from inference.core.registries.roboflow import RoboflowModelRegistry
from inference.models.utils import ROBOFLOW_MODEL_TYPES
from inference_cli.lib.benchmark.dataset import load_dataset_images
from inference.core.models.base import Model
import numpy as np
import pytest
from inference.models.yolov8 import yolov8_keypoints_detection
from inference.models.yolov8 import codeflash_yolov8_keypoints_detection


@pytest.fixture
def dataset_reference() -> tuple[list[np.ndarray], set[tuple[int, int]]]:
    dataset_images = load_dataset_images(
        dataset_reference="coco",
    )
    return dataset_images, {i.shape[:2] for i in dataset_images}


def test_roboflow_model_registry(benchmark):
    model_registry = RoboflowModelRegistry(ROBOFLOW_MODEL_TYPES)
    benchmark(
        model_registry.get_model,
        "yolov8n-seg-640",
        api_key=None,
    )


# won't work due to onix session being inside the predict method
# also currently fails with FileNotFoundError: [Errno 2] No such file or directory: '/tmp/cache/coco-dataset-vdnr1/2/keypoints_metadata.json'
# def test_yolov8n(benchmark, dataset_reference):
#     images, image_sizes = dataset_reference
#     inference_configuration = {}

#     model = yolov8_keypoints_detection.YOLOv8KeypointsDetection(
#         model_id="yolov8n-seg-640",
#         api_key=None,
#     )

#     benchmark(
#         model.predict,
#         images[0],
#         **inference_configuration,
#     )


# something like this could work
def test_yolov8_cf(benchmark, dataset_reference):
    images, image_sizes = dataset_reference
    inference_configuration = {}

    model = codeflash_yolov8_keypoints_detection.YOLOv8KeypointsDetection(
        model_id="yolov8n-seg-640",
        api_key=None,
    )
    img = images[0]
    predictions = model.prepare(
        img
    )  # this calls the onix session beforehand, so we don't have to worry about it

    # since onix session is setup, now we can actually benchmark the predict method
    benchmark(
        model.predict,
        img,
        predictions,
        **inference_configuration,
    )
