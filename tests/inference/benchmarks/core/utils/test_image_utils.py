import base64
import io
import numpy as np
import pytest
from PIL import Image
from _io import _IOBase
from inference.core.utils.image_utils import (
    attempt_loading_image_from_string,
    load_image_from_numpy_object,
    validate_numpy_image,
    load_image_from_encoded_bytes,
    convert_gray_image_to_bgr,  # noqa: F401
    xyxy_to_xywh,
    encode_image_to_jpeg_bytes,
)
from inference_cli.lib.benchmark.dataset import (
    load_dataset_images,
    download_image,
    PREDEFINED_DATASETS,
)
import requests


@pytest.fixture
def images() -> tuple[str, bytes, bytearray, _IOBase]:
    image_url = PREDEFINED_DATASETS["coco"][0]
    response = requests.get(image_url)
    response.raise_for_status()
    with open("test_image.jpg", "wb") as f:
        f.write(response.content)
    b64_string = base64.b64encode(response.content).decode("utf-8")
    image_bytes = base64.b64decode(b64_string)
    image_IO_buffer: _IOBase = io.BytesIO(response.content)
    numpy_image_str = ...
    return b64_string, image_bytes, image_IO_buffer, numpy_image_str


@pytest.fixture
def numpy_image() -> np.ndarray:
    image_url = PREDEFINED_DATASETS["coco"][0]
    response = requests.get(image_url)
    response.raise_for_status()
    with open("test_image.jpg", "wb") as f:
        f.write(response.content)
    image = Image.open("test_image.jpg")
    numpy_image = np.array(image)
    return numpy_image


def test_prepare_image_to_registration(benchmark, images):
    b64_string, image_bytes, image_IO_buffer, numpy_image_str = images
    # uses load_image_base64
    benchmark(
        attempt_loading_image_from_string,
        b64_string,
        benchmark_name_suffix="b64_string",
    )
    # uses load_image_from_encoded_bytes
    benchmark(
        attempt_loading_image_from_string,
        image_bytes,
        benchmark_name_suffix="image_bytes",
    )
    # uses load_image_from_buffer
    benchmark(
        attempt_loading_image_from_string,
        image_IO_buffer,
        benchmark_name_suffix="image_IO_buffer",
    )
    # todo: add numpy_image_str to the test


def test_load_image_from_numpy_object(benchmark, numpy_image):
    # Load the image using the benchmark function
    benchmark(
        load_image_from_numpy_object,
        numpy_image,
    )


def test_validate_numpy_image(benchmark, numpy_image):
    # Validate the image using the benchmark function
    benchmark(
        validate_numpy_image,
        numpy_image,
    )


def test_load_image_from_encoded_bytes(benchmark, images):
    b64_string, image_bytes, image_IO_buffer, numpy_image_str = images
    benchmark(
        load_image_from_encoded_bytes,
        image_bytes,
    )


def test_convert_gray_image_to_bgr(benchmark, numpy_image):
    # TODO: Add a test for the convert_gray_image_to_bgr function
    ...


def test_xyxy_to_xywh(benchmark):
    xyxy = [10, 20, 30, 40]

    benchmark(xyxy_to_xywh, xyxy)


def test_encode_image_to_jpeg_bytes(benchmark, numpy_image):
    jpeg_qualities = [50, 75, 90, 95]
    for jpeg_quality in jpeg_qualities:
        benchmark(
            encode_image_to_jpeg_bytes,
            numpy_image,
            jpeg_quality=jpeg_quality,
        )
