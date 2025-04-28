import numpy as np
import pytest
from inference.core.utils.preprocess import (
    prepare,  # noqa: F401
    static_crop_should_be_applied,
    take_static_crop,
    contrast_adjustments_should_be_applied,  # noqa: F401
    apply_contrast_adjustment,  # noqa: F401
    apply_contrast_stretching,
    apply_histogram_equalisation,
    apply_adaptive_equalisation,
    grayscale_conversion_should_be_applied,
    apply_grayscale_conversion,
    letterbox_image,
    downscale_image_keeping_aspect_ratio,
    resize_image_keeping_aspect_ratio,
    STATIC_CROP_KEY,
    GRAYSCALE_KEY,
)


@pytest.fixture()
def large_image():
    return np.random.randint(0, 256, size=(1080, 1920, 3), dtype=np.uint8)


@pytest.fixture()
def medium_image():
    return np.random.randint(0, 256, size=(720, 1280, 3), dtype=np.uint8)


@pytest.fixture()
def small_image():
    return np.random.randint(0, 256, size=(480, 640, 3), dtype=np.uint8)


def test_prepare(benchmark): ...  # TODO: Implement benchmarks


@pytest.mark.parametrize(
    "preproc, disable_preproc_static_crop, expected",
    [
        ({STATIC_CROP_KEY: {"enabled": True}}, False, True),
        ({STATIC_CROP_KEY: {"enabled": False}}, False, False),
        ({}, False, False),
        ({STATIC_CROP_KEY: {"enabled": True}}, True, False),
    ],
)
def test_static_crop_should_be_applied(
    benchmark, preproc, disable_preproc_static_crop, expected
):
    benchmark(
        static_crop_should_be_applied,
        preprocessing_config=preproc,
        disable_preproc_static_crop=disable_preproc_static_crop,
    )


@pytest.mark.parametrize(
    "image_fixture, crop_parameters",
    [
        ("small_image", {"x_min": 10, "y_min": 10, "x_max": 90, "y_max": 90}),
        ("medium_image", {"x_min": 0, "y_min": 20, "x_max": 80, "y_max": 100}),
        ("large_image", {"x_min": 25, "y_min": 25, "x_max": 75, "y_max": 75}),
    ],
)
def test_take_static_crop(benchmark, image_fixture, crop_parameters, request):
    image = request.getfixturevalue(image_fixture)
    benchmark(take_static_crop, image=image.copy(), crop_parameters=crop_parameters)


def test_contrast_adjustments_should_be_applied(
    benchmark,
): ...  # TODO: Implement benchmarks


def test_apply_contrast_adjustment(benchmark): ...  # TODO: Implement benchmarks


@pytest.mark.parametrize(
    "image_fixture", ["small_image", "medium_image", "large_image"]
)
def test_apply_contrast_stretching(benchmark, image_fixture, request):
    image = request.getfixturevalue(image_fixture)
    benchmark(apply_contrast_stretching, image=image.copy())


@pytest.mark.parametrize(
    "image_fixture", ["small_image", "medium_image", "large_image"]
)
def test_apply_histogram_equalisation(benchmark, image_fixture, request):
    image = request.getfixturevalue(image_fixture)
    benchmark(apply_histogram_equalisation, image=image.copy())


@pytest.mark.parametrize(
    "image_fixture", ["small_image", "medium_image", "large_image"]
)
def test_apply_adaptive_equalisation(benchmark, image_fixture, request):
    image = request.getfixturevalue(image_fixture)
    benchmark(apply_adaptive_equalisation, image=image.copy())


@pytest.mark.parametrize(
    "preproc, disable_preproc_grayscale",
    [
        ({GRAYSCALE_KEY: {"enabled": True}}, False),
        ({GRAYSCALE_KEY: {"enabled": False}}, False),
        ({}, False),
        ({GRAYSCALE_KEY: {"enabled": True}}, True),
    ],
)
def test_grayscale_conversion_should_be_applied(
    benchmark, preproc, disable_preproc_grayscale
):
    benchmark(
        grayscale_conversion_should_be_applied,
        preprocessing_config=preproc,
        disable_preproc_grayscale=disable_preproc_grayscale,
    )


@pytest.mark.parametrize(
    "image_fixture", ["small_image", "medium_image", "large_image"]
)
def test_apply_grayscale_conversion(benchmark, image_fixture, request):
    image = request.getfixturevalue(image_fixture)
    benchmark(apply_grayscale_conversion, image=image.copy())


@pytest.mark.parametrize(
    "image_fixture, desired_shape",
    [
        ("small_image", (320, 320)),
        ("medium_image", (640, 640)),
        ("large_image", (416, 416)),
        ("small_image", (640, 480)),
        ("large_image", (128, 128)),
    ],
)
def test_letterbox_image(benchmark, image_fixture, desired_shape, request):
    image = request.getfixturevalue(image_fixture)
    benchmark(letterbox_image, image=image.copy(), desired_size=desired_shape)


@pytest.mark.parametrize(
    "image_fixture, desired_shape",
    [
        ("small_image", (320, 320)),
        ("medium_image", (640, 640)),
        ("large_image", (416, 416)),
        ("large_image", (128, 128)),
    ],
)
def test_downscale_image_keeping_aspect_ratio(
    benchmark, image_fixture, desired_shape, request
):
    image = request.getfixturevalue(image_fixture)
    if image.shape[0] > desired_shape[0] or image.shape[1] > desired_shape[1]:
        benchmark(
            downscale_image_keeping_aspect_ratio,
            image=image.copy(),
            desired_size=desired_shape,
        )
    else:
        pytest.skip("Image is not larger than desired shape for downscaling.")


@pytest.mark.parametrize(
    "image_fixture, desired_shape",
    [
        ("small_image", (320, 320)),
        ("medium_image", (640, 640)),
        ("large_image", (416, 416)),
        ("small_image", (1280, 720)),
        ("medium_image", (1920, 1080)),
    ],
)
def test_resize_image_keeping_aspect_ratio(
    benchmark, image_fixture, desired_shape, request
):
    image = request.getfixturevalue(image_fixture)
    benchmark(
        resize_image_keeping_aspect_ratio,
        image=image.copy(),
        desired_size=desired_shape,
    )
