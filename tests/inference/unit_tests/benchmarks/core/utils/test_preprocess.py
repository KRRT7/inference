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


def test_static_crop_should_be_applied_enabled_true(benchmark):
    preproc = {STATIC_CROP_KEY: {"enabled": True}}
    disable_preproc_static_crop = False
    benchmark(
        static_crop_should_be_applied,
        preprocessing_config=preproc,
        disable_preproc_static_crop=disable_preproc_static_crop,
    )


def test_take_static_crop_large_image_benchmark(benchmark, large_image):
    crop_parameters = {"x_min": 25, "y_min": 25, "x_max": 75, "y_max": 75}
    benchmark(
        take_static_crop, image=large_image.copy(), crop_parameters=crop_parameters
    )


def test_contrast_adjustments_should_be_applied(
    benchmark,
): ...  # TODO: Implement benchmarks


def test_apply_contrast_adjustment(benchmark): ...  # TODO: Implement benchmarks


def test_apply_contrast_stretching_large_image_benchmark(benchmark, large_image):
    benchmark(apply_contrast_stretching, image=large_image.copy())


def test_apply_histogram_equalisation_large_image_benchmark(benchmark, large_image):
    benchmark(apply_histogram_equalisation, image=large_image.copy())


def test_apply_adaptive_equalisation_large_image_benchmark(benchmark, large_image):
    benchmark(apply_adaptive_equalisation, image=large_image.copy())


def test_grayscale_conversion_should_be_applied_enabled_true_benchmark(benchmark):
    preproc = {GRAYSCALE_KEY: {"enabled": True}}
    disable_preproc_grayscale = False
    benchmark(
        grayscale_conversion_should_be_applied,
        preprocessing_config=preproc,
        disable_preproc_grayscale=disable_preproc_grayscale,
    )


def test_apply_grayscale_conversion_large_image_benchmark(benchmark, large_image):
    benchmark(apply_grayscale_conversion, image=large_image.copy())


def test_letterbox_image_large_to_small_benchmark(benchmark, large_image):
    desired_shape = (128, 128)
    benchmark(letterbox_image, image=large_image.copy(), desired_size=desired_shape)


def test_downscale_image_keeping_aspect_ratio_large_to_small_benchmark(
    benchmark, large_image
):
    desired_shape = (128, 128)
    benchmark(
        downscale_image_keeping_aspect_ratio,
        image=large_image.copy(),
        desired_size=desired_shape,
    )


def test_resize_image_keeping_aspect_ratio_large_to_hd_benchmark(
    benchmark, large_image
):
    desired_shape = (1920, 1080)
    benchmark(
        resize_image_keeping_aspect_ratio,
        image=large_image.copy(),
        desired_size=desired_shape,
    )
