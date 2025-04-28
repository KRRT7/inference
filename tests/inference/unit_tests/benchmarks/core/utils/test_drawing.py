from inference_cli.lib.benchmark.dataset import load_dataset_images
from inference.core.utils.drawing import (
    create_tiles,
    _aggregate_images_shape,
    _establish_grid_size,
    _negotiate_grid_size,
    _generate_tiles,  # noqa: F401
    _merge_tiles_elements,  # noqa: F401
    _generate_color_image,
)
import pytest
import numpy as np


@pytest.fixture
def dataset_reference() -> tuple[list[np.ndarray], set[tuple[int, int]]]:
    dataset_images = load_dataset_images(
        dataset_reference="coco",
    )
    return dataset_images, {i.shape[:2] for i in dataset_images}


def test_create_tiles(benchmark, dataset_reference):
    images, image_sizes = dataset_reference
    benchmark(
        create_tiles,
        images,
        grid_size=(2, 5),
    )


@pytest.mark.parametrize("mode", ["min", "max", "avg"])
def test_calculate_aggregated_images_shape(benchmark, dataset_reference, mode):
    images, image_sizes = dataset_reference
    benchmark(
        _aggregate_images_shape,
        images,
        mode,
    )


def test_negotiate_grid_size(benchmark, dataset_reference):
    images, image_sizes = dataset_reference
    large_images = images * 10
    benchmark(
        _negotiate_grid_size,
        large_images,
    )


# the below two benchmarks are commented out because they rely on implementation details, i'll let roboflow decide how to handle them
# @pytest.mark.parametrize("grid_size", [(2, 5), (3, 4)])
# @pytest.mark.parametrize("tile_size", [(100, 100), (200, 200)])
# @pytest.mark.parametrize("tile_padding_color", [(0, 0, 0), (255, 255, 255)])
# @pytest.mark.parametrize("tile_margin", [0, 5])
# @pytest.mark.parametrize("tile_margin_color", [(0, 0, 0), (255, 255, 255)])
# def test_generate_tiles(
#     benchmark,
#     dataset_reference,
#     grid_size,
#     tile_size,
#     tile_padding_color,
#     tile_margin,
#     tile_margin_color,
# ):
#     images, image_sizes = dataset_reference
#     benchmark(
#         _generate_tiles,
#         images,
#         grid_size,
#         tile_size,
#         tile_padding_color,
#         tile_margin,
#         tile_margin_color,
#     )


# def test_merge_tiles_elements(benchmark):
#     tile_size = (100, 100)
#     grid_size = (2, 5)
#     tile_margin = 5
#     tile_margin_color = (0, 0, 0)
#     single_tile_size = (tile_size[0] + tile_margin, tile_size[1] + tile_margin)
#     tiles_elements = [
#         [np.zeros((*tile_size, 3), dtype=np.uint8) for _ in range(grid_size[1])]
#         for _ in range(grid_size[0])
#     ]

#     benchmark(
#         _merge_tiles_elements,
#         tiles_elements,
#         grid_size,
#         single_tile_size,
#         tile_margin,
#         tile_margin_color,
#     )


def test_generate_color_image(benchmark):
    shape = (100, 100)
    color = (0, 0, 0)
    benchmark(
        _generate_color_image,
        shape,
        color,
    )
