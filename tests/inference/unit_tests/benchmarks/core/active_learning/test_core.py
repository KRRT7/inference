from inference.core.active_learning.core import (
    prepare_image_to_registration,
    collect_tags,
    is_prediction_registration_forbidden,
)

from inference.core.active_learning.entities import (
    ActiveLearningConfiguration,
    ImageDimensions,
)
from inference.core.active_learning.entities import (
    BatchReCreationInterval,
)
from unittest import mock
import inference.core.active_learning.core as core


from inference_cli.lib.benchmark.dataset import load_dataset_images
import pytest
import numpy as np


@pytest.fixture
def dataset_reference() -> tuple[list[np.ndarray], set[tuple[int, int]]]:
    dataset_images = load_dataset_images(
        dataset_reference="coco",
    )
    return dataset_images, {i.shape[:2] for i in dataset_images}


def test_prepare_image_to_registration_standalone(benchmark, dataset_reference):
    images, image_sizes = dataset_reference
    desired_size = ImageDimensions(100, 100)
    jpeg_compression_level = 75
    benchmark(
        prepare_image_to_registration,
        images[0],
        desired_size,
        jpeg_compression_level,
    )


def test_collect_tags_single_case(benchmark):
    persist = True
    num_config_tags = 5
    num_strategy_tags = 5
    env_tags = ["env_tag1", "env_tag2"]
    strategy = "strategy_a"

    config_tags = [f"config_tag_{i}" for i in range(num_config_tags)]
    strategies = ["strategy_a", "strategy_b"]
    strategy_tags_dict = {
        s: [f"{s}_tag_{i}" for i in range(num_strategy_tags)] for s in strategies
    }
    if strategy not in strategy_tags_dict:
        strategy_tags_dict[strategy] = [
            f"{strategy}_tag_{i}" for i in range(num_strategy_tags)
        ]

    configuration = ActiveLearningConfiguration(
        max_image_size=None,
        jpeg_compression_level=95,
        persist_predictions=persist,
        sampling_methods=[],
        batches_name_prefix="al_batch",
        batch_recreation_interval=BatchReCreationInterval.DAILY,
        max_batch_images=None,
        workspace_id="my_workspace",
        dataset_id="my-dataset",
        model_id="my-dataset/3",
        strategies_limits={s: [] for s in strategies},
        tags=config_tags,
        strategies_tags=strategy_tags_dict,
    )

    with mock.patch.object(core, "ACTIVE_LEARNING_TAGS", env_tags):
        benchmark(
            collect_tags,
            configuration=configuration,
            sampling_strategy=strategy,
        )


def test_is_prediction_registration_forbidden_single(benchmark):
    benchmark(
        is_prediction_registration_forbidden,
        prediction={"predictions": [{"x": 37}], "top": "cat"},
        persist_predictions=True,
        roboflow_image_id="some_id",
    )
