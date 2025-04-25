from inference.core.active_learning.core import (
    prepare_image_to_registration,
    collect_tags,
    is_prediction_registration_forbidden,
)

import itertools
from inference.core.active_learning.entities import (
    ActiveLearningConfiguration,
    ImageDimensions,
    Prediction,
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


@pytest.mark.parametrize(
    "desired_size",
    [
        ImageDimensions(100, 100),
        None,
    ],
)
@pytest.mark.parametrize("jpeg_compression_level", [50, 75, 90])
def test_prepare_image_to_registration(
    benchmark, dataset_reference, desired_size, jpeg_compression_level
):
    images, image_sizes = dataset_reference
    benchmark(
        prepare_image_to_registration,
        images[0],
        desired_size,
        jpeg_compression_level,
    )


@pytest.mark.parametrize("persist", [True, False])
@pytest.mark.parametrize("num_config_tags", [0, 5, 20])
@pytest.mark.parametrize("num_strategy_tags", [0, 5, 20])
@pytest.mark.parametrize("env_tags", [None, ["env_tag1", "env_tag2"]])
@pytest.mark.parametrize("strategy", ["strategy_a", "strategy_b"])
def test_collect_tags(
    benchmark, persist, num_config_tags, num_strategy_tags, env_tags, strategy
):
    strategies = ["strategy_a", "strategy_b"]
    config_tags = [f"config_tag_{i}" for i in range(num_config_tags)]
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


@pytest.mark.parametrize(
    "prediction",
    [
        {"predictions": [{"x": 37}], "top": "cat"},
        {"top": "cat"},
        {"predictions": []},
        {"is_stub": True},
        {},
    ],
)
@pytest.mark.parametrize("persist", [True, False])
@pytest.mark.parametrize("roboflow_image_id", ["some_id", None])
def test_is_prediction_registration_forbidden(
    benchmark, prediction, persist, roboflow_image_id
):
    benchmark(
        is_prediction_registration_forbidden,
        prediction=prediction,
        persist_predictions=persist,
        roboflow_image_id=roboflow_image_id,
    )
