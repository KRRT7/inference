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




def test_prepare_image_to_registration(benchmark, dataset_reference):
    images, image_sizes = dataset_reference
    desired_sizes = [
        ImageDimensions(100, 100),
        None,
    ]
    jpeg_compression_levels = [50, 75, 90]
    param_combinations = itertools.product(
        desired_sizes,
        jpeg_compression_levels,
    )
    for desired_size, jpeg_compression_level in param_combinations:
        benchmark(
            prepare_image_to_registration,
            images[0],
            desired_size,
            jpeg_compression_level,
            benchmark_name_suffix=(
                f"desired_size_{desired_size}_"
                f"jpeg_compression_level_{jpeg_compression_level}"
            ),
        )


def test_collect_tags(benchmark):
    persist_options = [True, False]
    num_config_tags_options = [0, 5, 20]
    num_strategy_tags_options = [0, 5, 20]
    env_tags_options = [None, ["env_tag1", "env_tag2"]]
    strategies = ["strategy_a", "strategy_b"]

    param_combinations = itertools.product(
        persist_options,
        num_config_tags_options,
        num_strategy_tags_options,
        env_tags_options,
        strategies,
    )

    for (
        persist,
        num_config_tags,
        num_strategy_tags,
        env_tags,
        strategy,
    ) in param_combinations:
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
                benchmark_name_suffix=(
                    f"persist_{persist}_"
                    f"config_tags_{num_config_tags}_"
                    f"strategy_tags_{num_strategy_tags}_"
                    f"env_tags_{env_tags is not None}_"
                    f"strategy_{strategy}"
                ),
            )


def test_is_prediction_registration_forbidden(benchmark):
    predictions: list[Prediction] = [
        {"predictions": [{"x": 37}], "top": "cat"},
        {"top": "cat"},
        {"predictions": []},
        {"is_stub": True},
        {},
    ]
    persist_options = [True, False]
    roboflow_image_id_options = ["some_id", None]

    param_combinations = itertools.product(
        predictions,
        persist_options,
        roboflow_image_id_options,
    )

    for i, (prediction, persist, roboflow_image_id) in enumerate(param_combinations):
        benchmark(
            is_prediction_registration_forbidden,
            prediction=prediction,
            persist_predictions=persist,
            roboflow_image_id=roboflow_image_id,
            benchmark_name_suffix=(
                f"prediction_case_{i}_"
                f"persist_{persist}_"
                f"has_image_id_{roboflow_image_id is not None}"
            ),
        )
