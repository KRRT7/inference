from typing import Tuple

import numpy as np

from inference.core.models.instance_segmentation_base import (
    InstanceSegmentationBaseOnnxRoboflowInferenceModel,
)
from inference.core.utils.onnx import run_session_via_iobinding


class YOLOv8InstanceSegmentation(InstanceSegmentationBaseOnnxRoboflowInferenceModel):
    """YOLOv8 Instance Segmentation ONNX Inference Model.

    This class is responsible for performing instance segmentation using the YOLOv8 model
    with ONNX runtime.

    Attributes:
        weights_file (str): Path to the ONNX weights file.

    Methods:
        predict: Performs inference on the given image using the ONNX session.
    """

    @property
    def weights_file(self) -> str:
        """Gets the weights file for the YOLOv8 model.

        Returns:
            str: Path to the ONNX weights file.
        """
        return "weights.onnx"

    def predict(self, img_in: np.ndarray, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """Performs inference on the given image using the ONNX session.

        Args:
            img_in (np.ndarray): Input image as a NumPy array.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Tuple containing two NumPy arrays representing the predictions and protos. The predictions include boxes, confidence scores, class confidence scores, and masks.
        """
        predictions, protos = run_session_via_iobinding(
            self.onnx_session, self.input_name, img_in
        )

        # predictions has shape (batch, num_items, num_outputs).
        # Transpose only if needed (batch, num_items, features).
        pred = predictions.transpose(0, 2, 1)
        boxes = pred[:, :, :4]
        class_confs = pred[:, :, 4:-32]
        # Use keepdims for expand_dims-like effect with less copying
        confs = np.max(class_confs, axis=2, keepdims=True)
        masks = pred[:, :, -32:]
        # Faster memory layout for concatenate
        out = np.concatenate((boxes, confs, class_confs, masks), axis=2)
        return out, protos
