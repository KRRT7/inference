from typing import TYPE_CHECKING, List, Union

import numpy as np
import onnxruntime as ort

if TYPE_CHECKING:
    import torch

ImageMetaType = Union[np.ndarray, "torch.Tensor"]


def get_onnxruntime_execution_providers(value: str) -> List[str]:
    """Extracts the ONNX runtime execution providers from the given string.

    The input string is expected to be a comma-separated list, possibly enclosed
    within square brackets and containing single quotes.

    Args:
        value (str): The string containing the list of ONNX runtime execution providers.

    Returns:
        List[str]: A list of strings representing each execution provider.
    """
    if len(value) == 0:
        return []
    value = value.replace("[", "").replace("]", "").replace("'", "").replace(" ", "")
    return value.split(",")


def run_session_via_iobinding(
    session: ort.InferenceSession, input_name: str, input_data: ImageMetaType
) -> List[np.ndarray]:
    # Fast path: input is CPU numpy array, just run the session.
    if isinstance(input_data, np.ndarray):
        return session.run(None, {input_name: input_data})

    # Check if CUDAExecutionProvider is present
    if "CUDAExecutionProvider" not in session.get_providers():
        # Move tensor to CPU and convert to numpy if it's a torch tensor
        input_data = input_data.cpu().numpy()
        return session.run(None, {input_name: input_data})

    # GPU iobinding path (very rare for np.ndarray input)
    binding = session.io_binding()
    dtype = None
    predictions = []
    for output in session.get_outputs():
        # Use correct dtype
        if dtype is None:
            dtype = np.float16 if "16" in output.type else np.float32
        prediction = np.empty(output.shape, dtype=dtype)
        binding.bind_output(
            name=output.name,
            device_type="cpu",
            device_id=0,
            element_type=dtype,
            shape=output.shape,
            buffer_ptr=prediction.ctypes.data,
        )
        predictions.append(prediction)

    # Only executed if input_data is torch.Tensor on CUDA
    input_data = input_data.contiguous()
    binding.bind_input(
        name=input_name,
        device_type=input_data.device.type,
        device_id=(
            input_data.device.index if input_data.device.index is not None else 0
        ),
        element_type=dtype,
        shape=input_data.shape,
        buffer_ptr=input_data.data_ptr(),
    )
    binding.synchronize_inputs()
    session.run_with_iobinding(binding)

    # Convert any float16 output to float32 for consistency
    return [
        pred.astype(np.float32) if pred.dtype != np.float32 else pred
        for pred in predictions
    ]
