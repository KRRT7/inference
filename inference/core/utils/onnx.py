from typing import TYPE_CHECKING, List, Union

import numpy as np
import onnxruntime as ort
import torch

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
    """
    Runs the ONNX session – will use IO binding for CUDA sessions and torch tensor input.

    Returns:
        List of np.ndarray outputs from the session.
    """
    # Fast path for numpy or list input or non-CUDA session
    if (
        isinstance(input_data, (np.ndarray, list))
        or "CUDAExecutionProvider" not in session.get_providers()
    ):
        if not isinstance(input_data, np.ndarray):
            input_data = np.asarray(input_data)
        # Covers both numpy input and the no-CUDA path.
        return session.run(None, {input_name: input_data})

    # Fetch the dtype once for all outputs.
    outputs = session.get_outputs()
    output_dtypes = []
    output_shapes = []
    for output in outputs:
        # ONNX output.type always has a string like 'tensor(float16)' or 'tensor(float)'
        if "16" in output.type:
            output_dtypes.append(np.float16)
        else:
            output_dtypes.append(np.float32)
        output_shapes.append(tuple(output.shape))

    binding = session.io_binding()
    predictions = []
    # Prepare fast output arrays and bindings
    for output, dtype, shape in zip(outputs, output_dtypes, output_shapes):
        arr = np.empty(shape, dtype=dtype)
        binding.bind_output(
            name=output.name,
            device_type="cpu",
            device_id=0,
            element_type=dtype,
            shape=shape,
            buffer_ptr=arr.ctypes.data,
        )
        predictions.append(arr)

    # Ensure contiguous tensor and direct binding to input
    input_tensor = input_data.contiguous()
    device = input_tensor.device
    dtype = input_tensor.dtype
    binding.bind_input(
        name=input_name,
        device_type=device.type,
        device_id=(device.index if device.index is not None else 0),
        element_type=input_tensor.numpy().dtype,
        shape=input_tensor.shape,
        buffer_ptr=input_tensor.data_ptr(),
    )

    binding.synchronize_inputs()
    session.run_with_iobinding(binding)

    # Always cast outputs to float32 per comment/spec
    return [arr.astype(np.float32, copy=False) for arr in predictions]
