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
    # Fast path for np.ndarray or list (typical inference entry)
    if isinstance(input_data, (np.ndarray, list)):
        return session.run(None, {input_name: input_data})
    # If we don't have CUDA we must use a CPU copy
    providers = session.get_providers()
    if "CUDAExecutionProvider" not in providers:
        return session.run(None, {input_name: input_data.cpu().numpy()})

    # CUDA I/O binding: direct memory access, less copying
    binding = session.io_binding()
    outputs = session.get_outputs()
    dtype = np.float16 if any("16" in o.type for o in outputs) else np.float32

    predictions = []
    # Use pre-allocated numpy output buffers for ONNX to write into, speeds up large results
    for output in outputs:
        buf = np.empty(output.shape, dtype=dtype)
        binding.bind_output(
            name=output.name,
            device_type="cpu",
            device_id=0,
            element_type=dtype,
            shape=output.shape,
            buffer_ptr=buf.ctypes.data,
        )
        predictions.append(buf)

    tensor = input_data.contiguous()
    device = tensor.device
    binding.bind_input(
        name=input_name,
        device_type=device.type,
        device_id=0 if device.index is None else device.index,
        element_type=dtype,
        shape=tensor.shape,
        buffer_ptr=tensor.data_ptr(),
    )
    binding.synchronize_inputs()
    session.run_with_iobinding(binding)
    # Always return as float32 arrays for downstream ease
    return [arr.astype(np.float32, copy=False) for arr in predictions]
