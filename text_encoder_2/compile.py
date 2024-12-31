import argparse
import copy
import os
import torch
import torch_neuronx
import neuronx_distributed
from diffusers import FluxPipeline
from transformers import T5EncoderModel

from model import (
    TracingT5TextEncoderWrapper,
    init_text_encoder_2,
)

COMPILER_WORKDIR_ROOT = os.path.dirname(__file__)


def build_text_encoder_2():
    """
    Build the T5 text encoder model, shard it, and wrap it in our tracing class.
    Returns the final model + any (optional) kwargs in a dict, i.e. (model, {}).
    This pattern is required by neuronx_distributed.trace.parallel_model_trace.
    """
    # Load pipeline and copy the text_encoder_2 from your flux pipeline
    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        torch_dtype=torch.bfloat16
    )
    # text_encoder_2 is presumably a T5EncoderModel or custom T5-based module
    text_encoder_2 = copy.deepcopy(pipe.text_encoder_2)
    del pipe

    # Optionally, if text_encoder_2 is not already a T5EncoderModel, you can
    # modify the approach. But presumably, it's a T5EncoderModel or T5-like module.
    # Shard the T5 model parameters across tp_degree ranks:
    init_text_encoder_2(text_encoder_2)

    # Wrap it in a tracing module
    wrapper = TracingT5TextEncoderWrapper(text_encoder_2)

    # Return the model plus optional kwargs (unused here)
    return wrapper, {}


def trace_text_encoder_2(max_sequence_length=512):
    """
    Trace the T5 text encoder with parallel model trace.
    """
    # Prepare sample inputs for T5
    # Typically T5 expects: (batch, seq_len) for input_ids
    input_ids = torch.zeros((1, max_sequence_length), dtype=torch.int64)
    attention_mask = torch.ones((1, max_sequence_length), dtype=torch.int64)

    sample_inputs = (input_ids, attention_mask)

    # Use neuronx_distributed.trace.parallel_model_trace
    model = neuronx_distributed.trace.parallel_model_trace(
        build_text_encoder_2,
        sample_inputs,
        tp_degree=2,  # we want 2 shards
        compiler_workdir=os.path.join(COMPILER_WORKDIR_ROOT, "compiler_workdir"),
        compiler_args=["--enable-fast-loading-neuron-binaries"],
    )

    # Load asynchronously
    torch_neuronx.async_load(model)

    # Save compiled artifact
    compiled_model_path = os.path.join(COMPILER_WORKDIR_ROOT, "compiled_model")
    if not os.path.exists(compiled_model_path):
        os.mkdir(compiled_model_path)

    # This saves multiple shards: tp_0.pt and tp_1.pt
    model_filename = os.path.join(compiled_model_path, "text_encoder_2")
    neuronx_distributed.trace.parallel_model_save(model, model_filename)

    # Cleanup
    del model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--max_sequence_length",
        type=int,
        default=512,
        help="maximum sequence length for the text embeddings"
    )
    args = parser.parse_args()
    trace_text_encoder_2(args.max_sequence_length)

