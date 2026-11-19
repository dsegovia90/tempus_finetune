"""Export a trained classifier to ONNX for ONNX Runtime and the browser page.

Writes runs/<name>/onnx/: `fp32.onnx`; `fp16.onnx`, the same graph with its weights stored in
16 bits and cast back to 32 on load (half the download, same arithmetic); and `int8.onnx`,
dynamically quantized (8-bit weights, including the embedding table). Inputs are `input_ids`
and `attention_mask` (batch, tokens), the output is `logits` (batch, tokens, classes).
Needs the `classifier` dependency group.
"""

from pathlib import Path

OPSET = 17


def export(model_dir: Path) -> list[Path]:
    """Write the three ONNX files next to `model_dir` and return their paths."""
    import onnx
    import torch
    from onnxruntime.quantization import QuantType, quantize_dynamic
    from transformers import AutoModelForTokenClassification

    out = model_dir.parent / "onnx"
    out.mkdir(exist_ok=True)
    model = AutoModelForTokenClassification.from_pretrained(model_dir).eval()

    class Logits(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = model

        def forward(self, input_ids, attention_mask):
            return self.model(input_ids=input_ids, attention_mask=attention_mask).logits

    fp32, fp16, int8 = out / "fp32.onnx", out / "fp16.onnx", out / "int8.onnx"
    sample = torch.ones((1, 8), dtype=torch.long)
    axes = {0: "batch", 1: "tokens"}
    torch.onnx.export(
        Logits(), (sample, sample), fp32, input_names=["input_ids", "attention_mask"],
        output_names=["logits"], dynamic_axes={"input_ids": axes, "attention_mask": axes, "logits": axes},
        opset_version=OPSET, dynamo=False,
    )
    onnx.save(_half_weights(onnx.load(fp32)), fp16)
    quantize_dynamic(fp32, int8, weight_type=QuantType.QUInt8)
    return [fp32, fp16, int8]


def _half_weights(graph_model):
    """Store every float weight in 16 bits, with a Cast back to 32 bits where it is used."""
    import numpy as np
    from onnx import TensorProto, helper, numpy_helper

    graph = graph_model.graph
    casts = []
    for index, weight in enumerate(graph.initializer):
        if weight.data_type != TensorProto.FLOAT:
            continue
        name = weight.name
        half = numpy_helper.from_array(numpy_helper.to_array(weight).astype(np.float16), f"{name}.fp16")
        graph.initializer[index].CopyFrom(half)
        casts.append(helper.make_node("Cast", [half.name], [name], to=TensorProto.FLOAT))
    nodes = [*casts, *graph.node]  # casts first, so the graph stays topologically sorted
    del graph.node[:]
    graph.node.extend(nodes)
    return graph_model
