#!/usr/bin/env python3
"""Build an 8-bit-weight copy of the Parakeet TDT encoder for CUDA.

Usage: quantize-parakeet.py OUT_DIR

The encoder's MatMul weights are stored as 8-bit blocks (ONNX Runtime
MatMulNBits). They are expanded back to fp32 on the GPU at run time, so the
math is still fp32: same transcripts as the fp32 model and a few ms slower on
short clips. It cuts VRAM from ~3.1 GB to ~1.2-1.6 GB. fp16 isn't an option:
on GPUs without tensor cores (GTX 16xx) fp16 MatMul is ~6x slower than fp32.

The source is the official fp32 model that onnx-asr already downloads. OUT_DIR
holds files named for onnx-asr's quantization="q8" lookup. It is written to a
temp dir and renamed into place, so an interrupted run leaves nothing behind.
"""
import os
import shutil
import sys
from pathlib import Path

REPO = "istupakov/parakeet-tdt-0.6b-v3-onnx"  # onnx-asr's default for nemo-parakeet-tdt-0.6b-v3
ENCODER = "encoder-model.onnx"
COPY = {"decoder_joint-model.onnx": "decoder_joint-model.q8.onnx", "config.json": "config.json", "vocab.txt": "vocab.txt"}


def _load_encoder(src: Path):
    """onnx.load, but reading the external weights ourselves.

    The HF cache stores files as symlinks into blobs/, and onnx refuses to
    read external data through a symlink (or a hardlink), so load the graph
    alone and fill each tensor from the resolved .data file.
    """
    import onnx

    model = onnx.load(str(src / ENCODER), load_external_data=False)
    tensors = list(model.graph.initializer) + [
        a.t for n in model.graph.node if n.op_type == "Constant" for a in n.attribute if a.HasField("t")]
    with open((src / (ENCODER + ".data")).resolve(), "rb") as f:
        for t in tensors:
            if t.data_location != onnx.TensorProto.EXTERNAL:
                continue
            info = {e.key: e.value for e in t.external_data}
            if info["location"] != ENCODER + ".data":
                raise RuntimeError(f"unexpected external data file {info['location']}")
            f.seek(int(info.get("offset", 0)))
            t.raw_data = f.read(int(info["length"]))
            t.data_location = onnx.TensorProto.DEFAULT
            del t.external_data[:]
    return model


def _link_or_copy(src: Path, dst: Path) -> None:
    # OUT_DIR must outlive an HF cache cleanup, so no symlinks into it.
    src = src.resolve()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copyfile(src, dst)


def main(out: Path) -> None:
    import onnx
    from huggingface_hub import snapshot_download
    from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer

    src = Path(snapshot_download(REPO, allow_patterns=[ENCODER, ENCODER + ".data", *COPY]))
    tmp = out.with_name(f"{out.name}.tmp-{os.getpid()}")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    try:
        model = _load_encoder(src)
        q = MatMulNBitsQuantizer(model, bits=8, block_size=64, is_symmetric=True, op_types_to_quantize=("MatMul",))
        q.process()
        onnx.save(q.model.model, str(tmp / "encoder-model.q8.onnx"),
                  save_as_external_data=True, location="encoder-model.q8.onnx.data")
        for f, name in COPY.items():
            _link_or_copy(src / f, tmp / name)
        shutil.rmtree(out, ignore_errors=True)
        tmp.rename(out)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"Wrote {out}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(Path(sys.argv[1]))
