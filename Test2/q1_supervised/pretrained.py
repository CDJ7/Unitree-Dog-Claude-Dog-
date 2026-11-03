"""ImageNet-pretrained ResNet18 weights, with a fallback for networks that block PyTorch's servers.

`load_imagenet_resnet18()` returns a torchvision `resnet18` with ImageNet-1k weights:
1. normally from torchvision (`ResNet18_Weights.IMAGENET1K_V1`, download.pytorch.org): what
   Colab uses;
2. if that host is unreachable (the cloud container this project was built in blocks it), from the
   ONNX Model Zoo's ImageNet ResNet18 v1 (`resnet18-v1-7.onnx`, GitHub LFS, sha256 checked).
   The ONNX graph has the same architecture as torchvision's resnet18 (7x7 stem, BasicBlocks with
   the stride on the first 3x3 conv, 1x1 downsample, BN eps 1e-5), so its tensors are copied
   one-to-one (convs matched in graph order by shape, each BN by the conv it follows). The
   conversion is checked numerically against onnxruntime on random inputs before it is cached.
Both are ImageNet-1k classifiers expecting the standard mean/std normalisation; they are different
training runs, so their numbers differ slightly. Which source was used is logged with the results.
"""
from __future__ import annotations

import hashlib
import os
import urllib.request
from pathlib import Path

import numpy as np
import torch
import torchvision

ONNX_URL = ("https://media.githubusercontent.com/media/onnx/models/main/validated/vision/"
            "classification/resnet/model/resnet18-v1-7.onnx")
ONNX_SHA256 = "4e8f8653e7a2222b3904cc3fe8e304cd8b339ce1d05fd24688162f86fb6df52c"
CACHE = Path(os.environ.get("TEST2_CACHE", Path.home() / ".cache" / "test2"))


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def convert_onnx_resnet18(onnx_path):
    """torchvision resnet18 state_dict from the ONNX Model Zoo ResNet18 v1 graph."""
    import onnx
    from onnx import numpy_helper
    g = onnx.load(str(onnx_path)).graph
    init = {t.name: numpy_helper.to_array(t) for t in g.initializer}
    convs = [n for n in g.node if n.op_type == "Conv"]
    bn_of = {n.input[0]: n for n in g.node if n.op_type == "BatchNormalization"}
    gemm = next(n for n in g.node if n.op_type == "Gemm")

    model = torchvision.models.resnet18(weights=None)
    sd = model.state_dict()
    torch_convs = [k[:-len(".weight")] for k in sd if k.endswith(".weight") and sd[k].ndim == 4]
    used = set()
    out = {}
    for name in torch_convs:
        shape = tuple(sd[name + ".weight"].shape)
        node = next(n for n in convs if n.name not in used and n.output[0] not in used
                    and init[n.input[1]].shape == shape)
        used.add(node.output[0])
        out[name + ".weight"] = init[node.input[1]]
        bn = bn_of[node.output[0]]
        bname = name.replace("conv", "bn") if "downsample" not in name else name[:-1] + "1"
        for suffix, idx in (("weight", 1), ("bias", 2), ("running_mean", 3), ("running_var", 4)):
            out[f"{bname}.{suffix}"] = init[bn.input[idx]]
    out["fc.weight"] = init[gemm.input[1]]
    out["fc.bias"] = init[gemm.input[2]]
    state = {k: torch.from_numpy(np.ascontiguousarray(v)).float() for k, v in out.items()}
    for k in sd:
        if k.endswith("num_batches_tracked"):
            state[k] = sd[k]
    missing = set(sd) - set(state)
    if missing:
        raise RuntimeError(f"conversion missed {sorted(missing)[:5]}")
    return state


def _verify(model, onnx_path, tol=1e-3):
    import onnxruntime as ort
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    x = np.random.default_rng(0).normal(size=(2, 3, 224, 224)).astype(np.float32)
    ref = sess.run(None, {sess.get_inputs()[0].name: x})[0]
    model.eval()
    with torch.no_grad():
        got = model(torch.from_numpy(x)).numpy()
    err = float(np.abs(got - ref).max() / (np.abs(ref).max() + 1e-9))
    if err > tol:
        raise RuntimeError(f"ONNX -> torch conversion mismatch: relative error {err:.2e}")
    return err


def load_imagenet_resnet18(verbose=True):
    """(model, source string)."""
    try:
        w = torchvision.models.ResNet18_Weights.IMAGENET1K_V1
        m = torchvision.models.resnet18(weights=w)
        return m, "torchvision ResNet18_Weights.IMAGENET1K_V1"
    except Exception as e:  # noqa: BLE001
        if verbose:
            print(f"[pretrained] torchvision weights unavailable ({type(e).__name__}); using ONNX Model Zoo copy")
    CACHE.mkdir(parents=True, exist_ok=True)
    pth = CACHE / "resnet18_imagenet_from_onnx_v1-7.pth"
    onnx_path = CACHE / "resnet18-v1-7.onnx"
    m = torchvision.models.resnet18(weights=None)
    if not pth.exists():
        if not onnx_path.exists() or _sha256(onnx_path) != ONNX_SHA256:
            urllib.request.urlretrieve(ONNX_URL, onnx_path)
        if _sha256(onnx_path) != ONNX_SHA256:
            raise RuntimeError("downloaded ONNX ResNet18 has the wrong checksum")
        m.load_state_dict(convert_onnx_resnet18(onnx_path))
        err = _verify(m, onnx_path)
        if verbose:
            print(f"[pretrained] converted ONNX ResNet18 v1; max relative error vs onnxruntime {err:.1e}")
        torch.save(m.state_dict(), pth)
    else:
        m.load_state_dict(torch.load(pth, map_location="cpu"))
    return m, "ONNX Model Zoo resnet18-v1-7 (ImageNet-1k), converted to torchvision resnet18"
