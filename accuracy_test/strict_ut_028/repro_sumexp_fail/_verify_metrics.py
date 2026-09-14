"""Verify repro_sumexp.py components against stored data and sources.

(1) metric formulas: recompute the 3 FAIL ratios from 08281443 .pt data,
    must match ratio_20260828_144618.md numbers.
(2) input generation: torch CPU randn with seed 42 is deterministic across
    platforms - verified by comparing rebuild() digests is impossible without
    torch locally, so instead check ref() logic via source-line diff (done
    separately by grep).
"""
import io
import pickle
import zipfile
import numpy as np

DTYPES = {
    "FloatStorage": np.float32, "DoubleStorage": np.float64,
    "LongStorage": np.int64, "IntStorage": np.int32,
    "HalfStorage": np.float16, "BoolStorage": np.bool_, "ByteStorage": np.uint8,
}


def load_pt(path):
    z = zipfile.ZipFile(path)
    prefix = z.namelist()[0].split("/")[0]

    class Stor:
        def __init__(self, dtype, data):
            self.dtype, self.data = dtype, data

    class U(pickle.Unpickler):
        def find_class(self, module, name):
            if module == "torch" and name.endswith("Storage"):
                return type(name, (), {"__name__": name})
            if module == "torch._utils" and name.startswith("_rebuild"):
                def rebuild(storage, offset, size, stride, *a, **k):
                    arr = np.frombuffer(storage.data, dtype=storage.dtype)
                    strides = tuple(int(s) * np.dtype(storage.dtype).itemsize for s in stride)
                    return np.lib.stride_tricks.as_strided(
                        arr[int(offset):], shape=tuple(int(s) for s in size),
                        strides=strides).copy()
                return rebuild
            return super().find_class(module, name)

        def persistent_load(self, pid):
            _, cls, key, _loc, numel = pid
            dtype = DTYPES[cls.__name__]
            raw = z.read(f"{prefix}/data/{key}")
            return Stor(dtype, raw)

    return U(io.BytesIO(z.read(f"{prefix}/data.pkl"))).load()


REL_EPS = 1e-7
FLOOR = 2.0 ** -30
root = r"c:\Users\x30084275\Desktop\git\024\双标杆比对结果\260828\08281443"
EXPECT = {  # from ratio_20260828_144618.md
    ("deepseek_2l_129280v_3spec", "target_local_sumexp"): (4.91, 3.56, 3.55),
    ("deepseek_2l_129280v_3spec", "draft_local_sumexp"): (1.41, 2.1, 2.08),
    ("multi_4l_16384v_2spec", "target_local_sumexp"): (8.11, 3.65, 2.62),
    # PASS rows to confirm formula generality
    ("multi_4l_16384v_2spec", "draft_local_sumexp"): (1.41, 1.82, 1.97),
    ("small_2l_1024v_1spec", "target_local_sumexp"): (1, 1, 1),
}
CID = {"deepseek_2l_129280v_3spec": "28aa8a942a3b",
       "multi_4l_16384v_2spec": "2e6949c0d5e0",
       "small_2l_1024v_1spec": "b655998afe21"}


def metrics(a, g):
    diff = np.abs(a - g)
    rel = diff / (np.abs(g) + REL_EPS)
    return rel.max(), rel.mean(), float(np.sqrt((diff ** 2).mean()))


ok = True
for (case, out), exp in EXPECT.items():
    d = {s: load_pt(f"{root}\\{s}\\cases\\compute_local_logits_stats\\{CID[case]}.pt")
         for s in ("cpu", "gpu", "npu")}
    g = d["cpu"][out].astype(np.float64)
    gm = metrics(d["gpu"][out].astype(np.float64), g)
    nm = metrics(d["npu"][out].astype(np.float64), g)
    got = tuple(round(n / max(gv, FLOOR), 2) for n, gv in zip(nm, gm))
    match = all(abs(a - b) <= 0.02 for a, b in zip(got, exp))
    ok &= match
    print(f"{case:28s} {out:22s} got={got} expect={exp} {'OK' if match else 'MISMATCH'}")

print("\nmetric-formula verification:", "PASS" if ok else "FAIL")
