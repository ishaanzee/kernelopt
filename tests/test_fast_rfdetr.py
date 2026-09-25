"""pytest tests/ -q   (needs cv2 for the preprocessing reference and the Ballform reference npz)"""
import threading

import cv2
import numpy as np
import pytest

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import kernelopt_paths  # noqa: E402
from fast_rfdetr import FastBasketballDetector  # noqa: E402
from fast_rfdetr.evaluation import accuracy_report, load_reference  # noqa: E402
from fast_rfdetr.resize_tables import resize_reference  # noqa: E402

MODEL, NPZ = kernelopt_paths.model_path(required=False), kernelopt_paths.reference_npz(required=False)
if MODEL is None or NPZ is None:
    pytest.skip("set KERNELOPT_MODEL / KERNELOPT_REFERENCE_NPZ or create paths.local.json", allow_module_level=True)
MODEL = str(MODEL)
MEAN = np.asarray([.485, .456, .406], dtype=np.float32)
STD = np.asarray([.229, .224, .225], dtype=np.float32)


def ballform_preprocess(frame):
    rgb = cv2.cvtColor(cv2.resize(frame, (640, 640)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.
    return np.ascontiguousarray(((rgb - MEAN) / STD).transpose(2, 0, 1)[None])


@pytest.fixture(scope="module")
def det(tmp_path_factory):
    return FastBasketballDetector(MODEL, warmup_batch_sizes=(1, 3), cache_dir=tmp_path_factory.mktemp("cache"))


@pytest.fixture(scope="module")
def ref():
    return load_reference(NPZ)


@pytest.mark.parametrize("hw", [(1080, 1920), (1080, 1152), (720, 1280), (500, 333), (2160, 3840), (641, 1279)])
def test_resize_model_bit_exact(hw):
    img = np.random.default_rng(hw[0] * hw[1]).integers(0, 256, (*hw, 3), dtype=np.uint8)
    assert np.array_equal(resize_reference(img), cv2.resize(img, (640, 640)))


def test_gpu_preprocess_bitwise_identical(det, ref):
    for _, img, _, _ in ref:
        gpu = np.array(det._preprocess(img))
        assert np.array_equal(gpu.view(np.uint32), ballform_preprocess(img)[0].transpose(1, 2, 0).view(np.uint32))


@pytest.mark.parametrize("batch", [1, 3])
def test_accuracy_bar(det, batch):
    rep = accuracy_report(det.raw_batch, NPZ, batch_size=batch, verbose=False)
    assert rep["passed"], rep["misses"]
    assert rep["max_logit_diff"] < 0.01 and rep["max_box_diff"] < 0.001


def test_batch_matches_single(det, ref):
    imgs = [r[1] for r in ref[:3]]
    bb, bl = det.raw_batch(imgs)
    for k, im in enumerate(imgs):
        sb, sl = det.raw(im)
        assert np.abs(bb[k] - sb[0]).max() < 1e-3 and np.abs(bl[k] - sl[0]).max() < 1e-2


def test_two_instances_two_threads(det, ref, tmp_path):
    other = FastBasketballDetector(MODEL, cache_dir=tmp_path)
    imgs = [r[1] for r in ref]
    expected = [det.raw(im) for im in imgs]
    errors = []

    def work(d, off):
        for k in range(30):
            i = (k + off) % len(imgs)
            b, lg = d.raw(imgs[i])
            if not (np.array_equal(b, expected[i][0]) and np.array_equal(lg, expected[i][1])):
                errors.append(i)

    th = [threading.Thread(target=work, args=(d, o)) for d, o in ((det, 0), (other, 5))]
    [t.start() for t in th]
    [t.join() for t in th]
    assert not errors
