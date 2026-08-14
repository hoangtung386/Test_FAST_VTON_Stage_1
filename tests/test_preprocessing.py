import numpy as np
from PIL import Image

from fast_vton.preprocessing import AgnosticBuilder, build_mask_latent


def _grey_figure(size=(64, 80)):
    """A simple image where the top third is plain clothing-coloured."""
    image = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    image[: size[1] // 3] = (200, 30, 30)  # pretend "clothing"
    image[size[1] // 3 :] = (20, 20, 20)  # background / skin
    return Image.fromarray(image)


def test_build_agnostic_paints_grey(monkeypatch):
    person = _grey_figure()

    # Stub the segmentation model so the test needs no network / weights.
    class _FakeLogits:
        def argmax(self, dim=1):
            seg = np.zeros((1, 80, 64), dtype=np.int64)
            seg[0, : 80 // 3] = 4  # class 4 == upper_clothes over the top third
            return __import__("torch").from_numpy(seg)

    class _FakeModel:
        def __call__(self, **kwargs):
            return type("O", (), {"logits": _FakeLogits()})()

    class _FakeProc:
        def __call__(self, images=None, return_tensors=None):
            return type("O", (), {})()

    builder = AgnosticBuilder("dummy/repo", (4, 5, 6, 7), "cpu")
    builder._processor = _FakeProc()
    builder._model = _FakeModel()

    out = builder(person)
    arr = np.array(out)
    # Clothing region should now be the grey placeholder.
    assert tuple(arr[2, 2]) == (128, 128, 128)
    # Non-clothing region is untouched.
    assert tuple(arr[60, 30]) == (20, 20, 20)


def test_build_mask_latent_shape_and_dtype():
    person = _grey_figure()
    agnostic = _grey_figure()
    # Force a diff so the mask is non-empty.
    agnostic = np.array(agnostic)
    agnostic[:10] = 0
    agnostic = Image.fromarray(agnostic)

    mask = build_mask_latent(
        person,
        agnostic,
        (64, 80),
        diff_threshold=12,
        morph_kernel=9,
        device="cpu",
        dtype=np.float32,
    )
    assert mask.shape == (1, 1, 80 // 8, 64 // 8)
    assert mask.dtype == np.float32
