import numpy as np
import pytest

from ivp.workspace import comparison_wipe, run_details, source_name


def test_wipe_alignment_and_endpoints():
    original = np.zeros((4, 8, 3), dtype=np.uint8)
    processed = np.full_like(original, 100)
    assert np.array_equal(comparison_wipe(original, processed, 0), processed)
    assert np.array_equal(comparison_wipe(original, processed, 100), original)
    wipe = comparison_wipe(original, processed, 50)
    assert (wipe[:, :4] == 0).all()
    assert (wipe[:, 4] == 255).all()
    assert (wipe[:, 5:] == 100).all()
    assert (processed == 100).all()
    with pytest.raises(ValueError):
        comparison_wipe(original, processed[:2], 50)


def test_history_tolerates_partial_reports(tmp_path):
    (tmp_path / "report.json").write_text("{broken")
    (tmp_path / "request.json").write_text('{"source": "H:/uploads/street.jpg"}')
    assert source_name(run_details(tmp_path)) == "street.jpg"
    (tmp_path / "request.json").write_text("[]")
    assert run_details(tmp_path) == {}
