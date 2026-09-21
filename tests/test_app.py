import urllib.error
import urllib.request
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from ivp.downloads import FileServer
from ivp.media import write_json
from ivp.jobs import now


def test_app_renders_and_preset_changes():
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=20).run()
    assert not app.exception
    app.selectbox(key="preset").select("Low light").run()
    assert not app.exception
    assert app.slider(key="cfg_gamma").value == 1.5
    assert app.checkbox(key="cfg_clahe").value is True
    app.selectbox(key="preset").select("Original").run()
    assert app.slider(key="cfg_gamma").value == 1.0
    assert app.checkbox(key="cfg_clahe").value is False


def test_completed_ai_run_exposes_scene_observations(tmp_path):
    from PIL import Image

    run = tmp_path / "test_scene_run"
    scene = run / "scene_features" / "0001"
    scene.mkdir(parents=True)
    for name in ("road", "sidewalk", "white_road_regions"):
        Image.new("L", (20, 20)).save(scene / (name + ".png"))
    Image.new("RGB", (20, 20)).save(run / "processed.png")
    Image.new("RGB", (20, 20)).save(run / "detections.png")
    write_json(run / "status.json", {"state": "complete", "progress": 1., "updated_at": now(), "message": "Ready"})
    write_json(run / "report.json", {"source": "test-street.png", "ai": {"events_by_rule": {}, "assessment_counts": {"red_light/cannot_assess": 1}}})
    write_json(scene / "scene.json", {"sign_observations": [{"label": "No vehicles", "status": "recognised_candidate",
                                                           "proposal_source": "ivp_colour_shape", "reason": "Approach unresolved"}]})
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=20)
    app.session_state["video_run"] = str(run)
    app.run()
    assert not app.exception
    assert app.selectbox(key="scene_" + run.name).value == "0001"
    assert {"Detected objects", "Reported violations"} <= {tab.label for tab in app.tabs}
    assert any("Sign" in frame.value.columns and frame.value.iloc[0]["Sign"] == "No vehicles" for frame in app.dataframe)
    assert any("Assessment" in frame.value.columns and "cannot assess" in frame.value["Assessment"].values for frame in app.dataframe)


def test_unreadable_saved_run_does_not_hide_completed_results(tmp_path, monkeypatch):
    import ivp.jobs as jobs

    monkeypatch.setattr(jobs, "RUNS", tmp_path)
    unreadable = tmp_path / "unreadable"
    unreadable.mkdir()
    write_json(unreadable / "status.json", {"state": "running"})
    original = (unreadable / "status.json").read_bytes()
    complete = tmp_path / "complete"
    complete.mkdir()
    write_json(complete / "status.json", {"state": "complete", "progress": 1., "message": "Saved result ready"})
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=20)
    app.session_state["video_run"] = str(unreadable)
    app.run()
    assert not app.exception
    app.radio(key="page").set_value("Saved runs").run()
    assert not app.exception
    assert app.button(key="inspect_" + complete.name)
    assert any("Job status is unreadable" in element.value for element in app.markdown)
    assert (unreadable / "status.json").read_bytes() == original


def test_classroom_samples_and_saved_examples():
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=20).run()
    app.selectbox(key="demo_sample").select("Low-light street · simulated darkness").run()
    app.selectbox(key="preset").select("Low light").run()
    assert not app.exception
    app.radio(key="workspace_view").set_value("Quality + histograms").run()
    assert app.metric[0].value
    app.radio(key="page").set_value("Classroom demo").run()
    for example in ["Crossing segmentation", "Traffic-sign recognition", "Road users + video"]:
        app.selectbox(key="classroom_example").select(example).run()
        assert not app.exception


def test_workspace_navigation_reset_and_comparison():
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=20).run()
    assert app.radio(key="page").value == "Workspace"
    app.selectbox(key="demo_sample").select("Low-light street · simulated darkness").run()
    app.selectbox(key="preset").select("Low light").run()
    next(r for r in app.radio if r.label == "Comparison layout").set_value("Wipe comparison").run()
    assert not app.exception
    app.radio(key="page").set_value("Saved runs").run()
    assert not app.exception
    app.radio(key="page").set_value("Workspace").run()
    assert app.selectbox(key="demo_sample").value == "Low-light street · simulated darkness"
    assert app.slider(key="cfg_gamma").value == 1.5
    next(b for b in app.button if b.label == "Reset adjustments").click().run()
    assert app.slider(key="cfg_gamma").value == 1.0
    app.radio(key="workspace_view").set_value("AI analysis").run()
    assert any(b.label == "Review this frame with local vision" for b in app.button)
    assert not app.exception


def test_streamed_download_range_and_unregistered_path(tmp_path):
    path = tmp_path / "sample.mp4"
    path.write_bytes(bytes(range(256)) * 100)
    server = FileServer()
    try:
        url = server.url(path)
        req = urllib.request.Request(url, headers={"Range": "bytes=20-39"})
        with urllib.request.urlopen(req) as result:
            assert result.status == 206
            assert result.read() == path.read_bytes()[20:40]
            assert result.headers["Content-Range"] == "bytes 20-39/25600"
        with urllib.request.urlopen(server.url(path, download=True)) as result:
            assert "attachment" in result.headers["Content-Disposition"]
        with pytest.raises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(url.replace(server.token, "invalid-token"))
        assert error.value.code == 404
    finally:
        server.close()
