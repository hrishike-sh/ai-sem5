"""Primary wiring and export contract, with explicitly mocked perception only."""
import json
from types import SimpleNamespace

import pytest

from traffic_ai import analyzer as module
from traffic_ai.control_association import ConflictAwareApproachResolver, POLICY
from tests.test_control_association import (
    test_actual_analyzer_hook_with_mocked_perception_exports_withheld_and_no_green_boxes as rehearsal,
)
from tests.test_ai_worker import FakeAnalyzer, image_request as image_request


def test_live_analyzer_binding_and_exported_graph_use_new_policy_without_rebinding_resolver(monkeypatch):
    # This assertion would fail if only the isolated resolver tests passed but
    # the production Analyzer still constructed the legacy resolver.
    assert module.ApproachResolver is ConflictAwareApproachResolver
    actual_analyzer=module.Analyzer
    instances=[]
    def capture(*args,**kwargs):
        instance=actual_analyzer(*args,**kwargs)
        instances.append(instance)
        return instance
    monkeypatch.setattr(module,'Analyzer',capture)

    class PerceptionOnlyPatches:
        def setattr(self,target,name,value):
            if target is module and name=='ApproachResolver':
                assert module.ApproachResolver is value
                return  # Exercise the production import, never replace it.
            monkeypatch.setattr(target,name,value)

    # Reuse the pinned synthetic scenario with fake detections/scene models.
    # Its actual Analyzer/rule/event/render path must withhold the conflict.
    rehearsal(PerceptionOnlyPatches())
    instance=instances[0]
    assert isinstance(instance.approach_resolver,ConflictAwareApproachResolver)
    summary=instance.summary()
    scene=instance.scene_snapshot()
    for exported in (summary,scene):
        graph=exported['control_association']
        assert graph['policy']==POLICY
        assert graph['graphs']['signal']['observations'][0]['reason']=='competing_observations'
        assert graph['graphs']['sign']['observations'][0]['source_pointer']=='/sign_observations/0'
        assert json.loads(json.dumps(graph,allow_nan=False))==graph
        for kind in graph['graphs'].values():
            for record in kind['observations']:
                collection,index=record['source_pointer'].strip('/').split('/')
                assert exported[collection][int(index)]['association']['source_pointer']==record['source_pointer']
    assert summary['frames']==5 and summary['events_by_rule']=={}
    assert summary['assessment_counts'] and sum(summary['assessment_counts'].values())==45


@pytest.mark.parametrize('stable',[True,False,None])
def test_worker_records_actual_camera_gate_without_inventing_missing_stability(image_request,monkeypatch,stable):
    from traffic_ai import worker
    source,run=image_request
    before=source.read_bytes()
    class GateAnalyzer(FakeAnalyzer):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            self.scene=SimpleNamespace(stationary_camera=stable)
    monkeypatch.setattr(worker,'Analyzer',GateAnalyzer)
    worker.process_run(run)
    record=json.loads((run/'detections.jsonl').read_text())
    if stable is None:
        assert 'camera_stable' not in record
    else:
        assert record['camera_stable'] is stable
    assert source.read_bytes()==before
