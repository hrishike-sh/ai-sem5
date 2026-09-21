import pytest
from traffic_ai.contracts import Detection
from traffic_ai.helmet_observations import observe

MODEL='iam-tsr/yolov8n-helmet-detection'


def candidates(label='helmet',confidence=.95,head_box=(46,22,62,40)):
    return [Detection('motorcycle',.95,(20,60,100,120)),
            Detection('person',.95,(40,20,70,110)),Detection(label,confidence,head_box,model=MODEL)]


@pytest.mark.parametrize('label,state',[('helmet','helmet_present'),('no_helmet','bare_head')])
def test_explicit_head_model_observation_with_indexed_association(label,state):
    summary=observe(candidates(label),(200,200))
    row=summary['observations'][0]
    assert row['state']==state
    assert row['motorcycle_detection_index']==0 and row['person_detection_index']==1
    assert row['head_detection_indices']==[2]
    assert row['eligible_for_rules'] is False and row['legal_violation'] is False


@pytest.mark.parametrize('case',['missing','weak','conflicting','shared','outside','tiny','clipped','wrong_model','weak_person'])
def test_unknowns_never_become_bare_head(case):
    ds=candidates('no_helmet')
    if case=='missing':ds=ds[:2]
    elif case=='weak':ds=candidates('no_helmet',.69)
    elif case=='conflicting':ds.append(Detection('helmet',.98,(46,22,62,40),model=MODEL))
    elif case=='shared':ds.append(Detection('person',.95,(42,20,72,110)))
    elif case=='outside':ds=candidates('no_helmet',head_box=(120,22,136,40))
    elif case=='tiny':ds=candidates('no_helmet',head_box=(46,22,50,26))
    elif case=='clipped':ds[1]=Detection('person',.95,(40,0,70,110))
    elif case=='wrong_model':ds[2]=Detection('no_helmet',.95,ds[2].box,model='unverified')
    elif case=='weak_person':ds[1]=Detection('person',.4,ds[1].box)
    summary=observe(ds,(200,200))
    assert summary['observations'] and all(r['state']=='unknown' for r in summary['observations'])
    assert all(not r['head_detection_indices'] for r in summary['observations'])


def test_no_motorcycle_is_not_compliance():
    summary=observe(candidates()[1:],(200,200))
    assert summary['motorcycle_candidates']==0 and not summary['observations']
    assert summary['unassociated_heads']==1


@pytest.mark.parametrize('primary,other',[('no_helmet','helmet'),('helmet','no_helmet')])
def test_lower_confidence_opposite_head_candidate_vetoes_claim(primary,other):
    ds=candidates(primary)
    ds.append(Detection(other,.69,ds[2].box,model=MODEL))
    row=observe(ds,(200,200))['observations'][0]
    assert row['state']=='unknown' and not row['head_detection_indices']
    assert row['candidate_head_detection_indices']==[2,3]
    assert 'Conflicting' in row['reason']
