import json
import numpy as np
import pytest
from ivp.zebra import LABEL, Occupancy, assess, draw, validate_region


def band():
    return validate_region({'schema_version':1,'semantic':'zebra_crossing','reviewed':True,
        'media_id':'m','source_revision':'r','reference_seconds':0.,
        'points':[[.2,.3],[.8,.3],[.8,.6],[.2,.6]]},'m','r',0.)


@pytest.mark.parametrize('box,status',[
    ((40,40,60,80),'observed'),  # Bottom centre lies outside the band.
    ((40,10,60,25),'no_overlap'),
    ((40,61,60,81),'no_overlap'),
    ((40,60,60,80),'withheld'),  # Float32 tangent noise must never get a label.
    ((79,59,99,79),'withheld'),
    ((0,35,30,55),'withheld'),
    ((40,40,60,100),'withheld'),
])
def test_box_overlap_and_uncertainty(box,status):
    row=assess(band(),box,(100,100),label='car',confidence=.9)
    if box==(40,60,60,80):
        assert row['status'] in ('no_overlap','withheld')
    else:
        assert row['status']==status
    assert row['label']==(LABEL if row['status']=='observed' else None)
    assert row['legal_violation'] is False
    assert row['eligible_for_geometry'] is False


def test_full_paint_occlusion_is_not_a_dependency():
    # No image or paint mask is an input. A reviewed persistent band can be
    # completely hidden by the candidate box without losing geometric overlap.
    row=assess(band(),(15,25,85,65),(100,100),label='car',confidence=.9)
    assert row['label']==LABEL
    assert row['crossing_overlap_fraction']==pytest.approx(1.)
    assert 'rectangle' in row['basis']


@pytest.mark.parametrize('change',[
    {'reviewed':False},{'semantic':'paint'},{'source_revision':'old'},
    {'points':[[0,.3],[.8,.3],[.8,.6],[0,.6]]},
    {'points':[[.2,.3],[.8,.6],[.8,.3],[.2,.6]]},
    {'legacy':{'eligible_for_rules':True}},
])
def test_unreviewed_cropped_stale_legacy_rejected(change):
    region={k:v for k,v in band().items() if k!='region_id'};region.update(change)
    with pytest.raises(ValueError):
        validate_region(region,'m','r',0.)


def track(box=(40,40,20,40),**kw):
    return [(7,dict(box=box,model_label='car',confidence=.9,tracking_state='observed',**kw))]


def test_episode_distinct_frames_debounce_exits_and_occlusion():
    engine=Occupancy(band())
    assert engine.step(track(),0,(100,100))[0]['status']=='pending'
    assert engine.step(track(),0,(100,100))[0]['status']=='withheld'
    assert engine.step(track(),.1,(100,100))[0]['status']=='pending'
    assert engine.step(track(),.2,(100,100))[0]['episode_id']==1
    assert engine.episode_count==1
    engine.step([],1.,(100,100))
    for t in (1.1,1.2):
        assert engine.step(track(),t,(100,100))[0]['status']=='pending'
    row=engine.step(track(),1.3,(100,100))[0]
    assert row['episode_id']==1 and row['new_episode'] is False
    engine.step(track((40,65,20,20)),1.4,(100,100))
    engine.step(track((40,65,20,20)),1.4,(100,100))  # Duplicate cannot be exit #2.
    assert engine.step(track(),1.5,(100,100))[0]['status']=='pending'
    for t in (1.6,1.7):
        row=engine.step(track(),t,(100,100))[0]
    assert row['episode_id']==1
    engine.step(track((40,65,20,20)),1.8,(100,100))
    engine.step(track((40,65,20,20)),1.9,(100,100))
    for t in (2.,2.1,2.2):
        row=engine.step(track(),t,(100,100))[0]
    assert row['episode_id']==2 and row['new_episode'] is True


@pytest.mark.parametrize('gate',[{'registration_valid':False},{'scene_changed':True}])
def test_video_uncertainty_cannot_label(gate):
    engine=Occupancy(band())
    for stamp in (0,.1,.2):
        rows=engine.step(track(),stamp,(100,100),**gate)
        assert rows[0]['label'] is None
    assert engine.episode_count==0


def test_overlay_and_rows_agree():
    rgb=np.zeros((100,100,3),np.uint8)
    region=band()
    row=assess(region,(40,40,60,80),rgb.shape,label='car',confidence=.9)
    rendered=draw(rgb,region,[row])
    assert not np.array_equal(rendered,draw(rgb,region,[]))
    assert not rgb.any()


@pytest.mark.parametrize('box,confidence',[
    ((40,float('nan'),60,80),.9),((40,40,60,80),float('inf')),
    ((40,40,60,80),None),((40,'unknown',60,80),.9),
])
def test_malformed_candidates_withheld_as_json_safe(box,confidence):
    row=assess(band(),box,(100,100),confidence=confidence)
    assert row['status']=='withheld' and row['label'] is None
    json.dumps(row,allow_nan=False)


def test_absence_of_texture_is_not_current_registration_evidence():
    from ivp.motion import MotionAnalyzer
    engine=MotionAnalyzer()
    rgb=np.zeros((120,160,3),np.uint8)
    engine.step(rgb,0.)
    assert engine.registration_observed is True  # The reference frame itself.
    engine.step(rgb,.1)
    assert engine.stabilizer.failed is False  # Legacy flag alone is insufficient.
    assert engine.registration_observed is False
