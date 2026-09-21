"""Current source images must still match the historical review before display."""
import json

from PIL import Image
import pytest

from ivp.media import write_json
from tests.test_evidence_bridge import evidence_source, finished, fingerprints
from tests.test_web_server import server, request


@pytest.mark.parametrize('change', ['missing', 'different_bytes'])
def test_missing_or_changed_source_thumbnail_is_not_substituted_into_saved_facts(server, evidence_source, change):
    source, event = evidence_source
    event['thumbnail'] = 'fixture.png'
    image = source/event['thumbnail']
    Image.new('RGB', (10,10), 'red').save(image)
    (source/'motion-incidents.jsonl').write_text(json.dumps(event)+'\n',encoding='utf8')
    write_json(source/'motion-incidents.json', {'event_count':1,'counts':{'withheld':1},'events':[event],'omitted_from_index':0})
    ready = finished(server, request(server,'/api/evidence-review',{'source_run':source.name}))
    assert ready['review']['events'][0]['thumbnail_url']
    artifact = source.parent/'evidence-reviews'/ready['id']/'offline'
    before = fingerprints(artifact)
    if change == 'missing':
        image.unlink()
    else:
        Image.new('RGB', (10,10), 'blue').save(image)
    restored = request(server,'/api/evidence-review?id='+ready['id'])
    assert restored['artifact_integrity']['state']=='verified'
    assert restored['source_integrity']['state']=='changed'
    assert restored['review']['events'][0]['event']==event
    assert not restored['review']['events'][0].get('thumbnail_url')
    assert restored['review']['events'][0]['thumbnail_notice']=='Recorded thumbnail is missing or changed; image withheld.'
    assert restored['downloads'] and before==fingerprints(artifact)
