"""One black-box export covers untrusted metadata, quoted filenames and CSV text."""
import csv
import hashlib
from html.parser import HTMLParser
import io
import json

from PIL import Image

from tests.test_evidence_review import incident as incident,write_run,TRACK_HEADER
from traffic_ai.review_cli import export_review,validate_review_bundle


class Markup(HTMLParser):
    def __init__(self):super().__init__();self.tags=[];self.attrs=[];self.text=[]
    def handle_starttag(self,tag,attrs):self.tags.append(tag);self.attrs.extend(attrs)
    def handle_data(self,data):self.text.append(data)


def test_export_quotes_filenames_and_keeps_hostile_csv_and_nested_text_inert(tmp_path,incident):
    payload='</pre><svg onload="alert(1)"></svg><script>alert(2)</script> & \'quoted\''
    source=tmp_path/"source 'quoted' & café";output=tmp_path/"review"
    filename="evidence 'quoted' & café.png"
    incident.update(event_id='id" onpointerenter="alert(3)',reason=payload,thumbnail=filename)
    incident['future_exporter_field']={'nested':payload}
    write_run(source,[incident]);Image.new('RGB',(10,10),'gray').save(source/filename)
    rows=io.StringIO(newline='');writer=csv.writer(rows)
    writer.writerow(TRACK_HEADER.strip().split(',')+['future_text'])
    writer.writerow([.2,4,'12.30','px/s','estimated','observed',payload])
    (source/'motion-tracks.csv').write_text(rows.getvalue(),encoding='utf-8')
    before={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in source.iterdir()}
    document=export_review(source,output)
    assert validate_review_bundle(output)['complete'] is True
    assert document['events'][0]['event']==incident
    assert document['events'][0]['track_context']['samples'][0]['values']['future_text']==payload
    page=(output/'review.html').read_text(encoding='utf-8');parsed=Markup();parsed.feed(page)
    assert not {'script','svg','iframe','object','embed','base','form'} & set(parsed.tags)
    assert not any(name.lower().startswith('on') for name,value in parsed.attrs)
    resources=[value for name,value in parsed.attrs if name in {'src','href','action','poster','srcset','data'}]
    assert len(resources)==1 and resources[0].startswith('data:image/png;base64,')
    text=''.join(parsed.text)
    assert payload in text and filename in text and source.name in text
    assert incident['event_id'] in text
    assert json.dumps(incident,ensure_ascii=False,indent=2) in text
    assert before=={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in source.iterdir()}
