from pathlib import Path
import urllib.error

import pytest
import web_server

from ivp.jobs import now
from ivp.media import write_json
from tests.test_web_server import server, request


def saved(root,name,source='other.mp4',state='complete'):
    folder=root/'runs'/name
    folder.mkdir(parents=True)
    write_json(folder/'request.json',{'source':source})
    write_json(folder/'status.json',{'state':state,'message':'Saved result','updated_at':now()})
    return folder


def test_global_search_finds_an_older_run_and_returns_only_metadata(server,tmp_path,monkeypatch):
    for i in range(40):saved(tmp_path,f'20260101-{i:06}',source='india-city.mp4' if i==0 else 'other.mp4')
    target=tmp_path/'runs/20260101-000000/report.json'
    write_json(target,{'source':'india-city.mp4','large_measurements':['unused']*100})
    original=Path.open
    def no_heavy_report(path,*args,**kwargs):
        if path==target:raise AssertionError('Library rows must use the small request instead of the processing report')
        return original(path,*args,**kwargs)
    monkeypatch.setattr(Path,'open',no_heavy_report)
    page=request(server,'/api/run-library?q=INDIA-CITY')
    assert len(page['runs'])==1 and page['runs'][0]['id']=='20260101-000000'
    assert page['runs'][0]['source']=='india-city.mp4'
    assert set(page['runs'][0])=={'id','source','status'} and not page['has_more']


def test_pagination_reaches_older_runs_without_duplicates_when_new_runs_arrive(server,tmp_path):
    ids=[f'20260101-{i:06}' for i in range(40)]
    for identifier in ids:saved(tmp_path,identifier)
    first=request(server,'/api/run-library')
    assert len(first['runs'])==30 and first['has_more']
    saved(tmp_path,'20260101-999999')
    second=request(server,'/api/run-library?before='+first['next_before'])
    assert len(second['runs'])==10 and not second['has_more'] and second['next_before'] is None
    assert [r['id'] for r in first['runs']+second['runs']]==sorted(ids,reverse=True)


def test_status_filter_is_applied_across_the_archive_before_page_limit(server,tmp_path):
    for i in range(40):saved(tmp_path,f'20260101-{i:06}',state='cancelled' if i==0 else 'complete')
    page=request(server,'/api/run-library?status=cancelled')
    assert [r['id'] for r in page['runs']]==['20260101-000000']


def test_library_name_cache_refreshes_when_metadata_changes(server,tmp_path):
    folder=saved(tmp_path,'same-run',source='before.mp4')
    assert len(request(server,'/api/run-library?q=before')['runs'])==1
    write_json(folder/'request.json',{'source':'changed-longer-name.mp4'})
    assert request(server,'/api/run-library?q=before')['runs']==[]
    assert request(server,'/api/run-library?q=changed')['runs'][0]['source']=='changed-longer-name.mp4'


def test_damaged_metadata_and_status_do_not_hide_other_saved_runs(server,tmp_path):
    bad=saved(tmp_path,'bad-run')
    (bad/'request.json').write_text('{broken',encoding='utf8')
    (bad/'status.json').write_text('{broken',encoding='utf8')
    saved(tmp_path,'good-run',source='india-city.mp4')
    page=request(server,'/api/run-library')
    assert {r['id'] for r in page['runs']}=={'bad-run','good-run'}
    row=next(r for r in page['runs'] if r['id']=='bad-run')
    assert row['status']['state']=='unknown' and row['source']=='Unknown source'


def test_legacy_runs_api_contract_is_unchanged(server,tmp_path,monkeypatch):
    saved(tmp_path,'existing-run',source='india-city.mp4')
    monkeypatch.setattr(web_server,'history',lambda:sorted((tmp_path/'runs').glob('*/status.json'),reverse=True))
    old=request(server,'/api/runs')
    assert isinstance(old,list) and old[0]['report']['source']=='india-city.mp4'
    assert 'downloads' in old[0] and 'preview' in old[0]


def test_paging_accepts_existing_legacy_folder_names(server,tmp_path):
    saved(tmp_path,'run.01')
    saved(tmp_path,'run.02')
    first=request(server,'/api/run-library?limit=1')
    second=request(server,'/api/run-library?limit=1&before='+first['next_before'])
    assert [r['id'] for r in first['runs']+second['runs']]==['run.02','run.01']


@pytest.mark.parametrize('query',['limit=0','limit=101','status=anything','before=../outside','q='+('a'*201)])
def test_library_query_validation(server,query):
    with pytest.raises(urllib.error.HTTPError) as error:request(server,'/api/run-library?'+query)
    assert error.value.code==400
