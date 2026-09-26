"""Loopback integration tests with synthetic evidence and fake auth/model services."""
import io
import json
import urllib.error
import urllib.request

import pytest
from PIL import Image

from ivp.media import source_revision
from tests.test_web_server import server, request


@pytest.fixture
def saved_review(server):
    root = server.lab.violation_reviews.root
    source = root / 'uploads' / 'synthetic.png'
    source.parent.mkdir(parents=True)
    Image.new('RGB', (32, 24), 'navy').save(source)
    run = root / 'runs' / 'synthetic-contract'
    run.mkdir(parents=True)
    payloads = {
        'status.json': {'state': 'complete'},
        'request.json': {'source': str(source), 'synthetic': True,
                         'info': {'kind': 'image', 'source_revision': source_revision(source)}},
        'report.json': {'ai': {}, 'synthetic': True},
        'detections.jsonl': {'timestamp': 0., 'scene_id': 0, 'synthetic': True,
                            'decisions': [{'rule': 'helmet_absence', 'timestamp': 0., 'track_id': 1,
                            'verdict': 'violation', 'reason': 'Synthetic contract fixture; not inference.',
                            'evidence': {'heads': [[2, 2, 6, 6]], 'association': 'Synthetic rider association'}}]},
    }
    for name, value in payloads.items():
        (run / name).write_text(json.dumps(value), encoding='utf-8')
    return request(server, '/api/violation-review?id=' + run.name)


@pytest.mark.parametrize('path', ['/', '/ai', '/ai/'])
def test_guided_page_has_session_token_and_security_headers(server, path):
    with urllib.request.urlopen(f'http://127.0.0.1:{server.server_port}{path}') as response:
        page = response.read().decode('utf-8')
        assert 'ViolationLens' in page and '/violation_lens.js' in page
        assert server.lab.token in page and '__REQUEST_TOKEN__' not in page
        assert response.headers['Cache-Control'] == 'no-store'
        assert "frame-ancestors 'none'" in response.headers['Content-Security-Policy']


def test_evidence_origin_matches_actual_file_service(server):
    assert request(server, '/api/config')['evidence_origin'] == f'http://127.0.0.1:{server.lab.files.server.server_port}'


def test_review_frame_disposition_and_export_round_trip(server, saved_review):
    view = saved_review
    card = view['cards'][0]
    bound = {'id': view['run_id'], 'card_id': card['id'], 'fingerprint': view['fingerprint']}
    frames = request(server, '/api/violation-frames', bound)['frames']
    during = next(frame for frame in frames if frame['role'] == 'during')
    assert during['available'] and during['timestamp'] == 0
    with urllib.request.urlopen(during['url']) as response:
        assert Image.open(io.BytesIO(response.read())).size == (32, 24)
    updated = request(server, '/api/violation-decision', {**bound, 'decision': 'accept', 'note': 'Synthetic reviewer note'})
    assert updated['cards'][0]['review']['decision'] == 'accept'
    assert updated['cards'][0]['status'] == card['status']
    export = request(server, '/api/violation-export', bound)
    with urllib.request.urlopen(export['url']) as response:
        assert 'attachment' in response.headers['Content-Disposition']
        result = json.load(response)
    assert result['synthetic'] is True and result['external_reporting'] is False
    assert result['cards'][0]['review']['note'] == 'Synthetic reviewer note'


def test_review_rejects_stale_or_cross_origin_mutations(server, saved_review):
    bound = {'id': saved_review['run_id'], 'card_id': saved_review['cards'][0]['id'],
             'fingerprint': 'stale', 'decision': 'accept'}
    with pytest.raises(urllib.error.HTTPError) as error:
        request(server, '/api/violation-decision', bound)
    assert error.value.code == 400
    with pytest.raises(urllib.error.HTTPError) as error:
        request(server, '/api/violation-decision', bound, token=False)
    assert error.value.code == 403


def test_explanation_uses_current_server_card_and_requires_consent(server, saved_review):
    calls = []
    class FakeExplainer:
        def explain(self, card, consent=False):
            if not consent:
                raise ValueError('Explicit consent required.')
            calls.append(card)
            return {'state': 'fixture', 'explanation': 'Synthetic adapter fixture; no live request.'}
    server.lab.violation_explainer = FakeExplainer()
    bound = {'id': saved_review['run_id'], 'card_id': saved_review['cards'][0]['id'],
             'fingerprint': saved_review['fingerprint'], 'facts': ['INVENTED CLIENT FACT']}
    with pytest.raises(urllib.error.HTTPError):
        request(server, '/api/violation-explain', bound)
    result = request(server, '/api/violation-explain', {**bound, 'consent': True})
    assert result['card_id'] == bound['card_id']
    assert calls == [saved_review['cards'][0]]
    assert 'INVENTED CLIENT FACT' not in json.dumps(calls)


def test_auth_start_needs_approval_and_csrf_before_persistent_state(server, tmp_path):
    from traffic_ai.violation_auth import SubscriptionAuth
    store = tmp_path / 'never-created-auth'
    server.lab.violation_auth = SubscriptionAuth(storage_dir=store,
        callback_uri=f'http://127.0.0.1:{server.server_port}/auth/callback')
    for data, token, code in [({}, True, 400), ({'approve': True}, False, 403)]:
        with pytest.raises(urllib.error.HTTPError) as error:
            request(server, '/api/violation-auth/start', data, token=token)
        assert error.value.code == code
    assert not store.exists()


def test_auth_cookie_binds_same_browser_callback_without_network(server):
    calls = []
    class FakeAuth:
        def start(self, approve=False, session_id=''):
            assert approve is True and session_id
            calls.append(('start', session_id))
            return {'authorization_url': 'https://auth.openai.com/fake-fixture'}
        def complete(self, query, session_id=''):
            assert session_id == calls[0][1] and query == {'state': 'fixture', 'code': 'synthetic'}
            calls.append(('complete', session_id))
    server.lab.violation_auth = FakeAuth()
    with pytest.raises(urllib.error.HTTPError) as wrong_host:
        request(server, '/api/violation-auth/start', {'approve': True}, host=f'localhost:{server.server_port}')
    assert wrong_host.value.code == 400 and calls == []
    start = urllib.request.Request(f'http://127.0.0.1:{server.server_port}/api/violation-auth/start',
        data=b'{"approve":true}', headers={'X-Traffic-Token': server.lab.token})
    with urllib.request.urlopen(start) as response:
        cookie = response.headers['Set-Cookie']
        assert 'HttpOnly' in cookie and 'SameSite=Lax' in cookie
        assert json.load(response)['url'] == 'https://auth.openai.com/fake-fixture'
    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, *args):
            return None
    callback = urllib.request.Request(f'http://127.0.0.1:{server.server_port}/auth/callback?state=fixture&code=synthetic',
                                      headers={'Cookie': cookie.split(';')[0]})
    with pytest.raises(urllib.error.HTTPError) as error:
        urllib.request.build_opener(NoRedirect()).open(callback)
    assert error.value.code == 303
    assert error.value.headers['Location'] == '/?signin=connected'
    assert error.value.headers['Referrer-Policy'] == 'no-referrer'
    assert len(calls) == 2
