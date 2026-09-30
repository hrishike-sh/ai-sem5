"""Loopback integration tests with synthetic evidence and fake auth/model services."""
import io
import json
from copy import deepcopy
import urllib.error
import urllib.request

import pytest
from PIL import Image

from ivp.media import source_revision
from traffic_ai.violation_images import image_descriptors, image_fingerprint
from traffic_ai.violation_llm import card_fingerprint
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


def test_image_explanation_requires_preview_and_binds_server_images(server, saved_review):
    calls = []
    class FakeExplainer:
        def explain(self, card, consent=False, images=None):
            calls.append((card, consent, images))
            return {'state': 'fixture', 'vision_mode': 'selected_frames', 'explanation': 'Generated fixture; no live model.'}
    server.lab.violation_explainer = FakeExplainer()
    card = saved_review['cards'][0]
    bound = {'id': saved_review['run_id'], 'card_id': card['id'], 'fingerprint': saved_review['fingerprint']}
    with pytest.raises(urllib.error.HTTPError):
        request(server, '/api/violation-explain', {**bound, 'mode': 'images_and_facts', 'consent': True})
    assert not calls
    preview = request(server, '/api/violation-prepare', bound)
    assert preview['mode'] == 'images_and_facts' and not calls
    available = [f for f in preview['frames'] if f['available']]
    assert len(available) == 1
    result = request(server, '/api/violation-explain', {**bound, 'mode': 'images_and_facts', 'consent': True,
        'preparation_id': preview['preparation_id'], 'images': [{'data_url': 'https://untrusted.invalid/image'}]})
    assert result['mode'] == 'images_and_facts'
    assert len(calls) == 1 and calls[0][1] is True
    assert calls[0][2][0]['id'] == available[0]['id']
    assert calls[0][2][0]['sha256'] == available[0]['sha256']
    assert calls[0][2][0]['data_url'].startswith('data:image/jpeg;base64,')


def test_image_prepare_and_send_fail_on_source_change(server, saved_review):
    calls = []
    class FakeExplainer:
        def explain(self, *args, **kwargs):
            calls.append(True)
            return {}
    server.lab.violation_explainer = FakeExplainer()
    bound = {'id': saved_review['run_id'], 'card_id': saved_review['cards'][0]['id'], 'fingerprint': saved_review['fingerprint']}
    preview = request(server, '/api/violation-prepare', bound)
    source = server.lab.violation_reviews.root / 'uploads/synthetic.png'
    source.write_bytes(b'changed source')
    with pytest.raises(urllib.error.HTTPError):
        request(server, '/api/violation-explain', {**bound, 'mode': 'images_and_facts', 'consent': True,
                                                'preparation_id': preview['preparation_id']})
    assert not calls


class CompletedImageFixture:
    """Return the real adapter's result shape using only synthetic enum choices."""
    discarded_text = 'DISCARDED_FIXTURE_PROSE synthetic-oauth-never-issued'

    def __init__(self):
        self.calls = []
        self.invalid_enum = False

    def explain(self, card, consent=False, images=None):
        assert consent is True and images
        self.calls.append((deepcopy(card), deepcopy(images)))
        observations = [{'frame_id': image['id'], 'category': category, 'observation': observation,
                         'text': self.discarded_text}
                        for image in images for category, observation in (
                            ('head_visibility', 'unclear'), ('objects', 'unclear'),
                            ('direction_context', 'direction_context_unverified'))]
        if self.invalid_enum:
            observations[0]['observation'] = 'guilty_speeding_120_kmh'
        return {'state': 'explained', 'provider': 'chatgpt_subscription', 'model': 'fixture-model',
                'card_id': card['id'], 'card_fingerprint': card_fingerprint(card),
                'rule': card['rule'], 'status': card['status'], 'reason': card['reason'],
                'explanation': self.discarded_text,
                'facts': deepcopy(card['facts']), 'missing_facts': deepcopy(card['missing_facts']),
                'fact_ids': [item['id'] for item in card['facts']],
                'missing_fact_ids': [item['id'] for item in card['missing_facts']],
                'frame_ids': [image['id'] for image in images], 'vision_mode': 'selected_frames',
                'images_sent': image_descriptors(images), 'image_fingerprint': image_fingerprint(images),
                'visual_observations': observations, 'visual_notice': self.discarded_text}


@pytest.fixture
def completed_image_request(server, saved_review):
    run = server.lab.violation_reviews.root / 'runs' / saved_review['run_id']
    # Keep a genuine missing prerequisite in this generated record so the whole
    # API/cache/export round trip must preserve it even after human acceptance.
    record = json.loads((run / 'detections.jsonl').read_text(encoding='utf-8'))
    record['decisions'][0]['evidence'].pop('association')
    (run / 'detections.jsonl').write_text(json.dumps(record), encoding='utf-8')
    view = request(server, '/api/violation-review?id=' + run.name)
    card = view['cards'][0]
    assert card['missing_facts']
    bound = {'id': view['run_id'], 'card_id': card['id'], 'fingerprint': view['fingerprint']}
    preview = request(server, '/api/violation-prepare', bound)
    model = CompletedImageFixture()
    server.lab.violation_explainer = model
    payload = {**bound, 'mode': 'images_and_facts', 'consent': True,
               'preparation_id': preview['preparation_id']}
    originals = {name: (run / name).read_bytes() for name in
                 ('status.json', 'request.json', 'report.json', 'detections.jsonl')}
    return bound, payload, card, model, run, originals


def _assert_no_exported_media_or_secrets(value, server, model):
    serialized = json.dumps(value, ensure_ascii=False)
    assert server.lab.token not in serialized
    assert CompletedImageFixture.discarded_text not in serialized
    assert 'synthetic-oauth-never-issued' not in serialized
    assert 'data_url' not in serialized and 'data:image/' not in serialized
    for _, images in model.calls:
        for image in images:
            assert image['data_url'].split(',', 1)[1] not in serialized


def test_completed_selected_image_result_survives_decision_reopen_and_export(server, completed_image_request):
    bound, payload, original, model, run, originals = completed_image_request
    for kwargs, code in (({'token': False}, 403), ({'data': {**payload, 'consent': False}}, 400)):
        with pytest.raises(urllib.error.HTTPError) as error:
            request(server, '/api/violation-explain', **{'data': payload, **kwargs})
        assert error.value.code == code
    assert model.calls == [] and not (run / 'violation-explanations.json').exists()

    result = request(server, '/api/violation-explain', payload)
    assert result['saved'] is True and result['state'] == 'explained'
    assert result['vision_mode'] == 'selected_frames' and result['mode'] == 'images_and_facts'
    assert result['image_fingerprint'] == image_fingerprint(model.calls[0][1])
    assert len(result['visual_observations']) == len(model.calls[0][1]) * 3
    assert all(item['text'].startswith('Model observation: ') for item in result['visual_observations'])
    assert result['missing_facts'] == original['missing_facts']

    decided = request(server, '/api/violation-decision', {**bound, 'decision': 'accept', 'note': 'Synthetic reviewer decision'})
    reopened = request(server, '/api/violation-review?id=' + run.name)
    for view in (decided, reopened):
        card = view['cards'][0]
        cached = view['explanations'][original['id']]
        assert card['review']['decision'] == 'accept'
        assert {key: value for key, value in card.items() if key != 'review'} == {
            key: value for key, value in original.items() if key != 'review'}
        assert cached['saved'] is True and cached['saved_at'] == result['saved_at']
        assert cached['evidence_fingerprint'] == result['evidence_fingerprint']
        assert cached['card_fingerprint'] == card_fingerprint(card)
        for key in ('status', 'reason', 'facts', 'missing_facts', 'explanation', 'visual_observations', 'images_sent'):
            assert cached[key] == result[key]
        _assert_no_exported_media_or_secrets(view, server, model)

    export = request(server, '/api/violation-export', bound)
    with urllib.request.urlopen(export['url'], timeout=15) as response:
        assert 'attachment' in response.headers['Content-Disposition']
        exported = json.load(response)
    assert exported['explanations'][original['id']] == reopened['explanations'][original['id']]
    assert exported['cards'][0]['status'] == original['status'] and exported['external_reporting'] is False
    _assert_no_exported_media_or_secrets(exported, server, model)
    _assert_no_exported_media_or_secrets(json.loads((run / 'violation-explanations.json').read_bytes()), server, model)
    assert all((run / name).read_bytes() == raw for name, raw in originals.items())
    assert len(model.calls) == 1  # Reviewing, reopening and exporting cannot trigger inference.


@pytest.mark.parametrize('failure', ['invalid-enum', 'cache-capacity', 'cache-write'])
def test_failed_completed_image_result_preserves_previous_cached_explanation(
        server, completed_image_request, monkeypatch, failure):
    import traffic_ai.violation_review as review_module

    bound, payload, original, model, run, originals = completed_image_request
    first = request(server, '/api/violation-explain', payload)
    cache = run / 'violation-explanations.json'
    previous = cache.read_bytes()
    if failure == 'invalid-enum':
        model.invalid_enum = True
    elif failure == 'cache-capacity':
        monkeypatch.setattr(review_module, 'MAX_JSON', 16)
    else:
        def failed_write(path, encoded):
            assert path == cache
            raise OSError('Synthetic cache storage failure')
        monkeypatch.setattr(server.lab.violation_reviews, '_write_bytes', failed_write)
    with pytest.raises(urllib.error.HTTPError) as error:
        request(server, '/api/violation-explain', payload)
    assert error.value.code == 400
    assert cache.read_bytes() == previous
    reopened = request(server, '/api/violation-review?id=' + run.name)
    cached = reopened['explanations'][original['id']]
    for key in ('saved_at', 'status', 'reason', 'missing_facts', 'explanation', 'visual_observations', 'image_fingerprint'):
        assert cached[key] == first[key]
    assert len(model.calls) == 2
    assert all((run / name).read_bytes() == raw for name, raw in originals.items())
    _assert_no_exported_media_or_secrets(reopened, server, model)


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
