"""Loopback API integration using synthetic pictures and mocked Astra responses."""
import io
import json
import urllib.error

from PIL import Image
import pytest

from traffic_ai.violation_images import image_fingerprint
from tests.test_direct_vision import service
from tests.test_web_server import server, request


class SyntheticAstra:
    def __init__(self):
        self.calls = 0

    def review(self, images, consent=False):
        self.calls += 1
        value = {'image_fingerprint': image_fingerprint(images),
                 'observations': [{'kind': 'head_protection_unclear', 'evidence_ids': [images[0]['id']],
                                   'uncertainties': ['model_may_misread']}], 'suspected_issues': []}
        reviewer, _, _ = service(images, value)
        return reviewer.review(images, consent)


@pytest.fixture
def direct(server):
    pixels = io.BytesIO()
    Image.new('RGB', (64, 48), 'gray').save(pixels, 'PNG')
    source = request(server, '/api/upload?name=synthetic-direct.png', raw=pixels.getvalue())
    preview = request(server, '/api/direct-prepare', {'media': source['id'], 'source_revision': source['source_revision']})
    model = SyntheticAstra()
    server.lab.direct_vision = model
    return source, preview, model


def bound(preview):
    return {key: preview[key] for key in ('review_id', 'fingerprint')}


def test_direct_flow_needs_no_detector_and_reopens_decision_and_export(server, direct):
    _, preview, model = direct
    assert preview['frames'][0]['available'] and preview['model_result'] is None and model.calls == 0
    result = request(server, '/api/direct-analyze', {**bound(preview), 'consent': True})
    assert model.calls == 1 and result['model_result']['model'] == 'gpt-6-astra'
    assert result['model_result']['suspected_issues'] == []
    decided = request(server, '/api/direct-decision', {**bound(preview), 'decision': 'reject', 'note': 'Synthetic API fixture.'})
    assert decided['review']['decision'] == 'reject'
    reopened = request(server, '/api/direct-review?id=' + preview['review_id'])
    assert reopened['review']['note'] == 'Synthetic API fixture.' and model.calls == 1
    exported = request(server, '/api/direct-export', bound(preview))
    path = server.lab.direct_reviews.root / 'direct-reviews' / preview['review_id'] / exported['filename']
    payload = json.loads(path.read_text())
    assert payload['external_reporting'] is False and payload['model_result']['model'] == 'gpt-6-astra'
    assert 'data:image' not in path.read_text() and 'synthetic-oauth' not in path.read_text()
    assert any(item['review_id'] == preview['review_id'] for item in request(server, '/api/direct-library'))


@pytest.mark.parametrize('consent', [None, False, 'true', 1])
def test_no_direct_network_without_literal_consent(server, direct, consent):
    _, preview, model = direct
    with pytest.raises(urllib.error.HTTPError):
        request(server, '/api/direct-analyze', {**bound(preview), 'consent': consent})
    assert model.calls == 0


def test_changed_original_blocks_direct_send_before_model(server, direct):
    source, preview, model = direct
    path, _ = server.lab.media[source['id']]
    path.write_bytes(b'Changed fixture after preview')
    with pytest.raises(urllib.error.HTTPError):
        request(server, '/api/direct-analyze', {**bound(preview), 'consent': True})
    assert model.calls == 0
    assert request(server, '/api/direct-review?id=' + preview['review_id'])['source_readiness']['available'] is False


def test_direct_mutation_requires_csrf_and_ignores_client_supplied_images(server, direct):
    _, preview, model = direct
    with pytest.raises(urllib.error.HTTPError) as denied:
        request(server, '/api/direct-analyze', {**bound(preview), 'consent': True}, token=False)
    assert denied.value.code == 403 and model.calls == 0
    result = request(server, '/api/direct-analyze', {**bound(preview), 'consent': True,
                     'images': [{'data_url': 'https://untrusted.invalid/image'}], 'model': 'gpt-6-sol'})
    assert result['model_result']['evidence'][0]['sha256'] == preview['frames'][0]['sha256']
    assert result['model_result']['model'] == 'gpt-6-astra' and model.calls == 1
