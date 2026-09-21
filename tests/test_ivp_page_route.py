import re
import urllib.error
import urllib.request

import pytest

from tests.test_web_server import server  # Existing isolated server fixture.


@pytest.mark.parametrize('route,page', [('/', 'full'), ('/ivp', 'ivp'), ('/ivp/', 'ivp')])
def test_page_routes_keep_security_absolute_assets_and_request_token(server, route, page):
    with urllib.request.urlopen(f'http://127.0.0.1:{server.server_port}{route}') as response:
        body = response.read().decode()
        assert response.status == 200
        assert response.headers['Content-Type'].startswith('text/html')
        assert response.headers['Cache-Control'] == 'no-store'
        assert response.headers['X-Content-Type-Options'] == 'nosniff'
        assert "frame-ancestors 'none'" in response.headers['Content-Security-Policy']
    assert f'data-app-page="{page}"' in body
    assert 'href="/styles.css"' in body and 'src="/app.js"' in body
    assert f'content="{server.lab.token}"' in body and '__REQUEST_TOKEN__' not in body
    assert re.search(r'id="ivp-page-link"[^>]*href="/ivp"', body)
    assert re.search(r'id="full-page-link"[^>]*href="/"', body)


@pytest.mark.parametrize('route', ['/ivp/missing', '/ivp.js', '/somewhere-else'])
def test_new_page_does_not_turn_unknown_routes_into_html(server, route):
    with pytest.raises(urllib.error.HTTPError) as error:
        urllib.request.urlopen(f'http://127.0.0.1:{server.server_port}{route}')
    assert error.value.code == 404
