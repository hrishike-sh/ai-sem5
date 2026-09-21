"""One cancellable local overview, independent of the video worker lock."""
import json
import math
from pathlib import Path
import re
import secrets
import threading
import time

from .jobs import now
from .media import write_json


class SceneReviews:
    def __init__(self, root, executor):
        self.root = Path(root)/'runs'/'scene-reviews'
        self.executor = executor
        self.future = None
        self.current = None
        self.cancel_event = None
        self.started = None
        self.guard = threading.Lock()

    @property
    def active(self):
        return self.future is not None and not self.future.done()

    def folder(self, identifier):
        if not isinstance(identifier, str) or not re.fullmatch(r'[0-9a-f]{24}', identifier):
            raise ValueError('Invalid scene-review identifier.')
        folder = self.root/identifier
        if folder.resolve().parent != self.root.resolve() or not (folder/'request.json').is_file():
            raise ValueError('Scene review not found.')
        return folder

    @staticmethod
    def read(path):
        try:
            with path.open('rb') as stream:
                data = stream.read(2*1024*1024+1)
            if len(data)>2*1024*1024:
                raise ValueError('Overview record exceeds its read limit.')
            value = json.loads(data)
            if not isinstance(value, dict):
                raise ValueError('Expected an overview object.')
            return value
        except (OSError, ValueError) as exc:
            raise ValueError('Saved overview record is unreadable. Request a new overview; previous exports remain available.') from exc

    @staticmethod
    def validate(review, timestamp):
        """Check the fields consumed by the UI before publishing/replacing a result."""
        from traffic_ai.local_vision import validate_observation
        from traffic_ai.bounded_vision import VisionUnavailable, validate_selection
        try:
            if review.get('kind') != 'local_vision_review' or review.get('state') != 'complete':
                raise ValueError('Not a completed scene overview.')
            validate_observation(review.get('observation'))
            if review.get('provider') == 'local_llama_cpp_cpu' or 'selection' in review:
                if validate_selection(review.get('selection')) != review['observation']:
                    raise ValueError('Overview text differs from its bounded category selection.')
            elapsed = review.get('total_seconds')
            if elapsed is None:
                elapsed = review.get('duration_seconds')
            for value in (review.get('timestamp'), timestamp, elapsed):
                if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                    raise ValueError('Invalid overview timing.')
            if review['timestamp'] != timestamp:
                raise ValueError('Overview timestamp differs from its request.')
        except (TypeError, ValueError, KeyError, OverflowError, VisionUnavailable) as exc:
            raise ValueError('Saved overview contents are unreadable or invalid. Request a new overview; previous exports remain available.') from exc

    @staticmethod
    def state(folder, state, message):
        write_json(folder/'status.json', {'state':state, 'message':message, 'updated_at':now()})

    def start(self, rgb, timestamp, media_id, revision):
        with self.guard:
            if self.active:
                raise RuntimeError('A local scene overview is already running. Cancel it or wait for it to finish.')
            identifier = secrets.token_hex(12)
            folder = self.root/identifier
            folder.mkdir(parents=True, exist_ok=False)
            write_json(folder/'request.json', {'id':identifier, 'media_id':media_id,
                       'source_revision':revision, 'timestamp':timestamp, 'created_at':now()})
            self.state(folder, 'running', 'Local CPU overview running; startup budget 30 s, inference budget 60 s.')
            self.current, self.cancel_event, self.started = identifier, threading.Event(), time.monotonic()
            try:
                self.future = self.executor.submit(self.build, folder, rgb, timestamp, self.cancel_event)
            except RuntimeError:
                self.state(folder, 'failed', 'The overview service stopped. Reload and try again.')
                raise
        return self.get(identifier)

    def build(self, folder, rgb, timestamp, cancel):
        from traffic_ai.local_vision import review_frame
        try:
            review = review_frame(rgb, timestamp, output=folder/'model', cancel=cancel.is_set)
            if cancel.is_set():
                self.state(folder, 'cancelled', 'Overview cancelled; previous successful results are retained.')
                return
            self.validate(review, timestamp)
            write_json(folder/'review.json', review)
            self.state(folder, 'complete', 'Unverified local feature overview ready.')
        except Exception as exc:
            record = getattr(exc, 'record', None)
            if isinstance(record, dict):
                record = {k:v for k,v in record.items() if k not in ('observation', 'selection')}
                write_json(folder/'attempt.json', record)
            cancelled = cancel.is_set() or isinstance(record, dict) and record.get('state') == 'cancelled'
            self.state(folder, 'cancelled' if cancelled else 'unavailable',
                       'Overview cancelled; previous successful results are retained.' if cancelled else
                       str(exc) or 'Local overview unavailable; IVP results remain available.')

    def get(self, identifier):
        folder = self.folder(identifier)
        request = self.read(folder/'request.json')
        status = self.read(folder/'status.json')
        if status.get('state') not in ('running','complete','unavailable','cancelled','interrupted','failed'):
            raise ValueError('Saved overview status is unreadable. Request a new overview.')
        if status['state'] == 'running':
            if identifier != self.current or not self.active:
                self.state(folder, 'interrupted', 'Overview interrupted. Request a new overview; saved results remain available.')
                status = self.read(folder/'status.json')
            else:
                status['elapsed_seconds'] = round(time.monotonic()-self.started, 1)
                if self.cancel_event.is_set():
                    status.update(state='cancelling', message='Cancelling the owned overview request and stopping its CPU runtime…')
        result = {**request, **status, 'startup_budget_seconds':30, 'inference_budget_seconds':60}
        if status['state'] == 'complete':
            review = self.read(folder/'review.json')
            self.validate(review, request.get('timestamp'))
            result['review'] = review
        return result

    def cancel(self, identifier):
        self.folder(identifier)
        if identifier == self.current and self.active:
            self.cancel_event.set()
        return self.get(identifier)

    def close(self):
        if self.active:
            self.cancel_event.set()
