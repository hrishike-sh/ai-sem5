"""Local application bridge for read-only reviews of completed incident exports."""
from collections import Counter
import base64
import hashlib
import json
from pathlib import Path
import re
import secrets
import threading
import traceback

from .jobs import now
from .media import write_json


class EvidenceReviews:
    """Review jobs live outside the processing-job scan and never hold its lock."""
    def __init__(self, root, executor, files):
        self.runs = Path(root)/'runs'
        self.root = self.runs/'evidence-reviews'
        self.executor, self.files = executor, files
        self.future = None
        self.current = None
        self.guard = threading.Lock()

    @staticmethod
    def identifier(value):
        if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,120}', value):
            raise ValueError('Invalid saved-run identifier.')
        return value

    def source(self, value):
        path = self.runs/self.identifier(value)
        if path.resolve().parent != self.runs.resolve():
            raise ValueError('Saved source is outside the run library.')
        return path

    @staticmethod
    def read(path, limit=16*1024*1024, expected_sha256=None):
        with Path(path).open('rb') as stream:
            raw = stream.read(limit+1)
        if len(raw)>limit:
            raise ValueError('Saved review exceeds the browser-view limit. Use its full JSON download.')
        if expected_sha256 is not None and hashlib.sha256(raw).hexdigest()!=expected_sha256:
            raise ValueError('Saved review bytes changed after their completion check.')
        return json.loads(raw)

    def state(self, folder, state, message, **extra):
        write_json(folder/'status.json', {'state':state, 'message':message, 'updated_at':now(), **extra})

    def start(self, source_id, use_model=False):
        if type(use_model) is not bool:
            raise ValueError('Local model selection must be true or false.')
        source = self.source(source_id)
        if not (source/'status.json').is_file() or self.read(source/'status.json').get('state')!='complete':
            raise ValueError('Choose a completed run with incident evidence.')
        with self.guard:
            if self.future is not None and not self.future.done():
                if self.current['source_run']==source_id and self.current['use_model']==use_model:
                    return self.get(self.current['id'])
                raise RuntimeError('An evidence review is already running. Its offline review remains available; wait before starting another.')
            identifier = secrets.token_hex(12)
            folder = self.root/identifier
            folder.mkdir(parents=True, exist_ok=False)
            self.current = {'id':identifier, 'source_run':source_id, 'use_model':use_model, 'created_at':now()}
            write_json(folder/'request.json', self.current)
            self.state(folder, 'queued', 'Preparing saved evidence review.')
            try:
                self.future = self.executor.submit(self.build, folder, source, use_model)
            except RuntimeError:
                self.state(folder, 'failed', 'The review service stopped. Reload and try again.')
                raise
        return self.get(identifier)

    @staticmethod
    def page_cursor(offset, status, fingerprint):
        if offset == 0:
            return None
        return base64.urlsafe_b64encode(json.dumps([offset,status,fingerprint],separators=(',',':')).encode()).decode().rstrip('=')

    @staticmethod
    def page_offset(cursor, status, fingerprint, count):
        if cursor is None:
            return 0
        try:
            if not isinstance(cursor,str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,256}',cursor):
                raise ValueError
            value=json.loads(base64.b64decode(cursor+'='*(-len(cursor)%4),altchars=b'-_',validate=True))
            if (not isinstance(value,list) or len(value)!=3 or type(value[0]) is not int
                    or value[0]<0 or value[0]%20 or value[0]>=count or value[1]!=status or value[2]!=fingerprint):
                raise ValueError
            return value[0]
        except (ValueError,TypeError,UnicodeError):
            raise ValueError('Invalid or outdated evidence page. Return to the first page.') from None

    def browser_view(self, document, artifact, cursor=None, status_filter='all', fingerprint=None):
        # Retain source export order and stable IDs. Only one bounded page of
        # cards, with no raw CSV samples, crosses into the browser.
        view = {key:document[key] for key in ('schema_version','kind','generated_at','source','summary','warnings','limitations')}
        if 'grounding' in document:
            view['grounding'] = document['grounding']
        selected=[(i,item) for i,item in enumerate(document['events'])
                  if status_filter=='all' or item['event']['status']==status_filter]
        offset=self.page_offset(cursor,status_filter,fingerprint,len(selected))
        cards=[]
        for index,item in selected[offset:offset+20]:
            card={k:v for k,v in item.items() if k not in ('track_context','stop_context')}
            card['source_position']=index+1
            for field in ('track_context','stop_context'):
                if field in item:
                    card[field]={k:v for k,v in item[field].items() if k!='samples'}
                    card['browser_'+field+'_samples_in_full_download']=item[field]['shown_samples']
            cards.append(card)
        view.update(events=cards, indexed_events=len(cards),
                    omitted_events=max(0,len(document['events'])-len(cards)), artifact=artifact,artifact_integrity_version=1)
        if fingerprint:
            count=len(selected)
            view['pagination']={'page_size':20,'start':offset+1 if count else 0,'end':min(offset+20,count),
                                'filtered_total':count,'total_events':len(document['events']),'status_filter':status_filter,
                                'order':'source_export','artifact_sha256':fingerprint,
                                'has_previous':offset>0,'has_next':offset+20<count,
                                'current_cursor':self.page_cursor(offset,status_filter,fingerprint),
                                'next_cursor':self.page_cursor(offset+20,status_filter,fingerprint) if offset+20<count else None,
                                'previous_cursor':self.page_cursor(offset-20,status_filter,fingerprint) if offset else None}
        return view

    def publish_view(self, folder, document, artifact):
        from traffic_ai.review_cli import validate_review_bundle
        marker=validate_review_bundle(folder/artifact)
        view=self.browser_view(document,artifact,fingerprint=marker['files']['review.json'])
        write_json(folder/'view.json', view)
        with self.guard:
            index = self.read(self.root/'index.json') if (self.root/'index.json').exists() else {}
            index[document['source']['run_id']] = folder.name
            write_json(self.root/'index.json', index)

    def build(self, folder, source, use_model):
        try:
            from traffic_ai.review_cli import export_review
            self.state(folder, 'running', 'Reading completed incident evidence.', phase='offline')
            document = export_review(source, folder/'offline')
            self.publish_view(folder, document, 'offline')
            if use_model:
                self.state(folder, 'running', 'Offline review ready. Asking the optional local model to prioritise existing facts and questions.', phase='local_model')
                try:
                    focused = export_review(source, folder/'focused', local_llm=True,
                                            max_llm_events=1, llm_timeout=45)
                    self.publish_view(folder, focused, 'focused')
                    document = focused
                except (OSError, ValueError, TypeError):
                    # A provider/provenance failure must not remove the already
                    # completed deterministic artifact or its source snapshot.
                    self.state(folder, 'complete', 'Offline review retained; optional local model focus was unavailable.', model_states={'unavailable':1})
                    return
            states = dict(Counter(e['language_model']['state'] for e in document['events']))
            message = ('Review ready. Local model focus unavailable; all offline facts and warnings are retained.'
                       if states.get('unavailable') else 'Saved evidence review ready.')
            self.state(folder, 'complete', message, model_states=states)
        except Exception:
            (folder/'worker.log').write_text(traceback.format_exc(), encoding='utf-8')
            self.state(folder, 'failed', 'Could not review this export. Check that incident evidence is complete and consistent; the original run is unchanged.')

    def integrity(self, identifier, source, manifest):
        try:
            for entry in manifest:
                path=source/entry['path']
                if path.resolve().parent!=source.resolve():
                    raise ValueError('Evidence escaped its source run.')
            matches=True
            for entry in manifest:
                path=source/entry['path']
                with path.open('rb') as stream:
                    matches &= path.stat().st_size==entry['bytes'] and hashlib.file_digest(stream,'sha256').hexdigest()==entry['sha256']
            result={'state':'verified' if matches else 'changed',
                    'message':'Consumed export fingerprints match this saved review.' if matches else 'Current source exports differ from this saved snapshot. Inspect the original run and regenerate the review.'}
            return result
        except (OSError, ValueError, KeyError):
            return {'state':'changed','message':'Some source evidence is missing or changed. This remains an inspectable saved snapshot.'}

    def get(self, identifier=None, source_id=None, cursor=None, status_filter='all'):
        if status_filter not in ('all','candidate','observation','withheld'):
            raise ValueError('Choose all, candidate, observation or withheld assessments.')
        if identifier is None:
            self.source(source_id)
            if self.current and self.current['source_run']==source_id and self.future and not self.future.done():
                identifier=self.current['id']
            else:
                index=self.read(self.root/'index.json') if (self.root/'index.json').exists() else {}
                identifier=index.get(source_id)
                if identifier is None:
                    return {'state':'none','source_run':source_id}
        folder=self.root/self.identifier(identifier)
        if folder.resolve().parent!=self.root.resolve() or not (folder/'request.json').is_file():
            raise ValueError('Saved evidence review was not found.')
        request=self.read(folder/'request.json')
        status=self.read(folder/'status.json')
        active = (self.current and self.current['id']==identifier and self.future is not None and not self.future.done())
        if status['state'] in ('queued','running') and not active:
            status=self.read(folder/'status.json')
            # The future may have published completion after our initial read.
            # Once it is done, rereading sees its final write before recovery.
            if status['state'] in ('queued','running'):
                self.state(folder,'interrupted','Review stopped before publishing its final status. Any completed offline artifact is retained; start a new review to retry.')
                status=self.read(folder/'status.json')
        response={'id':identifier,'source_run':request['source_run'],'use_model':request['use_model'],**status,'downloads':[]}
        if (folder/'view.json').is_file():
            view=self.read(folder/'view.json')
            if view['artifact'] not in ('offline','focused'):
                raise ValueError('Saved review artifact identifier is invalid.')
            artifact=folder/view['artifact']
            verified=False
            if not (artifact/'review-complete.json').is_file() and not view.get('artifact_integrity_version'):
                response['artifact_integrity']={'state':'legacy','message':'This older review predates review-file integrity records. Its saved snapshot remains inspectable; build a fresh review for verified downloads.'}
                view.pop('pagination',None)
                if cursor is not None or status_filter!='all':
                    raise ValueError('This legacy snapshot contains only its saved cards. Build a fresh review for page navigation.')
            else:
                try:
                    from traffic_ai.review_cli import validate_review_bundle
                    marker=validate_review_bundle(artifact)
                    fingerprint=marker['files']['review.json']
                    document=self.read(artifact/'review.json',limit=128*1024*1024,expected_sha256=fingerprint)
                    if document['source']['run_id']!=request['source_run']:
                        raise ValueError('Review and source identifiers disagree.')
                    # Derive browser cards from the validated full artifact,
                    # never trust an independently edited view cache as evidence.
                    response['artifact_integrity']={'state':'verified','message':'Saved review files match their completion record.'}
                    verified=True
                except (OSError,ValueError,KeyError,TypeError):
                    response['artifact_integrity']={'state':'changed','message':'Saved review files are changed or incomplete. Rebuild the review; the original processing result is unchanged.'}
                    response['message']=response['artifact_integrity']['message']
                    return response
                # Invalid navigation is an actionable request error, distinct
                # from a failed artifact-integrity check above.
                view=self.browser_view(document,view['artifact'],cursor,status_filter,fingerprint)
            source=self.source(request['source_run'])
            response['source_integrity']=self.integrity(identifier,source,view['source']['files'])
            for event in view['events']:
                event['thumbnail_notice']='No exported thumbnail is available for this assessment.'
                thumbnail=event.get('thumbnail_ref',{}).get('path')
                if isinstance(thumbnail,str) and re.fullmatch(r'[^/\\:]+\.png',thumbnail):
                    path=source/thumbnail
                    event['thumbnail_notice']='Recorded thumbnail is missing or changed; image withheld.'
                    if path.resolve().parent==source.resolve():
                        try:
                            with path.open('rb') as stream:
                                matches=(path.stat().st_size==event['thumbnail_ref']['bytes'] and
                                         hashlib.file_digest(stream,'sha256').hexdigest()==event['thumbnail_ref']['sha256'])
                            if matches:
                                event['thumbnail_url']=self.files.url(path)
                                event.pop('thumbnail_notice',None)
                        except (OSError,KeyError):
                            pass
            response['review']=view
            if verified:
                for name,label in [('review.json','Evidence review + source fingerprints (JSON)'),('review.html','Portable evidence review (HTML download)'),('review-complete.json','Review integrity record (JSON)')]:
                    path=artifact/name
                    if path.is_file():
                        response['downloads'].append({'name':name,'label':label,'url':self.files.url(path,True)})
        return response
