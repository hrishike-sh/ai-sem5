"""Paged saved-run metadata, independent of large processing/evidence payloads."""
import json
from pathlib import Path

from .jobs import refreshed_status
from .workspace import source_name


class RunLibrary:
    def __init__(self, root):
        self.root=Path(root)
        self.names={}

    def name(self, run):
        signature=[]
        for filename in ('request.json','report.json'):
            try:
                stat=(run/filename).stat()
                signature.append((filename,stat.st_size,stat.st_mtime_ns))
            except OSError:
                pass
        cached=self.names.get(run.name)
        if cached and cached[0]==signature:
            return cached[1]
        name='Unknown source'
        # Video requests are small even when their final reports contain many
        # events. Do not open that report just to populate one library row.
        for filename,size,_ in signature:
            if size>64*1024*1024:
                continue
            try:
                with (run/filename).open(encoding='utf-8') as stream:
                    details=json.load(stream)
                if isinstance(details,dict) and isinstance(details.get('source'),str):
                    name=source_name(details)[:512]
                    break
            except (OSError,ValueError,UnicodeError):
                pass
        if len(self.names)>=2048:
            self.names.pop(next(iter(self.names)),None)
        self.names[run.name]=(signature,name)
        return name

    def page(self, query='', state='', before='', limit=30):
        if not isinstance(query,str) or len(query)>200:
            raise ValueError('Search text must be at most 200 characters.')
        if state not in ('','queued','running','encoding','complete','failed','cancelled','interrupted','unknown'):
            raise ValueError('Choose a supported run status.')
        if (not isinstance(before,str) or len(before)>180 or
                before and (Path(before).name!=before or before in ('.','..') or any(c in before for c in '/\\\x00:'))):
            raise ValueError('Invalid saved-run page cursor.')
        if type(limit) is not int or not 1<=limit<=100:
            raise ValueError('Use a saved-run page size between 1 and 100.')
        query=query.strip().casefold()
        matches=[]
        for path in sorted(self.root.glob('*/status.json'),key=lambda p:p.parent.name,reverse=True):
            run=path.parent
            if run.resolve().parent!=self.root.resolve() or before and run.name>=before:
                continue
            name=self.name(run)
            if query and query not in (name+' '+run.name).casefold():
                continue
            status=refreshed_status(run)
            if state and status.get('state')!=state:
                continue
            summary={k:status[k] for k in ('state','message','updated_at') if k in status}
            if isinstance(summary.get('message'),str) and len(summary['message'])>1000:
                summary['message']=summary['message'][:997]+'…'
            matches.append({'id':run.name,'source':name,'status':summary})
            if len(matches)>limit:
                break
        more=len(matches)>limit
        return {'runs':matches[:limit],'has_more':more,
                'next_before':matches[limit-1]['id'] if more else None}
