"""Conservative grouped-proposal diagnostic for optional recovery gates.

The main detector and ByteTrack passes never discard boxes using this helper.
"""
import numpy as np


def grouped_proposals(rows):
    """Flag a larger box explained by two distinct, stronger sub-boxes."""
    rows=np.asarray(rows)
    if len(rows)<3:return set()
    boxes=rows[:,:4];sizes=boxes[:,2:]-boxes[:,:2];areas=np.prod(sizes,axis=1)
    inter=np.prod(np.maximum(0,np.minimum(boxes[:,None,2:],boxes[None,:,2:])-np.maximum(boxes[:,None,:2],boxes[None,:,:2])),axis=2)
    overlaps=inter/np.maximum(1,areas[:,None]+areas[None,:]-inter)
    reject=set()
    for parent in range(len(rows)):
        children=np.flatnonzero((inter[parent]/np.maximum(1,areas)>.85)&(areas<areas[parent]*.8)&(rows[:,4]>=.15))
        for offset,a in enumerate(children):
            for b in children[offset+1:]:
                if (overlaps[a,b]<.35 and (inter[parent,a]+inter[parent,b]-inter[a,b])/areas[parent]>.7
                        and rows[a,4]+rows[b,4]>rows[parent,4]
                        and abs(boxes[a,3]-boxes[b,3])>.2*min(sizes[a,1],sizes[b,1])):
                    reject.add(parent)
    return reject
