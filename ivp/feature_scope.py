"""Validate explicit UI feature identity without altering traffic evaluators."""
FEATURES={'tracking','zebra','direction','red_line','restricted_red','speed_review','required_stop'}


def validate_feature_scope(feature,motion):
    if feature is None:return
    if not isinstance(feature,str) or feature not in FEATURES:
        raise ValueError('Unknown traffic feature identity.')
    if motion.get('speed_display') is not None and feature not in {'tracking','speed_review'}:
        raise ValueError('Speed display settings belong only to tracking or speed review.')
    incident=motion.get('incidents') or {}
    active=bool(incident.get('enabled'))
    if feature in {'tracking','zebra','direction'}:
        if active or bool(motion.get('zebra'))!=(feature=='zebra') or bool(motion.get('direction'))!=(feature=='direction'):
            raise ValueError('This feature cannot include settings from another traffic feature.')
        return
    if not active or motion.get('zebra') or motion.get('direction'):
        raise ValueError('Configure only the selected incident feature.')
    line,zone,approach,limit=(incident.get(k) for k in ['line','zone','approach_zone','speed_limit_kmh'])
    if feature=='red_line':valid=bool(line) and zone is None and approach is None and limit is None
    elif feature=='restricted_red':valid=line is None and bool(zone) and approach is None and limit is None and incident.get('zone_role')=='restricted'
    elif feature=='speed_review':valid=line is None and zone is None and approach is None and limit is not None and bool(motion.get('calibration'))
    else:valid=bool(line) and zone is None and bool(approach) and limit is None and bool(motion.get('calibration'))
    if not valid:raise ValueError('Hidden settings from another feature are not permitted; configure the selected feature only.')
    allowed={'observe'} if feature=='speed_review' else {'stop_required','unknown'} if feature=='required_stop' else {'red','green','unknown'}
    if incident.get('condition') not in allowed or any(v['state'] not in allowed for v in incident.get('condition_intervals',[])):
        raise ValueError('Configured conditions do not belong to the selected feature.')
