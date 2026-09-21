"""Shared, machine-readable boundary between appearance and configured rules."""


def review_only_policy():
    return {'status': 'review_only', 'semantic_class': None,
            'eligible_for_geometry': False, 'eligible_for_rules': False,
            'reason': 'Paint-like appearance is not verified road, crossing or stop-line geometry.'}


def is_review_only(value):
    """Keep explicit appearance/abstention records out of geometry consumers."""
    return isinstance(value, dict) and (
        value.get('status') == 'review_only' or value.get('eligible_for_geometry') is False
        or value.get('eligible_for_rules') is False)


def configured_geometry_policy():
    """Explicit configuration contract, not a legal or accuracy certification."""
    return {'status': 'configured_geometry', 'configuration_authority': 'explicit_configuration',
            'eligible_for_geometry': True, 'eligible_for_rules': True}


def geometry_is_configured(value):
    # Missing/legacy flags are not authority. Appearance results never receive
    # this contract; independent explicit scene/incident configuration has its
    # own validated input path. A record cannot opt in with a lone true flag.
    return isinstance(value, dict) and all(
        value.get(key) == expected and type(value.get(key)) is type(expected)
        for key, expected in configured_geometry_policy().items())


def mask_export_manifest(names, source_binding):
    return {'schema_version': 1, 'kind': 'ivp_appearance_mask_export',
            **review_only_policy(), 'selected_masks': list(names),
            'source_binding': dict(source_binding),
            'note': 'Image features are for inspection. No mask configures semantic road geometry or incident rules.'}
