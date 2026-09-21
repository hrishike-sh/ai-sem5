"""Truthful execution metadata for fixed-default frame detector preparation."""
import numpy as np


def preparation_record(original,result,settings,binding):
    flags={'denoise':settings.denoise!='Off','white_balance':bool(settings.white_balance),
           'exposure':bool(settings.exposure),'percentile_stretch':bool(settings.stretch),
           'gamma':settings.gamma!=1,'clahe':settings.clahe,'sharpen':bool(settings.sharpen),
           'saturation':settings.saturation!=1}
    return {'schema_version':1,'source_binding':dict(binding),'settings':settings.to_dict(),
            'enabled_colour_operations':[k for k,v in flags.items() if v],
            'disabled_or_identity_settings':[k for k,v in flags.items() if not v],
            'input_size':[original.shape[1],original.shape[0]],
            'analysis_size':[result.color_rgb.shape[1],result.color_rgb.shape[0]],
            'resize_applied':original.shape!=result.color_rgb.shape,
            'output_equals_display_oriented_input':bool(np.array_equal(original,result.color_rgb)),
            'feature_extraction_executed':False,'stage_capture_executed':False,
            'quality_diagnostics':'Computed separately; not inputs to the detector/geometry decision.',
            'contribution_limit':'Enabled operations alone do not establish a visible per-operation contribution. No Canny/HSV paint/threshold/morphology branch ran in this frame check.'}
