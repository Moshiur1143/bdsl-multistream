"""Map live image-normalized graph coordinates into the trained 16:9 geometry.

Coordinates are image-width/height normalized by the detector. Multiplying x
by source_aspect / training_aspect preserves the pixel-space relationship to y.
Subsequent shoulder centering removes translation. Local hand coordinates
already have their own crop aspect correction; do not modify them.
"""
import numpy as np
TRAINING_ASPECT = 16 / 9

def align_live_graph(item, width, height):
    if width <= 0 or height <= 0:
        raise ValueError('Frame dimensions must be positive')
    factor=(width/height)/TRAINING_ASPECT
    result=dict(item)
    xy=item['graph_xy'].copy()
    xy[...,0]*=factor
    xy*=np.asarray(item['graph_mask'])[...,None]
    result['graph_xy']=xy
    return result
