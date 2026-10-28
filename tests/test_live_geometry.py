import sys,csv,json
from pathlib import Path
import numpy as np
ROOT=Path(r'F:\ML');sys.path.insert(0,str(ROOT))
from src.live_geometry import align_live_graph
from src.data_v2 import normalize_v2
rows=list(csv.DictReader((ROOT/'manifests/feature_manifest_v2.csv').open()))
report=[]
for gloss in ['khabar','boi','lekha','baba','dhonnobad']:
 row=next(r for r in rows if r['gloss']==gloss and r['split']=='train')
 with np.load(ROOT/row['feature_path'],allow_pickle=False) as f:a={k:f[k].copy() for k in ['graph_xy','graph_mask','face_rgb','face_mask','frame_mask','unique_frame_mask','hand_local_xyz']}
 original=normalize_v2(a)
 same=align_live_graph(a,960,540);np.testing.assert_array_equal(same['graph_xy'],a['graph_xy'])
 # Simulate the same pixel-space pose in a portrait coordinate system.
 portrait=dict(a);portrait['graph_xy']=a['graph_xy'].copy();portrait['graph_xy'][...,0]*=(16/9)/(9/16)
 raw=normalize_v2(portrait);fixed=normalize_v2(align_live_graph(portrait,540,960))
 np.testing.assert_allclose(fixed['graph'],original['graph'],atol=2e-5)
 np.testing.assert_array_equal(fixed['face'],original['face'])
 np.testing.assert_array_equal(portrait['hand_local_xyz'],a['hand_local_xyz'])
 report.append(dict(gloss=gloss,unfixed_max_graph_error=float(abs(raw['graph']-original['graph']).max()),fixed_max_graph_error=float(abs(fixed['graph']-original['graph']).max())))
print(json.dumps(report,indent=2));Path('outputs/live_geometry_test.json').write_text(json.dumps(report,indent=2))
