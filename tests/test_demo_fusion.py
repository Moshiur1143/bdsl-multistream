from pathlib import Path
import sys,json
import numpy as np,cv2
ROOT=Path(r'F:\ML');sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'src'))
from bilingual_labels import BilingualLabels
from app import create_app
labels=BilingualLabels(ROOT/'manifests/label_map.json',ROOT/'manifests/translations.json',require_reviewed=False)
ids=json.loads((ROOT/'manifests/label_map.json').read_text())
assert labels.display(ids['dhonnobad'],accepted=True)['bangla']=='ধন্যবাদ'
assert labels.display(ids['sotru'],accepted=True)['english']=='Meaning awaiting review'
try:BilingualLabels(ROOT/'manifests/label_map.json',ROOT/'manifests/translations.json')
except ValueError:pass
else:raise AssertionError('Strict mode accepted pending meanings')
app,extractor=create_app(ROOT/'models/fusion_v2/best.pt',review_mode=True)
try:
    client=app.test_client();assert client.get('/').status_code==200
    assert client.get('/status').json['pending_translations']==2
    assert client.post('/frame',data=b'invalid').status_code==400
    assert client.post('/reset',headers={'Origin':'https://unrelated.example'}).status_code==403
    _,encoded=cv2.imencode('.jpg',np.zeros((240,320,3),np.uint8));latencies=[]
    for _ in range(35):
        response=client.post('/frame',data=encoded.tobytes(),content_type='image/jpeg')
        assert response.status_code==200,response.data
        data=response.json;latencies.append(data['processing_ms']);assert data['class_id'] is None
        assert 'bangla' in data and 'english' in data
    assert data['buffer_frames']==32
    result=dict(synthetic_blank_frames=35,no_false_accepts=True,bilingual_mapping='passed',strict_review_guard='passed',
      review_placeholder='passed',origin_guard='passed',mean_processing_ms=float(np.mean(latencies)),
      note='Synthetic blank-frame API test only; not webcam accuracy or end-to-end FPS.')
    (ROOT/'results/fusion_demo_api_test.json').write_text(json.dumps(result,indent=2));print(result)
finally:extractor.close()

