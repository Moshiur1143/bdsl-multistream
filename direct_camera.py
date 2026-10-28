"""DirectShow camera fallback. Local only, frames never saved."""
import threading
import cv2
from flask import request, Response, jsonify
from app import create_app, ROOT
app, extractor = create_app(ROOT/'models/fusion_v2/best.pt', review_mode=True)
camera = None
camera_lock = threading.Lock()
@app.before_request
def protect_camera():
    if request.path.startswith('/camera/') and request.headers.get('X-BdSL-Camera') != 'local':
        return 'Camera requests must originate from the demo', 403
@app.post('/camera/frame')
def camera_frame():
    global camera
    with camera_lock:
        if camera is None:
            camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            camera.set(cv2.CAP_PROP_FRAME_WIDTH,1280)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT,720)
        ok, frame = camera.read()
        if not ok:
            camera.release(); camera = None
            return jsonify(error='DirectShow camera unavailable. Start OBS Virtual Camera.'), 503
        h,w = frame.shape[:2]
        scale=min(1.,960/max(h,w))
        if scale<1: frame=cv2.resize(frame,(round(w*scale),round(h*scale)))
        ok, encoded=cv2.imencode('.jpg',frame)
        if not ok:return 'JPEG encoding failed',500
        return Response(encoded.tobytes(),mimetype='image/jpeg',headers={'Cache-Control':'no-store'})
@app.post('/camera/stop')
def camera_stop():
    global camera
    with camera_lock:
        if camera is not None:camera.release();camera=None
    return jsonify(ok=True)
@app.get('/direct')
def direct():
    return app.send_static_file('direct.html')
if __name__=='__main__':
    try:app.run(host='127.0.0.1',port=5000,threaded=True,use_reloader=False)
    finally:
        if camera is not None:camera.release()
        extractor.close()
