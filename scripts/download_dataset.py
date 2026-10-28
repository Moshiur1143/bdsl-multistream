"""Persistent, resumable download. Never removes a partial archive on failure."""
import argparse,hashlib,json,time,urllib.request,urllib.error,zipfile,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
URL='https://zenodo.org/records/6779843/files/SignBD-Word.zip?download=1'
MD5='fd0ca299610be2ba37de555eca52525b'

def verify(path):
    digest=hashlib.md5()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): digest.update(b)
    return digest.hexdigest()==MD5

def main():
    dest=ROOT/'dataset';dest.mkdir(exist_ok=True)
    archive=dest/'SignBD-Word.zip'; part=dest/'SignBD-Word.zip.part'
    if archive.exists():
        if not verify(archive): raise RuntimeError('Existing ZIP checksum differs. Preserved for inspection.')
        print('Existing archive verified. No download needed.',flush=True);return
    for attempt in range(8):
        start=part.stat().st_size if part.exists() else 0
        if shutil.disk_usage(dest).free<max(0,25*1024**3-start): raise RuntimeError('Insufficient disk space')
        req=urllib.request.Request(URL,headers={'Range':f'bytes={start}-','User-Agent':'BdSL-University-Project/1.0'})
        try:
            with urllib.request.urlopen(req,timeout=90) as response:
                if start and (response.status!=206 or not response.headers.get('Content-Range','').startswith(f'bytes {start}-')):
                    raise RuntimeError('Server did not resume correctly; partial file preserved.')
                total=start+int(response.headers.get('Content-Length',0)); last=time.monotonic(); count=start
                with part.open('ab') as f:
                    while True:
                        block=response.read(1024*1024)
                        if not block:break
                        f.write(block);count+=len(block)
                        if time.monotonic()-last>=15:
                            f.flush(); status=dict(bytes=count,total_bytes=total,percent=round(100*count/total,2) if total else None)
                            (dest/'download_progress.json').write_text(json.dumps(status,indent=2))
                            print(f'{count/1024**3:.2f}/{total/1024**3:.2f} GiB',flush=True);last=time.monotonic()
            if total and count!=total: raise OSError('Incomplete response')
            break
        except urllib.error.HTTPError as error:
            if error.code==416 and part.exists() and verify(part): break
            if attempt==7:raise
            print(f'Retry {attempt+1}: {error}',flush=True);time.sleep(min(60,5*(attempt+1)))
        except (OSError,TimeoutError) as error:
            if attempt==7:raise
            print(f'Retry {attempt+1}: {error}',flush=True);time.sleep(min(60,5*(attempt+1)))
    print('Verifying checksum...',flush=True)
    if not verify(part):raise RuntimeError('Checksum mismatch; partial file preserved, not extracted.')
    part.replace(archive)
    print('DOWNLOAD VERIFIED: '+str(archive),flush=True)
if __name__=='__main__':main()
