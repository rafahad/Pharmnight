"""Render all frames (2160x3840 @ 60 fps) in parallel to lossless RGB segments in build_psa/."""
import os, sys, subprocess, time
from multiprocessing import Process
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import timeline as TL

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, 'build_psa')
NW = int(os.environ.get('WORKERS', '4'))


def worker(k, a, b):
    import render
    render.init()
    seg = os.path.join(OUT, f'seg_{k:02d}.mkv')
    p = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
                          '-s', f'{TL.W}x{TL.H}', '-r', str(TL.FPS), '-i', '-',
                          '-c:v', 'libx264rgb', '-qp', '0', '-preset', 'ultrafast', seg], stdin=subprocess.PIPE)
    t0 = time.time()
    for i in range(a, b):
        p.stdin.write(render.render_frame(i / TL.FPS).tobytes())
        if (i - a) % 40 == 0:
            print(f'[w{k}] {i - a + 1}/{b - a} {time.time() - t0:.0f}s', flush=True)
    p.stdin.close(); p.wait()
    print(f'[w{k}] done', flush=True)


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    bounds = [round(TL.NF * k / NW) for k in range(NW + 1)]
    ps = [Process(target=worker, args=(k, bounds[k], bounds[k + 1])) for k in range(NW)]
    for p in ps: p.start()
    for p in ps: p.join()
    with open(os.path.join(OUT, 'segments.txt'), 'w') as f:
        for k in range(NW):
            f.write(f"file 'seg_{k:02d}.mkv'\n")
    print('ALL DONE')
