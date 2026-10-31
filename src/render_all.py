"""Render all 432 frames of the 4K master in parallel to lossless RGB segments."""
import sys, os, subprocess, time
from multiprocessing import Process
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cinema

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGO = os.path.join(ROOT, 'assets', 'logo_original.webp')
OUT = os.path.join(ROOT, 'build')
NW = int(os.environ.get('WORKERS', '4'))


def worker(k, a, b):
    cinema.init(LOGO)
    seg = os.path.join(OUT, f'seg_{k:02d}.mkv')
    cmd = ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
           '-s', f'{cinema.W}x{cinema.H}', '-r', str(cinema.FPS), '-i', '-',
           '-c:v', 'libx264rgb', '-qp', '0', '-preset', 'ultrafast', seg]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    t0 = time.time()
    for i in range(a, b):
        p.stdin.write(cinema.render_frame(i / cinema.FPS).tobytes())
        if (i - a) % 10 == 0:
            print(f'[w{k}] frame {i} ({i - a + 1}/{b - a}) {time.time() - t0:.0f}s', flush=True)
    p.stdin.close(); p.wait()
    print(f'[w{k}] done', flush=True)


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    n = cinema.NF
    bounds = [round(n * k / NW) for k in range(NW + 1)]
    ps = [Process(target=worker, args=(k, bounds[k], bounds[k + 1])) for k in range(NW)]
    for p in ps: p.start()
    for p in ps: p.join()
    with open(os.path.join(OUT, 'segments.txt'), 'w') as f:
        for k in range(NW):
            f.write(f"file 'seg_{k:02d}.mkv'\n")
    print('ALL DONE')
