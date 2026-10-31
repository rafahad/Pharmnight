import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cinema, cv2, numpy as np
cinema.configure(os.environ.get('LAYOUT', 'landscape'))
cinema.init(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'assets', 'logo_original.webp'))
times = [float(x) for x in sys.argv[1].split(',')]
out = sys.argv[2]
tw, th = (960, 540) if cinema.W > cinema.H else (432, 768)
cols = 2 if cinema.W > cinema.H else len(times)
tiles = []
for t in times:
    t0 = time.time()
    f = cinema.render_frame(t)
    print(f't={t:.2f} {time.time()-t0:.2f}s', flush=True)
    sm = cv2.resize(f, (tw, th), interpolation=cv2.INTER_AREA)
    cv2.putText(sm, f'{t:.2f}s', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    tiles.append(sm)
while len(tiles) % cols: tiles.append(np.zeros_like(tiles[0]))
rows = [np.hstack(tiles[i:i+cols]) for i in range(0, len(tiles), cols)]
cv2.imwrite(out, cv2.cvtColor(np.vstack(rows), cv2.COLOR_RGB2BGR))
