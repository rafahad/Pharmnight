import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import render, cv2, numpy as np
render.init()
times = [float(x) for x in sys.argv[1].split(',')]
tiles = []
for t in times:
    t0 = time.time(); f = render.render_frame(t); print(f't={t:.2f} {time.time()-t0:.2f}s', flush=True)
    sm = cv2.resize(f, (360, 640), interpolation=cv2.INTER_AREA)
    cv2.putText(sm, f'{t:.2f}', (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    tiles.append(sm)
cols = 6 if len(tiles) > 6 else len(tiles)
while len(tiles) % cols: tiles.append(np.zeros_like(tiles[0]))
cv2.imwrite(sys.argv[2], cv2.cvtColor(np.vstack([np.hstack(tiles[i:i+cols]) for i in range(0, len(tiles), cols)]), cv2.COLOR_RGB2BGR))
