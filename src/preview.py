import sys, time
sys.path.insert(0, '/home/user/Pharmnight/src')
import cinema, cv2, numpy as np
cinema.init('/home/user/Pharmnight/assets/logo_original.webp')
times = [float(x) for x in sys.argv[1].split(',')]
out = sys.argv[2]
tiles = []
for t in times:
    t0 = time.time()
    f = cinema.render_frame(t)
    print(f't={t:.2f} {time.time()-t0:.2f}s', flush=True)
    sm = cv2.resize(f, (960, 540), interpolation=cv2.INTER_AREA)
    cv2.putText(sm, f'{t:.2f}s', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    tiles.append(sm)
cols = 2
while len(tiles) % cols: tiles.append(np.zeros_like(tiles[0]))
rows = [np.hstack(tiles[i:i+cols]) for i in range(0, len(tiles), cols)]
cv2.imwrite(out, cv2.cvtColor(np.vstack(rows), cv2.COLOR_RGB2BGR))
