import sys
sys.path.insert(0, '/home/user/Pharmnight/src')
import cinema, cv2, numpy as np
cinema.init('/home/user/Pharmnight/assets/logo_original.webp')
times = [float(x) for x in sys.argv[1].split(',')]
tiles = []
for t in times:
    f = cinema.render_frame(t)
    lx, ly, ls = cinema.logo_screen(cinema.Cam(t))
    r = int(700 * ls)
    c = f[max(int(ly) - r, 0):int(ly) + r, max(int(lx) - r, 0):int(lx) + r]
    c = cv2.resize(c, (800, 800), interpolation=cv2.INTER_AREA)
    cv2.putText(c, f'{t:.2f}s', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    tiles.append(c)
while len(tiles) % 3: tiles.append(np.zeros_like(tiles[0]))
rows = [np.hstack(tiles[i:i+3]) for i in range(0, len(tiles), 3)]
cv2.imwrite(sys.argv[2], cv2.cvtColor(np.vstack(rows), cv2.COLOR_RGB2BGR))
