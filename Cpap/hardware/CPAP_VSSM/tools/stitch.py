import json,math,sys
d=json.load(open(sys.argv[1],encoding='utf-8'))
tr,vi,pa=[it['objects'] for it in d['items']]
R=12; CL=15
def seg_dist(px,py,x1,y1,x2,y2):
    dx,dy=x2-x1,y2-y1; L=dx*dx+dy*dy
    t=0 if L==0 else max(0,min(1,((px-x1)*dx+(py-y1)*dy)/L))
    return math.hypot(px-(x1+t*dx),py-(y1+t*dy))
f=float
pts=[]
for gx in range(100,4000,200):
  for gy in range(100,3000,200):
    x,y=gx,gy
    if x<90 or y<90 or x>3910 or y>2910: continue
    if any(math.hypot(x-hx,y-hy)<118+R+30 for hx,hy in [(150,150),(3850,150),(150,2850),(3850,2850)]): continue
    if 1600<=x<=2400 and 1950<=y<=3000: continue          # under ESP32 module + antenna
    ok=True
    for t in tr:
        if t['Layer'] not in ('TopLayer','BottomLayer'): continue
        if t.get('Net')=='GND': continue
        if seg_dist(x,y,f(t['X1']),f(t['Y1']),f(t['X2']),f(t['Y2'])) < f(t['Width'])/2+R+CL: ok=False;break
    if not ok: continue
    for v in vi:
        if math.hypot(x-f(v['X']),y-f(v['Y'])) < f(v['Size'])/2+R+CL: ok=False;break
    if not ok: continue
    for p in pa:
        l,b,r,t_=f(p['BoundsLeft']),f(p['BoundsBottom']),f(p['BoundsRight']),f(p['BoundsTop'])
        dx=max(l-x,0,x-r); dy=max(b-y,0,y-t_)
        if math.hypot(dx,dy) < R+CL+5: ok=False;break
    if ok: pts.append({"x":x,"y":y,"size":24,"hole_size":12,"net":"GND"})
json.dump(pts,open('stitch_vias.json','w'))
print(len(pts), len(tr), len(vi), len(pa))
