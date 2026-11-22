import re,sys,json,math,collections
G=sys.argv[1]; dump=json.load(open(sys.argv[2],encoding='utf-8'))
def parse(path):
    txt=open(path,encoding='latin-1').read()
    ap={}; cur=None; x=y=0; flashes=[]; regions=0; draws=0; inreg=False
    fs=re.search(r'%FSLAX(\d)(\d)Y',txt); dec=int(fs.group(2))
    for m in re.finditer(r'%ADD(\d+)([A-Za-z_0-9]+),?([^*]*)\*%',txt):
        ap[int(m.group(1))]=(m.group(2),m.group(3))
    body=re.sub(r'%[^%]*%','',txt)
    for cmd in body.split('*'):
        cmd=cmd.strip()
        if not cmd: continue
        if cmd=='G36': inreg=True; regions+=1; continue
        if cmd=='G37': inreg=False; continue
        m=re.match(r'^(?:G54)?D(\d+)$',cmd)
        if m and int(m.group(1))>=10: cur=int(m.group(1)); continue
        m=re.match(r'^(?:G0?1)?(?:X(-?\d+))?(?:Y(-?\d+))?D0?([123])$',cmd)
        if m:
            if m.group(1): x=int(m.group(1))/10**dec
            if m.group(2): y=int(m.group(2))/10**dec
            if m.group(3)=='3': flashes.append((x,y,cur))
            elif m.group(3)=='1': draws+=1
    return ap,flashes,regions,draws
L={}
for ext in ['GTL','GBL','GTS','GBS','GTP','GBP','GTO','GBO']:
    ap,fl,rg,dr=parse(f'{G}/SiPAP_RevC_v3.{ext}')
    L[ext]=(ap,fl)
    print(ext,'apertures',len(ap),'flashes',len(fl),'regions',rg,'draws',dr)
def near(fl,x,y,tol=0.02):
    return [f for f in fl if abs(f[0]-x)<tol and abs(f[1]-y)<tol]
# vias
vt=0;vbad=[]
for v in dump['vias']:
    x=float(v['X'])*0.0254; y=float(v['Y'])*0.0254
    t=bool(near(L['GTL'][1],x,y)); b=bool(near(L['GBL'][1],x,y))
    ms=near(L['GTS'][1],x,y)+near(L['GBS'][1],x,y)
    if not(t and b) or ms: vbad.append((v['Net'],round(x,2),round(y,2),t,b,len(ms)))
print('vias',len(dump['vias']),'problems',vbad[:10])
# pads
miss=collections.Counter(); ex=[]
for p in dump['pads']:
    x=float(p['X'])*0.0254; y=float(p['Y'])*0.0254
    smd=float(p['HoleSize'])==0
    need=['GTL','GTS','GTP'] if smd else ['GTL','GBL','GTS','GBS']
    for e in need:
        if not near(L[e][1],x,y):
            miss[e]+=1; ex.append((p['Component'],p['Name'],e))
    if not smd and near(L['GTP'][1],x,y): ex.append((p['Component'],p['Name'],'paste on THT'))
print('pads',len(dump['pads']),'missing',dict(miss)); print(ex[:40])
