import sys, numpy as np, rast2
A,B=sys.argv[1],sys.argv[2]; ppm=20; box=(-1,-1,102.6,77.2)
for e in ['GTL','GBL','GTO','GTS','GTP','GBS','GM']:
    a=rast2.raster(f'{A}/SiPAP_RevC_v3.{e}',box,ppm); b=rast2.raster(f'{B}/SiPAP_RevC_v3.{e}',box,ppm)
    d=a^b; n=int(d.sum())
    if n==0: print(e,'identical'); continue
    ys,xs=np.nonzero(d); xm=xs/ppm+box[0]; ym=box[3]-ys/ppm
    cells={}
    for X,Y in zip(xm,ym):
        k=(int(X//4)*4,int(Y//4)*4); cells[k]=cells.get(k,0)+1
    print(e,f'{n/ppm**2:.2f} mm2 differ, bbox x {xm.min():.1f}-{xm.max():.1f} y {ym.min():.1f}-{ym.max():.1f}; 4mm cells:',
          sorted(((k,round(v/ppm**2,2)) for k,v in cells.items()), key=lambda kv:-kv[1])[:10])
