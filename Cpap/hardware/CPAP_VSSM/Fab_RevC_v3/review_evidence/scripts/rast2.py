# Gerber rasteriser with layer polarity (%LPD%/%LPC%), regions, C/R/O flashes, D01 draws with round caps.
import re
import numpy as np
from PIL import Image,ImageDraw
def ops(path):
    txt=open(path,encoding='latin-1').read()
    dec=int(re.search(r'%FSLAX(\d)(\d)Y',txt).group(2))
    ap={}
    for m in re.finditer(r'%ADD(\d+)([A-Za-z_0-9]+),?([^*]*)\*%',txt):
        ap[int(m.group(1))]=(m.group(2),[float(v) for v in m.group(3).split('X') if v])
    toks=re.findall(r'%[^%]*%|[^*%]+\*',txt)
    cur=None;x=y=0;reg=None;pol='D';out=[]
    for t in toks:
        if t.startswith('%'):
            if t.startswith('%LPC'): pol='C'
            elif t.startswith('%LPD'): pol='D'
            continue
        c=t[:-1].strip()
        if not c: continue
        if c=='G36': reg=[];continue
        if c=='G37':
            if reg: out.append(('R',pol,reg))
            reg=None;continue
        m=re.match(r'^(?:G54)?D(\d+)$',c)
        if m and int(m.group(1))>=10: cur=int(m.group(1));continue
        m=re.match(r'^(?:G0?1)?(?:X(-?\d+))?(?:Y(-?\d+))?D0?([123])$',c)
        if not m: continue
        nx=int(m.group(1))/10**dec if m.group(1) else x
        ny=int(m.group(2))/10**dec if m.group(2) else y
        d=m.group(3)
        if reg is not None:
            if d=='2': reg.append([(nx,ny)])
            else:
                if not reg: reg.append([(x,y)])
                reg[-1].append((nx,ny))
        elif d=='3': out.append(('F',pol,nx,ny,ap.get(cur)))
        elif d=='1': out.append(('D',pol,x,y,nx,ny,ap.get(cur)))
        x,y=nx,ny
    return out
def raster(path,box,ppm):
    x0,y0,x1,y1=box; W=int((x1-x0)*ppm);H=int((y1-y0)*ppm)
    im=Image.new('1',(W,H),0); dr=ImageDraw.Draw(im)
    T=lambda x,y:((x-x0)*ppm,(y1-y)*ppm)
    for o in ops(path):
        col=1 if o[1]=='D' else 0
        if o[0]=='R':
            for poly in o[2]:
                if len(poly)>2: dr.polygon([T(*p) for p in poly],fill=col)
        elif o[0]=='F':
            _,_,x,y,a=o
            if not a: continue
            k,p=a;cx,cy=T(x,y)
            if k=='C': r=p[0]/2*ppm; dr.ellipse([cx-r,cy-r,cx+r,cy+r],fill=col)
            elif k in('R','O'): w=p[0]/2*ppm;h=p[1]/2*ppm; dr.rectangle([cx-w,cy-h,cx+w,cy+h],fill=col)
            else: r=0.3*ppm; dr.rectangle([cx-r,cy-r,cx+r,cy+r],fill=col)
        else:
            _,_,xa,ya,xb,yb,a=o; w=a[1][0] if a and a[1] else 0.1
            dr.line([T(xa,ya),T(xb,yb)],fill=col,width=max(1,int(round(w*ppm))))
            r=w/2*ppm
            for (x,y) in ((xa,ya),(xb,yb)):
                cx,cy=T(x,y); dr.ellipse([cx-r,cy-r,cx+r,cy+r],fill=col)
    return np.array(im,dtype=bool)
