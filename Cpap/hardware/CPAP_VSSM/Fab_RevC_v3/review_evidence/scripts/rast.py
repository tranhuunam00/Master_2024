# Minimal Gerber rasteriser (RS-274X subset used by Altium: C/R/O/RoundRect-macro flashes, D01 draws, G36 regions)
import re,sys,math
from PIL import Image,ImageDraw
def parse(path):
    txt=open(path,encoding='latin-1').read()
    fs=re.search(r'%FSLAX(\d)(\d)Y',txt); dec=int(fs.group(2))
    ap={}
    for m in re.finditer(r'%ADD(\d+)([A-Za-z_0-9]+),?([^*]*)\*%',txt):
        ap[int(m.group(1))]=(m.group(2),[float(v) for v in m.group(3).split('X') if v])
    body=re.sub(r'%[^%]*%','',txt)
    ops=[];cur=None;x=y=0;reg=None;interp=1
    for cmd in body.split('*'):
        cmd=cmd.strip()
        if not cmd: continue
        if cmd=='G36': reg=[];continue
        if cmd=='G37':
            if reg: ops.append(('R',reg))
            reg=None;continue
        if cmd in('G01','G1'): interp=1;continue
        m=re.match(r'^(?:G54)?D(\d+)$',cmd)
        if m and int(m.group(1))>=10: cur=int(m.group(1));continue
        m=re.match(r'^(G0?[123])?(?:X(-?\d+))?(?:Y(-?\d+))?(?:I(-?\d+))?(?:J(-?\d+))?D0?([123])$',cmd)
        if m:
            nx=int(m.group(2))/10**dec if m.group(2) else x
            ny=int(m.group(3))/10**dec if m.group(3) else y
            d=m.group(6)
            if reg is not None:
                if d=='2': reg.append([(nx,ny)])
                else:
                    if not reg: reg.append([(x,y)])
                    reg[-1].append((nx,ny))
            elif d=='3': ops.append(('F',nx,ny,ap.get(cur)))
            elif d=='1': ops.append(('D',x,y,nx,ny,ap.get(cur)))
            x,y=nx,ny
    return ops
def render(files,x0,y0,x1,y1,ppm,out,colors):
    W=int((x1-x0)*ppm);H=int((y1-y0)*ppm)
    im=Image.new('RGB',(W,H),(255,255,255));dr=ImageDraw.Draw(im,'RGBA')
    T=lambda x,y:((x-x0)*ppm,(y1-y)*ppm)
    for f,col in zip(files,colors):
        for op in parse(f):
            if op[0]=='R':
                for poly in op[1]:
                    if len(poly)>2: dr.polygon([T(*p) for p in poly],fill=col)
            elif op[0]=='F':
                _,x,y,a=op
                if not a: continue
                k,p=a; cx,cy=T(x,y)
                if k=='C': r=p[0]/2*ppm; dr.ellipse([cx-r,cy-r,cx+r,cy+r],fill=col)
                elif k in('R','O'): w=p[0]/2*ppm;h=p[1]/2*ppm; dr.rectangle([cx-w,cy-h,cx+w,cy+h],fill=col)
                else:
                    r=0.3*ppm; dr.rectangle([cx-r,cy-r,cx+r,cy+r],fill=col)
            else:
                _,xa,ya,xb,yb,a=op
                w=a[1][0] if a and a[1] else 0.1
                dr.line([T(xa,ya),T(xb,yb)],fill=col,width=max(1,int(w*ppm)))
    # grid every 1 mm
    for gx in range(int(x0),int(x1)+1):
        X=(gx-x0)*ppm; dr.line([(X,0),(X,H)],fill=(0,0,255,40) if gx%5 else (0,0,255,110))
    for gy in range(int(y0),int(y1)+1):
        Y=(y1-gy)*ppm; dr.line([(0,Y),(W,Y)],fill=(0,0,255,40) if gy%5 else (0,0,255,110))
    im.save(out)
if __name__=='__main__':
    G=sys.argv[1]; x0,y0,x1,y1=map(float,sys.argv[2].split(','))
    render([G+'/SiPAP_RevC_v3.GTL',G+'/SiPAP_RevC_v3.GTO'],x0,y0,x1,y1,float(sys.argv[3]),sys.argv[4],[(200,40,40,150),(0,0,0,255)])
