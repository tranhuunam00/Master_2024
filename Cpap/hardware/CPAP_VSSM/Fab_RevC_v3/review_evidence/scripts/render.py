import sys,fitz
doc=fitz.open(sys.argv[1]); pg=int(sys.argv[2])-1
clip=None
if len(sys.argv)>5:
    x0,y0,x1,y1=map(float,sys.argv[5].split(','))
    r=doc[pg].rect; clip=fitz.Rect(r.width*x0,r.height*y0,r.width*x1,r.height*y1)
pix=doc[pg].get_pixmap(dpi=int(sys.argv[4]),clip=clip); pix.save(sys.argv[3]); print(sys.argv[3])
