import olefile,struct,sys
def tracks(path):
    o=olefile.OleFileIO(path); d=o.openstream('Tracks6/Data').read(); i=0; out=[]
    while i<len(d):
        t=d[i]; n=struct.unpack('<I',d[i+1:i+5])[0]; rec=d[i+5:i+5+n]; i+=5+n
        if t!=4: continue
        layer=rec[0]; x1,y1,x2,y2,w=struct.unpack('<5i',rec[13:33])
        out.append((layer,round(x1/1e4,2),round(y1/1e4,2),round(x2/1e4,2),round(y2/1e4,2),round(w/1e4,2)))
    return out
for p in sys.argv[1:]:
    T=tracks(p)
    near=[t for t in T if (3100<t[1]<3200 and 1830<t[2]<1930) or (3100<t[3]<3200 and 1830<t[4]<1930) or (2985<t[1]<3115 and 1800<t[2]<1820) or (2985<t[3]<3115 and 1800<t[4]<1820)]
    print(p.split('/')[-1], len(T)); [print('  ',t) for t in sorted(near)]
