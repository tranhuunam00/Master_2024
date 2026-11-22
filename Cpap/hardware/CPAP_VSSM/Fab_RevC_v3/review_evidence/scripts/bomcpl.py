import csv,sys,json,re
F=sys.argv[1]; dump=sys.argv[2]
bom=list(csv.DictReader(open(F+'/SiPAP_RevC_v3_BOM_JLC.csv',encoding='utf-8-sig')))
cpl=list(csv.DictReader(open(F+'/SiPAP_RevC_v3_CPL_JLC.csv',encoding='utf-8-sig')))
gcpl=list(csv.DictReader(open(F+'/SiPAP_RevC_v3_CPL.csv',encoding='utf-8-sig')))
mbom=list(csv.DictReader(open(F+'/BOM_RevC_v3.csv',encoding='utf-8-sig')))
bd=set(); 
for r in bom:
    ds=[d.strip() for d in r['Designator'].split(',')]
    if len(ds)!=int(r['Quantity']): print('qty mismatch',r['Designator'],r['Quantity'])
    bd|=set(ds)
cd={r['Designator'] for r in cpl}
print('JLC BOM',len(bd),'CPL',len(cd),'onlyBOM',bd-cd,'onlyCPL',cd-bd)
d=json.load(open(dump,encoding='utf-8'))
C={c['Designator']:c for c in d['comps']}
print('dump comps',len(C),'generic CPL rows',len(gcpl))
# compare positions
bad=0
for r in cpl:
    c=C[r['Designator']]
    x=float(r['Mid X'].rstrip('mm')); y=float(r['Mid Y'].rstrip('mm'))
    dx=x-float(c['X'])*0.0254; dy=y-float(c['Y'])*0.0254
    if abs(dx)>0.01 or abs(dy)>0.01 or float(r['Rotation'])!=float(c['Rotation']):
        print('CPL vs dump',r['Designator'],round(dx,3),round(dy,3),r['Rotation'],c['Rotation']); bad+=1
print('cpl/dump diffs',bad)
# main BOM designators
md=set()
for r in mbom:
    s=r['Designators']
    for tok in s.split():
        m=re.match(r'([A-Z]+)(\d+)-[A-Z]*(\d+)$',tok)
        if m:
            for i in range(int(m.group(2)),int(m.group(3))+1): md.add(m.group(1)+str(i))
        else: md.add(tok)
    n=len([t for t in s.split()])
print('main BOM refs',len(md),'missing vs dump',set(C)-md,'extra',md-set(C))
# comment check: dump comment vs JLC comment
for r in bom:
    for ds in r['Designator'].split(','):
        ds=ds.strip(); cm=C[ds]['Comment']
        if r['Comment'].split()[0].lower() not in cm.lower(): print('comment?',ds,repr(cm),'vs',r['Comment'])
