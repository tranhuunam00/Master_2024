import json,sys,re,collections
L="C:/Users/ADMIN/OneDrive/Máy tính/Master_2024/Cpap/hardware/CPAP_VSSM/tools/layout_v3/"
d=json.load(open(L+'board_dump_v3_final.json',encoding='utf-8'))
for p in d['pads']:
    if p['Component']=='R10' and p['Name']=='1': p['Net']='REF_TOP'
for c in d['comps']:
    if c['Designator']=='R10': c['Comment']='36k'
d['comps']+= [
 {"Designator":"RV1","Comment":"50k","Pattern":"BOURNS_3224W","X":"1310","Y":"720","Rotation":"90","Layer":"TopLayer"},
 {"Designator":"R30","Comment":"100k","Pattern":"R_0805","X":"710","Y":"470","Rotation":"0","Layer":"TopLayer"}]
def pad(c,n,net,x,y,sx,sy,rot):
    return {"Component":c,"Name":n,"Net":net,"X":str(x),"Y":str(y),"TopXSize":str(sx),"TopYSize":str(sy),"HoleSize":"0","HoleType":"round","Rotation":str(rot),"Layer":"TopLayer"}
d['pads']+= [pad('RV1','2','REF_TOP',1310,662.9134,62.9921,78.7402,90), pad('RV1','1','5V',1359.2126,777.0866,62.9921,51.1811,90),
             pad('RV1','3','REF_TOP',1260.7874,777.0866,62.9921,51.1811,90),
             pad('R30','1','BLOWER_VSR',672.5984,470,39.3701,49.2126,0), pad('R30','2','GND',747.4016,470,39.3701,49.2126,0)]
d['vias']+= [{"Net":n,"X":str(x),"Y":str(y),"Size":"24","HoleSize":"12"} for n,x,y in
             (('5V',1400,940),('5V',1359.2126,855),('REF_TOP',1340,1002.5984),('REF_TOP',1260.7874,855.7874))]
json.dump(d,open(L+'board_dump_v3_rv1.json','w',encoding='utf-8'),indent=1)
# cross-check against the IPC-356 Altium just wrote
ipc="C:/Users/ADMIN/OneDrive/Máy tính/Master_2024/Cpap/hardware/CPAP_VSSM/Project Outputs for CPAP_VSSM/v3/Fabrication/Test Points/PCB_PART_NUMBER.ipc"
I=[];V=[]
for line in open(ipc,encoding='latin-1'):
    if line[:3] not in('317','327'): continue
    net=line[3:17].strip(); ref=line[20:26].strip(); pin=line[27:31].strip()
    m=re.search(r'X\s*([+-]?\d+)Y\s*([+-]?\d+)',line[30:]); x=int(m.group(1))/10; y=int(m.group(2))/10
    (V if ref=='VIA' else I).append((ref,pin,net,x,y))
print('IPC pads',len(I),'vias',len(V),'| dump pads',len(d['pads']),'vias',len(d['vias']))
dp={}
for p in d['pads']:
    dp.setdefault((p['Component'],p['Name']),[]).append((float(p['X']),float(p['Y']),p['Net']))
bad=0
for ref,pin,net,x,y in I:
    key=(ref if ref else '',pin)
    if ref.startswith('.') or ref=='': key=('',ref.lstrip('.') or pin)
    cands=dp.get(key,[])
    ok=any(abs(cx-x)<0.6 and abs(cy-y)<0.6 and (cn==net or (net=='N/C' and cn=='')) for cx,cy,cn in cands)
    if not ok: bad+=1; print('MISMATCH',ref,pin,net,x,y,cands[:2])
dv={(round(float(v['X'])),round(float(v['Y'])),v['Net']) for v in d['vias']}
iv={(round(x),round(y),n) for _,_,n,x,y in V}
print('pad mismatches',bad,'| vias only in IPC',sorted(iv-dv)[:5],'only in dump',sorted(dv-iv)[:5])
