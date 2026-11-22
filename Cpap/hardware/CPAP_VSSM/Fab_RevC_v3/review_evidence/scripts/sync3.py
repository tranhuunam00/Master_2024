import csv,re,collections,sys
F="C:/Users/ADMIN/OneDrive/Máy tính/Master_2024/Cpap/hardware/CPAP_VSSM/Fab_RevC_v3/"
sch=collections.defaultdict(set); pin_net={}
rows=list(csv.reader(open("C:/Users/ADMIN/EDA Agent/workspace/sipap_sch_netlist.csv",encoding='utf-8-sig')))
hdr=[h.lower() for h in rows[0]]; ci,pi,ni=hdr.index('component'),hdr.index('pin'),hdr.index('net')
for r in rows[1:]:
    sch[r[ni]].add((r[ci],r[pi])); pin_net[(r[ci],r[pi])]=r[ni]
pcb=collections.defaultdict(set); pcb_pin={}
for line in open(F+'SiPAP_RevC_v3.ipc',encoding='latin-1'):
    if line[:3] not in('317','327'): continue
    net=line[3:17].strip(); ref=line[20:26].strip(); pin=line[27:31].strip()
    if ref=='VIA' or ref.startswith('.') or ref=='': continue
    if ref=='U6' and pin=='MP': continue
    pcb[net].add((ref,pin)); pcb_pin[(ref,pin)]=net
# per-pin comparison
mism=[];nc_ok=0
for k,n in pin_net.items():
    p=pcb_pin.get(k)
    sch_single=len(sch[n])==1
    if p is None: mism.append(('pin missing on PCB',k,n)); continue
    if sch_single and p=='N/C': nc_ok+=1; continue
    if p!=n: mism.append(('net differs',k,'sch',n,'pcb',p))
extra=[k for k in pcb_pin if k not in pin_net]
multi=[n for n in sch if len(sch[n])>1]
print('schematic pins',len(pin_net),'PCB pins',len(pcb_pin),'| multi-pin nets',len(multi),'| single-pin nets = N/C on PCB:',nc_ok)
print('mismatches',mism[:20]); print('PCB pins not in schematic',extra[:20])
# net-level equality for multi-pin nets
bad=[n for n in multi if sch[n]!=pcb.get(n,set())]
print('multi-pin nets with different pin sets',bad)
# BOM vs schematic designators and values
def expand(s):
    out=[]
    for t in s.split():
        if '-' in t:
            a,b=t.split('-'); p=''.join(c for c in a if c.isalpha()); out+=[f'{p}{i}' for i in range(int(a[len(p):]),int(b[len(p):])+1)]
        else: out.append(t)
    return out
bom=list(csv.DictReader(open(F+'BOM_RevC_v3.csv',encoding='utf-8')))
bom_des={}
for r in bom:
    for d in expand(r['Designators']): bom_des.setdefault(d,[]).append(r)
schcomps={c for c,_ in pin_net}
print('in schematic, not in BOM',sorted(schcomps-set(bom_des)))
print('in BOM (parts), not in schematic',sorted(d for d in bom_des if d not in schcomps and not d.startswith(('MH','FID','TP'))))
print('on several BOM lines',[d for d,v in bom_des.items() if len(v)>1])
jlc=list(csv.DictReader(open(F+'SiPAP_RevC_v3_BOM_JLC.csv',encoding='utf-8')))
cpl={r['Designator'] for r in csv.DictReader(open(F+'SiPAP_RevC_v3_CPL_JLC.csv',encoding='utf-8'))}
jd=set(d for r in jlc for d in r['Designator'].split(','))
smd={d for d,v in bom_des.items() if v[0]['Mount']=='SMD'}
print('JLC BOM == CPL:',jd==cpl,'| JLC BOM == all SMD in main BOM:',jd==smd, sorted(smd^jd))
