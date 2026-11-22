import sys,re,collections
p=sys.argv[1]
nets=collections.defaultdict(list)
for line in open(p,encoding='latin-1'):
    if line[:3] in ('317','327'):
        net=line[3:17].strip(); ref=line[20:26].strip(); pin=line[27:31].strip()
        if ref=='VIA': continue
        nets[net].append(f"{ref}.{pin}")
for n in sorted(nets):
    print(n, len(nets[n]), ' '.join(sorted(nets[n])))
