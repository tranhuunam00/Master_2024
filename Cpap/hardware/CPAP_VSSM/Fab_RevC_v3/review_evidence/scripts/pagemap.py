import sys,fitz,re
d=fitz.open(sys.argv[1]); pats=sys.argv[2:]
for i,p in enumerate(d):
    t=p.get_text()
    hits=[q for q in pats if re.search(q,t)]
    if hits: print(i+1,hits)
