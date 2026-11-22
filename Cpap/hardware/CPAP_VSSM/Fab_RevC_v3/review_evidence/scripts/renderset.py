import sys,fitz,os
D=sys.argv[1]; out=sys.argv[2]
jobs={
 'TPS54202.pdf':[3,4,5,17],'SRN6045TA.pdf':[1],'MPX5010_b.pdf':[1,2,4,5],'AO4407A.pdf':[1,2],
 'AO3401A.pdf':[1],'2N7002_Diodes.pdf':[1],'LM393_TI.pdf':[4,9],'SFM3300.pdf':[3,4,6],
 'Bourns_SMBJ.pdf':[2],'Bourns_MF-R.pdf':[1],'PJ-102AH.pdf':[1,2],'TDK_PS_buzzer.pdf':[2],
 'esp32-s3-wroom-1_datasheet_v1.8.pdf':[11,12,41],'Wonsmart_WS4540-12-NZ03.pdf':[1],
}
for f,pages in jobs.items():
    doc=fitz.open(os.path.join(D,f))
    for p in pages:
        pix=doc[p-1].get_pixmap(dpi=110); fn=f"{out}/{os.path.splitext(f)[0]}_p{p}.png"; pix.save(fn)
print('ok')
