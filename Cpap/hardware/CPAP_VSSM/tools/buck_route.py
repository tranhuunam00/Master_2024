"""Hand-routed critical loops of the two TPS54202 cells (TI SLVSD26C layout guidance)."""
import json
T, V = [], []
def cell(ux, uy, sw, bst, fb, vout):
    t = lambda x1,y1,x2,y2,w,n: T.append(dict(x1=ux+x1,y1=uy+y1,x2=ux+x2,y2=uy+y2,width=w,net_name=n,layer="TopLayer"))
    # SW: pin2 -> left -> up into L pad1
    t(-53.15,0,-110,0,20,sw); t(-110,0,-110,167.5,20,sw)
    # boot cap SW side -> L pad1
    t(130,147.4,-20,147.4,20,sw)
    # BOOT pin6 -> C boot pad1
    t(53.15,37.4,130,72.6,15,bst)
    # VIN pin3 -> Cbyp -> Cin
    t(-53.15,-37.4,-300,-37.4,24,"VIN12V")
    # FB pin4 -> Rtop/Cff/Rbot node
    t(53.15,-37.4,100,-82.6,10,fb); t(100,-82.6,350,-82.6,10,fb)
    # Vout top of Rtop/Cff
    t(170,-157.4,260,-157.4,15,vout)
    # GND pin1 -> via
    t(-53.15,37.4,-10,80,15,"GND")
    for x,y in [(-10,80),(-190,90),(-300,90)]:
        V.append(dict(x=ux+x,y=uy+y,size=24,hole_size=12,net="GND"))
    t(-190,37.4,-190,90,20,"GND"); t(-300,37.4,-300,90,20,"GND")
cell(400,1700,"SW5","BST5","FB5","5V")
cell(1150,1700,"SW3","BST3","FB3","3V3")
json.dump({"tracks":T,"vias":V},open("buck_route.json","w"),indent=0)
print(json.dumps(T,separators=(',',':')));print(json.dumps(V,separators=(',',':')))
