import json, math
def rect(x0,y0,x1,y1): return [[x0,y0],[x1,y0],[x1,y1],[x0,y1]]
def circ(cx,cy,r,n=16): return [[round(cx+r*math.cos(2*math.pi*i/n),3),round(cy+r*math.sin(2*math.pi*i/n),3)] for i in range(n)]
B={}
# ESP32-S3-WROOM-1 DS v1.8 Fig10-1: 18x25.5, PCB 0.8, shield 15.8x17.6 (1.05 from bottom), total 3.1
B["ESP32-S3-WROOM-1_DS"]=[{"points":rect(-9,-12.75,9,12.75),"overall_height":0.8,"identifier":"module_pcb"},
  {"points":rect(-7.9,-11.7,7.9,5.9),"standoff_height":0.8,"overall_height":3.1,"identifier":"shield"}]
# MPXV5010GC6U Case 482A p7: A,B 10.79 max; C 13.21 max; V 6.22-6.48 port; W 3.17 port dia; S 18.41 lead span
B["MPXV5010GC6U_482A"]=[{"points":rect(-5.395,-5.395,5.395,5.395),"overall_height":6.99,"identifier":"body"},
  {"points":circ(0,0,1.585),"standoff_height":6.99,"overall_height":13.21,"identifier":"port"},
  {"points":rect(-9.205,-4.35,9.205,4.35),"overall_height":1.0,"identifier":"leads"}]
# PEC11R-4220F-S0024 p2/p3: body 12.5x13.4, depth 6.5 to mounting surface, LB=7 bushing M7, L=20 shaft dia 6
B["PEC11R-4xxxF-S_DS"]=[{"points":rect(-6.25,-6.7,6.25,6.7),"overall_height":6.5,"identifier":"body"},
  {"points":circ(0,0,3.5),"standoff_height":6.5,"overall_height":13.5,"identifier":"bushing_M7"},
  {"points":circ(0,0,3.0),"standoff_height":13.5,"overall_height":26.5,"identifier":"shaft"}]
# JEDEC MS-012 SOIC-8: 4.9 x 3.9 body, 6.0 lead span, 1.75 max
B["SOIC-8"]=[{"points":rect(-1.95,-2.45,1.95,2.45),"overall_height":1.75,"identifier":"body"},
  {"points":rect(-3.0,-2.2,3.0,2.2),"overall_height":0.4,"identifier":"leads"}]
# JEDEC TO-236 SOT-23: 2.9 x 1.3 body, 1.12 max (pins 1,2 at -y, 3 at +y)
B["SOT-23"]=[{"points":rect(-1.45,-0.65,1.45,0.65),"overall_height":1.12,"identifier":"body"}]
# EIA 0805: 2.0 x 1.25, MLCC up to 1.25 thick (worst case used for R and C)
B["R_0805"]=[{"points":rect(-1.0,-0.625,1.0,0.625),"overall_height":1.25,"identifier":"body"}]
# DO-214AB SMC nominal 7.0 x 6.0 x 2.6 (assumed, check chosen diode datasheet)
B["DIOM_SMC_DO-214AB"]=[{"points":rect(-3.5,-3.0,3.5,3.0),"overall_height":2.62,"identifier":"body"}]
# 2.54mm male headers: plastic 2.5, pin 6.0 above -> 8.5 (generic)
B["CONN_HEADER_4"]=[{"points":rect(-1.27,-1.27,8.89,1.27),"overall_height":2.5,"identifier":"housing"},
  {"points":rect(-0.5,-0.5,8.12,0.5),"standoff_height":2.5,"overall_height":8.5,"identifier":"pins"}]
B["CONN_HEADER_6"]=[{"points":rect(-1.27,-1.27,13.97,1.27),"overall_height":2.5,"identifier":"housing"},
  {"points":rect(-0.5,-0.5,13.2,0.5),"standoff_height":2.5,"overall_height":8.5,"identifier":"pins"}]
B["CONN_HEADER_4_2X2"]=[{"points":rect(-1.27,-1.27,3.81,3.81),"overall_height":2.5,"identifier":"housing"},
  {"points":rect(-0.5,-0.5,3.04,3.04),"standoff_height":2.5,"overall_height":8.5,"identifier":"pins"}]
# Buzzer: generic 12 mm magnetic buzzer 7.6 pitch, 9.5 tall (assumed)
B["BUZZER_THT_7.6"]=[{"points":circ(3.8,0,6.0),"overall_height":9.5,"identifier":"body"}]
# DC jack: generic 5.5x2.1 PCB jack ~9 x 14 x 11 (assumed)
B["DCJACK_THT_2PIN"]=[{"points":rect(-2.0,-7.0,7.0,7.0),"overall_height":11.0,"identifier":"body"}]
json.dump(B,open('bodies.json','w'))
items=[{"footprint_name":k,"bodies":v} for k,v in B.items()]
print(json.dumps(items))
