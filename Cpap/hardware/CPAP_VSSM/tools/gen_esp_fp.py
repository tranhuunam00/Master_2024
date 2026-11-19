import json
pads=[]
for i in range(14):           # left column pins 1..14, top to bottom
    pads.append({"number":str(1+i),"x":-8.75,"y":round(5.26-1.27*i,3),"width":1.5,"height":0.9})
for i in range(12):           # bottom row pins 15..26, left to right
    pads.append({"number":str(15+i),"x":round(-6.985+1.27*i,3),"y":-12.5,"width":0.9,"height":1.5})
for i in range(14):           # right column pins 27..40, bottom to top
    pads.append({"number":str(27+i),"x":8.75,"y":round(-11.25+1.27*i,3),"width":1.5,"height":0.9})
for dy in (1.4,0,-1.4):       # EPAD 3x3 sub-pads 0.9, 0.5 gap -> 3.7 overall
    for dx in (-1.4,0,1.4):
        pads.append({"number":"41","x":round(-1.5+dx,3),"y":round(-2.46+dy,3),"width":0.9,"height":0.9})
pkg={"family":"custom","name":"ESP32-S3-WROOM-1_DS",
 "body_x":18.0,"body_y":25.5,"height":3.1,
 "source":{"document":"ESP32-S3-WROOM-1 & WROOM-1U Datasheet","revision":"v1.8","page":45,"ref":"Figure 11-1 Recommended PCB Land Pattern",
           "path":r"C:\Users\ADMIN\OneDrive\Máy tính\Master_2024\Cpap\tools\datasheets\esp32-s3-wroom-1_datasheet_v1.8.pdf"},
 "custom_pads":pads,
 "drawing_numbers":[18,6,7.49,25.5,1.5,0.9,16.51,1.5,0.5,0.9,0.5,3.7,3.7,0.9,7.5,10.29,1.27,0.5,1.27,2.015,2.015,17.5],
 "drawing_dims":[
  {"pad":"1","mm":1.5,"axis":"x"},{"pad":"1","mm":0.9,"axis":"y"},
  {"pad":"15","mm":0.9,"axis":"x"},{"pad":"15","mm":1.5,"axis":"y"},
  {"from":"1","to":"14","mm":16.51,"axis":"y","between":"centres"},
  {"from":"1","to":"2","mm":1.27,"axis":"y","between":"centres"},
  {"from":"15","to":"16","mm":1.27,"axis":"x","between":"centres"},
  {"from":"1","to":"40","mm":17.5,"axis":"x","between":"centres"},
  {"pad":"41#1","mm":0.9,"axis":"x"},{"pad":"41#1","mm":0.9,"axis":"y"},
  {"from":"41#1","to":"41#2","mm":0.5,"axis":"x","between":"gap"},
  {"from":"41#1","to":"41#3","mm":3.7,"axis":"x","between":"outer"},
  {"from":"41#1","to":"41#7","mm":3.7,"axis":"y","between":"outer"}
 ]}
json.dump(pkg,open("esp_pkg.json","w"),indent=1)
print(len(pads))
