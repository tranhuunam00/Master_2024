# Hardware over-pressure trip vs MPXV5010 offset (U3). Values from IPC-356 netlist + BOM:
# PRESS_RED_ADC: R3 10k (from U3 Vout), R4 20k (to GND), R18 1M (to COMP_OUT); COMP_REF: R10 47k (5V) / R11 10k (GND)
# MPX5010 (Motorola/NXP DS p.2): Vout = Vs*(0.09*P + 0.04) nominal; Voff 0 / 0.2 / 0.425 V @Vs=5 V; VFSS 4.275-4.725 V
R3,R4,R18,R10,R11=10e3,20e3,1e6,47e3,10e3
Vs=5.0
vref=Vs*R11/(R10+R11)
par=R4*R18/(R4+R18); g=par/(R3+par)           # COMP_OUT low (~0 V): R18 to ground
v_trip=vref/g                                 # sensor Vout at trip
def P_kpa(vout,voff,span=4.5): return (vout-voff)/(span/10.0)   # span V over 10 kPa
cm=10.1972
print(f"COMP_REF {vref:.4f} V, divider gain {g:.5f}, sensor Vout at trip {v_trip:.4f} V")
for voff in (0.0,0.10,0.142,0.20,0.274,0.35,0.425):
    print(f"Voff {voff:.3f} V -> trip {P_kpa(v_trip,voff)*cm:5.1f} cmH2O (span 4.5 V); "
          f"span 4.275..4.725 V -> {P_kpa(v_trip,voff,4.725)*cm:5.1f}..{P_kpa(v_trip,voff,4.275)*cm:5.1f}")
# hysteresis: COMP_OUT high ~11.6 V through R18
vco=11.6
v_rel=(vref*(1/R3+1/R4+1/R18)-vco/R18)*R3
print(f"release Vout {v_rel:.4f} V, hysteresis {(v_trip-v_rel)/0.45*cm:.2f} cmH2O")
