# SiPAP Rev C v3 — adversarial review (one pass, read-only, 2026-11-22)

Nothing in the repository was modified, and no Altium tool was called. Scripts, outputs and page images are in this folder (see SOURCES.md).
Self-check: my Gerber parser reproduced the builder's numbers (268 pads, 135 tented vias, 175 holes, 101.6 x 76.2 mm outline).

**Verdict: FIX FIRST: M1, M2.** No CRITICAL defect found.

## MAJOR

**M1. The hardware over-pressure trip cannot be trimmed, and its datasheet spread is about 20–30 cmH2O, not "~25 cmH2O".**
- *Where:* U7.3 = PRESS_RED_ADC (R3 10k / R4 20k from U3, R18 1M from COMP_OUT); U7.2 = COMP_REF (R10 47k / R11 10k from 5V), per the IPC-356 file. Claims: BOM_RevC_v3.csv item 29 "~25 cmH2O"; FAB_NOTES step 3.
- *Evidence:* MPX5010 DS p.2 gives Voff min/typ/max = 0 / 0.2 / 0.425 V, VFSS 4.275–4.725 V and accuracy ±5.0 %VFSS. Transfer function is on p.4.
- *Calculation (checks/trip_calc.txt):*
  - Sensor Vout at trip = 1.3246 V, ratiometric.
  - Trip pressure = 30.0 cmH2O at Voff = 0 V, 25.5 at 0.2 V, 20.4 at 0.425 V. Span tolerance adds about ±1.2 cmH2O.
  - The firmware accepts zero readings of 0.10–0.35 V (config.h:28-29). That range alone maps to a trip at 27.7 down to 22.1 cmH2O.
- *Why it matters:*
  - Against SET_MAX 20 (config.h:51) and SOFT_LIMIT 22 (config.h:55), a high-offset U3 trips at or below the firmware limit, so it nuisance-trips at the top settings.
  - A low-offset U3 lets the pressure reach about 30 cmH2O.
  - The board's own self-test passes only 23.8–26.8 cmH2O (selftest.cpp:156). That window corresponds to U3 Voff of about 0.14–0.27 V, so in-spec sensors outside it fail acceptance and the board has no way to correct them.
- *Fix:* make the reference adjustable, e.g. R11 → a fixed resistor plus a trimmer, or a select-on-test pair of 0805 footprints. Calibrate the trip to 25 cmH2O at test. In the self-test, predict the trip from the measured U3 zero and reject boards outside the window.

**M2. Latched stopping alarms can be cleared without re-checking, and BLE commands need no authentication.**
- *Evidence:*
  - AL_ZERO is evaluated only at boot (sipap_v1.ino:312).
  - A short press keeps only AL_HW_TRIP and AL_FLOW (lines 253-257). BLE "CLEAR" does the same (line 137).
  - After that, startTherapy (line 104) runs using the default zero of 0.20 V (hw.cpp:5).
  - The zero window 0.10–0.35 V is narrower than the datasheet's 0–0.425 V, so a good sensor can raise AL_ZERO and teach the user to clear it.
  - With a true offset of 0.05 V, the firmware reads about 3.3 cmH2O low. The soft limit then acts at about 25 cmH2O real.
  - bleBegin (lines 86-100) sets no pairing or bonding, so anyone in range can send START, CLEAR or "PWM:255".
  - PWM mode runs open-loop at up to 100 % duty and skips AL_HIGH, which is only evaluated in the closed-loop branch (lines 179-192).
- *Fix:*
  - Keep AL_ZERO set until a successful re-zero with the blower off, or re-zero inside startTherapy.
  - Widen the window to match the datasheet.
  - Require BLE bonding or a whitelist.
  - Compile PWM/CLEAR out of release builds, or keep AL_HIGH and the soft limit active in manual mode.

## MINOR

**m3. The blower speed input has no defined state while the MCU is in reset.**
- *Where:* BLOWER_PWM = {U1.18, R28.1}; BLOWER_VSR = {R28.2, C26.1, J2.3}. There is no pull-down on either net (IPC-356).
- *Evidence:* The Wonsmart drawing p.1 defines VSR only as 0–3.3 V. It says nothing about an open input.
- *Why it matters:* During a WDT or brown-out reset, or while flashing through J3, IO10 is high-Z and blower speed is undefined by any document. Only the LM393 trip, with the M1 tolerance, limits pressure then.
- *Fix:* 100 kΩ from BLOWER_VSR to GND.

**m4. The datasheets kept as evidence do not match the parts JLC will fit.**
- LCSC C8545 is a **JSCJ** 2N7002, but the repo holds the Diodes datasheet and the sim uses the Diodes model.
- C7955 is an onsemi LM393DR2G (repo: TI datasheet).
- C1973187 is a Vishay SMBJ13CA-E3/52 (repo: Bourns datasheet).
- `MPX5010.pdf` is an NXP "Page not available" HTML page and `EC11.pdf` is a JPEG.
- FAB_NOTES and BOM cite "NXP MPX5010 DS p.5", but the only real MPX5010 file is a 2001 Motorola datasheet.
- *Fix:* download the official datasheets of the fitted MPNs and the current NXP MPX5010 datasheet, and replace the two stub files.

**m5. J1 description is wrong.**
- BOM item 11 says "DC barrel jack 2.1 mm". The PJ-102AH centre pin is Ø2.00 (DS p.2), and LCSC lists the part as "2mm x 6.5mm".
- *Fix:* state the pin diameter and the adapter plug size that must be used.

## Checked and found correct (do not re-audit)
1. **[ERRATUM 2026-11-22: this item was wrong. It counted only the VFB tolerance. With R19/R20 at 1 % and 100 ppm/C over 60 C, the 5V rail can reach 5.35 V, above the 5.25 V limit of MPX5010 and SFM3300. Fixed with 0.1 % Yageo RT parts, which give 4.93-5.23 V; see FAB_NOTES.md.]** TPS54202 dividers: 0.596·(1+100/13.3) = 5.08 V and 0.596·(1+100/22.1) = 3.29 V (VFB, p.5). L, R and C6 match Table 7-2 (p.17). The VFB tolerance keeps 5V at 4.95–5.21 V, inside MPXV (4.75–5.25 V) and SFM3300 (5 V ±5 %).
2. TPS54202 pinout matches the pads (p.3). BOOT cap is 100 nF on BOOT–SW. EN floating is allowed (p.3). The TVS 8/20 clamp of 28.0 V (SMBJ p.2) is below the VIN abs max of 30 V (p.4).
3. SRN6045TA ratings in the BOM and the land pattern (5.1 x 1.8 mm, 6.5 mm span) match DS p.1, checked on the page image.
4. Comparator: COMP_REF 0.877 V. Positive feedback through R18 gives 2.63 cmH2O hysteresis. LM393 VO max 36 V (p.4) allows the 12 V pull-up. VICR is fine (p.9). Unit B inputs are grounded.
5. Q1/Q4: VGS is −6 V at 12 V and −10.75 V at the TVS clamp, both within ±25 V (AO4407A p.1). RDS(on) is under 17 mΩ. Pinout S1-3, G4, D5-8 is correct. Q4 orientation suits reverse-polarity protection. An unpowered LM393 leaves Q1 off.
6. Q2/Q3: SOT-23 G1/S2/D3 (AOS p.1, Diodes p.1) matches the pad nets and geometry.
7. U4 order matches the SFM3300 pad layout (p.4). The 3V3 I2C pull-ups meet the 2.5 V VIH (p.3).
8. MPXV5010: pins 2/3/4 = Vs/GND/Vout, and pins 1 and 5–8 are left open (p.1). Decoupling is 1 µF + 10 nF + 470 pF.
9. Every GPIO in config.h matches the U1 pad nets and Table 3-1 (p.11–12). No strapping pin or IO35–37 is used. EN RC is 10k/1 µF (p.41). SAFETY_TRIP is about 2.9 V, above VIH = 0.75·VDD (p.27).
10. Netlist: all IC power and GND pins are connected. Single-pin nets are only NC pins and EN. D2 pad 1 (cathode) is on VIN12V and C25 "+" is on BLOWER_V12. J1 pads and slots match the PJ-102AH layout (p.2).
11. Gerbers: outline, layer set, mm units, drill = DRR (175 holes). The 0.25 mm/0.50 mm vias carry no JLC surcharge (jlcpcb.com: surcharge only below a 0.45 mm via diameter).
12. JLC BOM and CPL: 67/67. CPL position and rotation match the board dump. Basic/extended counts are 15/50 and 11/15. All 30 LCSC codes match the stated parts on lcsc.com; C277525 is rated 250 V where the BOM says 50 V, which is harmless.
13. MF-R400 dimensions (A 14.4, B 24.8, lead 0.81 mm) match the height map and the 1.1 mm hole. PS1240 pitch is 5 mm. C25 is D8 with 3.5 mm pitch.

## Suspicions — unverified, not findings
- DC-bias derating of the 0805 22 µF / 10 µF MLCCs versus the 44 µF in TI Table 7-2. No derating curve is on file.
- Blower FG output type (open-collector or not) is not specified by Wonsmart.
- L1 Isat 3.8 A is just below the TPS54202 I_LIM_HS max of 3.9 A. This matters only into a short.

## Tooling note
`pdftotext -layout` shifted table rows in SRN6045TA and SMBJ. A draft inductor finding built on that text was withdrawn after checking the page image.
