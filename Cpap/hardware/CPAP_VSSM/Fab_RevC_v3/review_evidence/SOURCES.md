# Evidence index — SiPAP Rev C v3 adversarial review (2026-11-22)

Read-only review. Nothing in the repository was edited. Every device fact below comes from a vendor
datasheet already stored in `Cpap/tools/datasheets/` (SHA-256 in `datasheet_sha256.txt`), or from an
official web page fetched on 2026-11-22 (no personal data sent). Page images of every cited page
are in `pages/`; raw text extracts (not copied into the repo, re-create them with `pdftotext -layout`) were used only to find pages (WARNING: `pdftotext -layout` shifted table
columns by one row in SRN6045TA and Bourns SMBJ. Use the page images as the reference, not the text).

## Datasheets (local, vendor PDFs)
| File | Document as printed | Pages cited | What is taken from it |
|---|---|---|---|
| TPS54202.pdf | TI SLVSD26C, Apr 2016, rev. Feb 2026 | p.3 Table 4-1; p.4 abs max; p.5 VFB, UVLO, I(LIM_HS); p.17 Table 7-2 | pinout, VIN abs max 30 V, VFB 0.581/0.596/0.611 V, UVLO 4.2/3.7 V, I_LIM_HS 2.5/3.2/3.9 A, L/COUT/R2/R3/C6 table |
| SRN6045TA.pdf | Bourns SRN6045TA series | p.1 table + recommended layout | 100M: 52 mΩ, Irms 3.20 A, Isat 4.60 A; 150M: 71 mΩ, 2.80 A, 3.80 A; land 5.1 x 1.8 mm, 6.5 mm span; height 4.25 ±0.3 |
| MPX5010_b.pdf | Motorola Sensor Device Data, MPX5010/MPXV5010G series (2001) | p.1 pinout case 482A; p.2 operating characteristics; p.4 transfer function; p.5 ordering | Vs 4.75–5.25 V; Voff 0 / 0.2 / 0.425 V (0–85 °C); VFSS 4.275–4.725 V; accuracy ±5.0 %VFSS; Vout = Vs(0.09P+0.04); GC6U/GC6T1 = case 482A |
| AO4407A.pdf | AOS AO4407A | p.1, p.2 | VGS ±25 V; RDS(on) <17 mΩ @ −6 V; pinout S1-3 G4 D5-8; VGS(th) −1.7…−3 V |
| AO3401A.pdf | AOS AO3401A | p.1 | SOT-23 G/S/D (dot = G = pin 1), VGS ±12 V |
| 2N7002_Diodes.pdf | Diodes Inc. 2N7002 | p.1 | SOT-23 top view G (1) / S (2) / D (3) |
| LM393_TI.pdf | TI SLCS005AH, rev. Apr 2025 | p.4 abs max; p.9 VICR | VO max 36 V (non-B); VICR 0…VCC−1.5 V |
| SFM3300.pdf | Sensirion SFM3300 v1.4, Mar 2022 | p.3 electrical; p.4 pad layout; p.6 §4.8 | 5 V ±5 %; I/O High ≥2.5 V, Low ≤1.1 V; pads HEAT_GND·VDD·SCK·GND·DATA·HEAT; cable ≤30 cm, shield >10 cm |
| Bourns_SMBJ.pdf | Bourns SMBJ series | p.2 table | SMBJ13A/CA: VRWM 13 V, VBR 14.4–15.9 V, VC 21.5 V @ 28.0 A (10/1000), 28.0 V @ 140 A (8/20) |
| Bourns_MF-R.pdf | Bourns MF-R series | p.1 ratings; p.3 dimensions | MF-R400 30 V, Ihold 4 A; A 14.4, B 24.8, C 5.1, E 3.0, lead 0.81 mm |
| PJ-102AH.pdf | CUI/Same Sky PJ-102AH (09/12/2024) | p.1 ratings; p.2 drawing + layout | 24 V, 5.0 A; centre pin Ø2.00; slots 1.00 x 1.60 at 7.7 / 13.7 / (10.7, −4.7) mm |
| TDK_PS_buzzer.pdf | TDK PS series | p.2 | PS1240P02BT pitch 5 mm, 4 kHz, 3 V0-P |
| esp32-s3-wroom-1_datasheet_v1.8.pdf | Espressif v1.8 | p.11–12 Table 3-1; p.27 VIH; p.41 EN RC | pin → GPIO map; IO35–37 used by octal PSRAM; IO47/48 1.8 V only on R16V; VIH 0.75·VDD; EN 10 kΩ / 1 µF |
| Wonsmart_WS4540-12-NZ03.pdf | Wonsmart drawing 2023-07-12 | p.1 | 12 V, 1.6 A ±15 %; 5.0 kPa static; wires: VSR 0–3.3 V, FG, GND, +12 V |

Not usable as sources: `MPX5010.pdf` is an NXP "Page not available" HTML page (768 bytes);
`EC11.pdf` is a 100x100 JPEG. `MPX5010_b.pdf` (Motorola) is the MPX5010 source actually present.

## Official web pages fetched 2026-11-22
- JLCPCB capabilities, https://jlcpcb.com/capabilities/pcb-capabilities: 2-layer min via "0.15mm hole size / 0.25mm via diameter"; "0.2mm or 0.25mm hole size with via diameter less than 0.45mm, will cost more"; "PTH annular ring ≧0.20mm".
- LCSC product pages https://www.lcsc.com/product-detail/<code>.html, all 30 codes in the BOMs. Each one matched the fitted part named in the BOM notes:
  C1973187 SMBJ13CA-E3/52 Vishay, bidirectional; C3238865 MPXV5010GC6T1 NXP; C440198 GRM21BR61H106KE43L 10 µF 50 V X5R;
  C45783 CL21A226MAQNNNE 22 µF 25 V X5R; C49678 CC0805KRX7R9BB104 100 nF 50 V X7R; C28323 CL21B105KBFNNNE 1 µF 50 V X7R;
  C1710 CL21B103KBANNNC 10 nF 50 V X7R; C277525 CC0805JRNPOYBN471 470 pF **250 V** NP0; C1838 0805CG750J500NT 75 pF C0G;
  C1828 0805CG560J500NT 56 pF C0G; C17414 / C4328 / C17673 / C17713 / C17514 / C149504 / C17513 / C17446 / C17552 UNI-ROYAL
  10k / 20k / 4.7k / 47k / 1M / 100k / 1k / 13.3k / 22.1k, all 1 %; C441993 ERJP06F3303V 330k 1 %; C191884 TPS54202DDCR;
  C2913201 ESP32-S3-WROOM-1-N8R8; C16072 AO4407A (AOS); C7955 LM393DR2G (onsemi); C15127 AO3401A (AOS);
  C8545 2N7002 (**JSCJ**); C208488 MF-R400 (Bourns, P = 5.08 mm, 4 A, 30 V); C3096093 PJ-102AH ("2mm x 6.5mm");
  C76871 PS1240P02BT; C178666 EEUFC1E221 D8 x 11.5.

## Design files read (repository, read-only)
`Fab_RevC_v3/` (FAB_NOTES.md, REVIEW_v3.md, both BOMs, both CPLs, pin1_check, IPC-356, Gerbers, drill, DRR, schematic PDF text layer),
`tools/layout_v3/board_dump_v3_final.json`, `sim/README.md`, `sim/sim_normal_operation.cir`, `sim/models/README.md`,
`3D/height_map.md`, `Cpap/hardware/wiring_harness.md`, `firmware/sipap_v1/*`.

## Own checks (scripts in `scripts/`, outputs in `checks/`)
- `ipc356_nets.txt`: every net with its pads, from the IPC-356 file.
- `gerber_check.txt`: 268/268 pads on the right copper/mask layers, paste missing only on TP1–4/FID1–3, 135/135 vias with copper on both sides and tented. This reproduces the builder's numbers.
- `bom_cpl_check.txt`: JLC BOM 67 = CPL 67, quantities correct, CPL position and rotation = board dump for 67/67.
- `trip_calc.txt`: hardware trip pressure vs U3 offset.

## Added during the fixes (2026-11-22, after the review)
Downloaded with the user's approval from the manufacturer or distributor site and stored in `Cpap/tools/datasheets/` (SHA-256 appended to `datasheet_sha256.txt`):

| File | Source URL | Pages cited | What is taken from it |
|---|---|---|---|
| Vishay_SMBJ.pdf | https://www.vishay.com/docs/88392/smbj.pdf | p.1 (CA suffix = bidirectional), p.2 table | fitted D2 (C1973187): SMBJ13A/CA VWM 13 V, VBR 14.4–15.9 V, IPPM 27.9 A, VC 21.5 V |
| onsemi_LM393.pdf | https://www.onsemi.com/pdf/datasheet/lm393-d.pdf | p.1 pin connections, abs max (VO −0.3 to +36 V) | fitted U7 (C7955): 1 output A, 2/3 inputs A, 4 GND, 8 VCC |
| JSCJ_2N7002.pdf | LCSC copy for C8545: https://datasheet.lcsc.com/datasheet/pdf/a141b8bd86b14475955ac8c4d3eea0a8.pdf | p.1 | fitted Q3: 1 gate, 2 source, 3 drain; VGS ±20 V |
| Bourns_3224.pdf | https://www.bourns.com/docs/Product-Datasheets/3224.pdf | p.1 (3224W views, land pattern, 1 CCW / 2 wiper / 3 CW, 12 turns, ±10 %, 0.25 W), p.3 (life-critical medical disclaimer) | RV1 footprint, 3D body 3.5 × 4.8 × 5.3 mm, adjust direction |

Official web pages: https://jlcpcb.com/partdetail/C4360 (0805W8F3602T5E, Extended) and https://jlcpcb.com/partdetail/C48484 (Bourns 3224W-1-503E, Extended); https://www.lcsc.com/product-detail/C4360.html and C48484.html (MPN, maker, package). https://www.nxp.com/docs/en/data-sheet/MPX5010.pdf returned HTTP 404 on 2026-11-22, so the Motorola MPX5010_b.pdf remains the MPX5010 source.

Post-fix checks in `checks/`: `gerber_check_after_fix.txt` (273 pads, 139 tented vias), `ipc356_nets_after_fix.txt`, `gerber_xor_vs_HEAD.txt` (XOR against commit 6f5d0b1; rasteriser `scripts/rast2.py` handles %LPC% clear polarity, which the first one, `scripts/rast.py`, ignored and so reported copper layers as identical: kept to show the tool error).
