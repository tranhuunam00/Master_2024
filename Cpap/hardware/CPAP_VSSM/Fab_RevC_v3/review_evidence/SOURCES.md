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

## R19/R20 change (2026-11-22, later)
| File | Source URL | Pages cited | What is taken from it |
|---|---|---|---|
| Yageo_RT.pdf | LCSC copy of the Yageo RT series datasheet V.16 (May 06, 2025), linked from both C122537 and C865182: https://datasheet.lcsc.com/datasheet/pdf/46d7760c582640714d605308a25a220a.pdf | p.2 ordering code (B = 0.1 %, R = paper taping, D = 25 ppm/C); p.5 Table 2 (RT0805 1/8 W, 150 V, 0.1 %/25 ppm range 1 R-1.5 M) | R19 RT0805BRD07100KL, R20 RT0805BRD0713K3L |

JLC pages (read by a web-fetch agent on 2026-11-22): https://jlcpcb.com/partdetail/C122537 and https://jlcpcb.com/partdetail/C865182, both Yageo, 0805, 0.1 %, 25 ppm, 125 mW, 150 V, "Extended". Stock was not shown. The yageo.com URLs tried for this datasheet returned 404.
Checks: `checks/rail5v_worstcase.txt`, `checks/gerber_xor_after_R19R20.txt` (7/7 layers identical), `checks/sch_pcb_bom_sync.txt`.

## Off-board parts (2026-11-22, later): E3–E11 in `BOM_RevC_v3_offboard.csv`
Downloaded on 2026-11-22 and stored in `Cpap/tools/datasheets/` (SHA-256 appended to `datasheet_sha256.txt`). Page images of every cited page are in `pages/` (file name = datasheet name + page).

| File | Source URL | Pages cited | What is taken from it |
|---|---|---|---|
| MeanWell_GSM60B.pdf | https://www.meanwell.com/Upload/PDF/GSM60B/GSM60B-SPEC.PDF (file dated 2026-05-09) | p.1 features, model encoding; p.2 specification | E4 GSM60B12-P1J: 12 V ±3.0 %, 5 A, ripple 100 mVp-p, OVP 12.6–16.2 V, touch current < 50 µA, 2×MOPP, IEC 60601-1 / 60601-1-11, Class II, C8 inlet; P1J = 2.1 × 5.5 × 11 mm, C+ |
| MeanWell_DC_plug.pdf | https://www.meanwell.com/upload/pdf/DC_plug.pdf | p.1 | E4b DC PLUG-P1J-P1I = 2.1 × 5.5 × 9.5 mm; no 2.0 mm option is offered |
| Adafruit_monochrome-oled-breakouts.pdf | https://cdn-learn.adafruit.com/downloads/pdf/monochrome-oled-breakouts.pdf | p.7 power; p.17 and p.49 STEMMA QT wiring; p.79 schematic (JP2, I2C address note) | E3: Vin 3–5 V, 3.3 V logic safe; I2C default, no RST needed; JP2 1 SDA … 7 VCC, 8 GND; 0x3D with DC/SA0 open |
| Solomon_SSD1306_Adafruit.pdf | https://cdn-shop.adafruit.com/datasheets/SSD1306.pdf (Solomon Systech Rev 1.1, Apr 2008) | p.48 DC characteristics | VDD 1.65–3.3 V, VIH 0.8 × VDD |
| Espressif_ESP-Prog_user_guide.html | https://docs.espressif.com/projects/esp-dev-kits/en/latest/other/esp-prog/user_guide.html | Program Interface table, Description of Components, Package Contents | E5: PROG 2.54 mm pin list; RX/TX always 3.3 V; only 1.27 mm flat cables in the box |
| Espressif_SCH_ESP32-PROG_V2.1.pdf, Espressif_DIM_ESP32-PROG_V2.1.pdf | https://dl.espressif.com/dl/schematics/SCH_ESP32-PROG_V2.1_20190709.pdf, …/DIM_ESP32-PROG_V2.1_20190709.pdf | SCH p.1 (crops `_uart_crop`, `_header_crop`) | FT_TXD → R20 → ESP_RXD0, FT_RXD → R21 → ESP_TXD0; header pin 1 EN, 3 TXD0, 5 RXD0 |
| Molex_SD-90119.pdf | LCSC copy for C588663: https://datasheet.lcsc.com/datasheet/pdf/c0c1a1b7bfbe9e5a950cae27bf7867eb.pdf (Molex drawing SD-90119, 2 sheets) | sheet 1 note 4; sheet 2 table | 90119-2110: plating E (0.38–0.64 µm Au), 22–24 AWG, insulation 1.02–1.47 mm; mates with 0.635 mm square pin |
| Molex_90123_catalog.pdf | Farnell copy of the Molex catalog page C-86: https://www.farnell.com/datasheets/7011.pdf | p.1 | 90123: 3.0 A, 350 V, width 2.54 mm max, stackable in 2 directions, ordering table 0102–0132 |
| Molex_90123-0102.pdf, Molex_90156-0144.pdf, Molex_90142-0006.pdf | Rapid copies of Molex part sheets: https://static.rapidonline.com/pdf/05-4394_v1.pdf, 05-4422_v1.pdf, 05-4407_v1.pdf | p.1 | circuits, rows, "Use With 90119", stackable (90123) |
| Molex_90156-0146.pdf | Farnell copy of the Molex part sheet: https://www.farnell.com/datasheets/2672904.pdf | p.1 | 6 circuits, single row, "Use With 90119" |
| Molex_90119-0111.pdf | Farnell copy of the Molex part sheet (generated 2025-03-08): https://www.farnell.com/datasheets/4507172.pdf | p.1–2 | tooling: hand crimp tool 63825-8100 for 24–22 AWG |
| Belden_9534.pdf, Belden_9535.pdf, Belden_8444.pdf | https://catalog.belden.com/techdata/EN/9534_techdata.pdf, 9535_techdata.pdf, 8444_techdata.pdf | p.1 | 9534/9535: 4/5 × 24 AWG, insulation 1.1 mm, foil + drain, 110/210 pF/m, 1.8 A; 8444: 4 × 22 AWG, insulation 1.3 mm, 2.4 A |

Not saved, and why:
- molex.com (drawings SD-90156, SD-90142, SD-90123 and product spec PS-99020-0001) timed out for curl and for the web-fetch tool. The pages load in a browser, but the site blocks sending the file anywhere, so the Molex part sheets and catalog page above (copies hosted by Farnell and Rapid) are used instead.
- Saint-Gobain Tygon ND 100-65 datasheet (medical.saint-gobain.com) answers with a browser check (HTTP 403) → E9 left open.
- Nordson Medical T20-6005: the product page (fluid-components.nordsonmedical.com) gives "Tee, 3/32 in (2.4 mm) ID tubing, polypropylene, bags of 1000"; no drawing PDF exists at the usual part-drawing address (HTTP 404) → E9 left open.
- Kilo International knobs: only distributor listings found, no maker drawing → E10 left open.

Check: `CPAP_VSSM/tools/check_harness.py` after switching to the Adafruit OLED and ESP-Prog: 23 PASS, 0 FAIL, 1 OPEN (blower FG output stage, as before).
