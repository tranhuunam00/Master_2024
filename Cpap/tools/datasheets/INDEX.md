# Datasheets — SiPAP Rev C v3

Each file and the parts it covers, from the `Datasheet` column of `hardware/CPAP_VSSM/Fab_RevC_v3/BOM_RevC_v3.csv` and `BOM_RevC_v3_offboard.csv`.
`tools/layout_v3/fab_v3.py` stops if a BOM line names a file that is not here.
Source: the official link for each file, machine-readable in `SOURCES.csv` (File, URL, Check). "SHA-256 identical" means the link was downloaded on 2026-11-22 and is byte-for-byte the file here. Files marked NOT VERIFIED or NO PUBLIC LINK have no checked link; the reason is given.

| File | Pages | SHA-256 (first 16) | Used for | Source |
|---|---|---|---|---|
| 2N7002_Diodes.pdf | 5 | `3984debf5b293762` | Q3 (2N7002) | https://www.diodes.com/assets/Datasheets/ds11303.pdf (SHA-256 identical to the repo file (2026-11-22)) |
| Adafruit_monochrome-oled-breakouts.pdf | 85 | `fd3bf6312ddd70b0` | off-board E3 Adafruit 326 | https://cdn-learn.adafruit.com/downloads/pdf/monochrome-oled-breakouts.pdf (downloaded from this link on 2026-11-22) |
| AO3401A.pdf | 5 | `0d8e3261ae280e00` | Q2 (AO3401A) | https://www.aosmd.com/res/datasheets/AO3401A.pdf (SHA-256 identical to the repo file (2026-11-22)) |
| AO4407A.pdf | 5 | `ef4b3686383ca726` | Q1 Q4 (AO4407A) | https://www.aosmd.com/res/datasheets/AO4407A.pdf (SHA-256 identical to the repo file (2026-11-22)) |
| Belden_8444.pdf | 2 | `6069b0345236c091` | off-board E6 cable | https://catalog.belden.com/techdata/EN/8444_techdata.pdf (downloaded from this link on 2026-11-22) |
| Belden_9534.pdf | 2 | `3fa391c6cae9fe63` | off-board E7, E8 cable | https://catalog.belden.com/techdata/EN/9534_techdata.pdf (downloaded from this link on 2026-11-22) |
| Belden_9535.pdf | 2 | `2e05e120d625c5ec` | off-board E5b cable | https://catalog.belden.com/techdata/EN/9535_techdata.pdf (downloaded from this link on 2026-11-22) |
| Bourns_3224.pdf | 3 | `db9857638bb6b034` | RV1 (3224W-1-503E) | https://www.bourns.com/docs/Product-Datasheets/3224.pdf (downloaded from this link on 2026-11-22) |
| Bourns_MF-R.pdf | 10 | `d22f0f06c8909784` | F1 (MF-R400) | https://www.bourns.com/docs/Product-Datasheets/mfr.pdf (SHA-256 identical to the repo file (2026-11-22)) |
| Bourns_MF-SM.pdf | 7 | `c55c1edf8cba275a` | **not used by the current BOM** | https://www.bourns.com/docs/Product-Datasheets/mfsm.pdf (SHA-256 identical to the repo file (2026-11-22)) |
| Bourns_SMBJ.pdf | 6 | `3303bff29fd46a97` | D2 (SMBJ13CA-E3/52) | https://www.bourns.com/docs/Product-Datasheets/SMBJ.pdf (SHA-256 identical to the repo file (2026-11-22)) |
| esp32-s3-wroom-1_datasheet_v1.8.pdf | 53 | `27d71971da07c280` | U1 (ESP32-S3-WROOM-1-N8R8) | https://www.espressif.com/sites/default/files/documentation/esp32-s3-wroom-1_wroom-1u_datasheet_en.pdf (SHA-256 identical to the repo file (2026-11-22); the link always serves the latest version, so it will differ after Espressif publishes v1.9) |
| Espressif_DIM_ESP32-PROG_V2.1.pdf | 1 | `6d67a938b88e8b89` | off-board E5 ESP-Prog | https://dl.espressif.com/dl/schematics/DIM_ESP32-PROG_V2.1_20190709.pdf (downloaded from this link on 2026-11-22) |
| Espressif_ESP-Prog_user_guide.html | HTML | `75e40db0b2f8601c` | off-board E5 ESP-Prog | https://docs.espressif.com/projects/esp-dev-kits/en/latest/other/esp-prog/user_guide.html (downloaded from this link on 2026-11-22 (page saved as HTML)) |
| Espressif_SCH_ESP32-PROG_V2.1.pdf | 1 | `ab1f074c7b6c75e7` | off-board E5 ESP-Prog | https://dl.espressif.com/dl/schematics/SCH_ESP32-PROG_V2.1_20190709.pdf (downloaded from this link on 2026-11-22) |
| FH_0805CG.pdf | 30 | `705023d3be045414` | C19 (0805CG750J500NT); C22 (0805CG560J500NT) | https://datasheet.lcsc.com/datasheet/pdf/09fe2daf22580a99fa7fea93ccb20fb7.pdf (LCSC copy; SHA-256 identical to the repo file (2026-11-22) (C1838, C1828)) |
| JSCJ_2N7002.pdf | 5 | `7941fb423af7c6c6` | Q3 (2N7002) | https://datasheet.lcsc.com/datasheet/pdf/a141b8bd86b14475955ac8c4d3eea0a8.pdf (LCSC copy; SHA-256 identical to the repo file (2026-11-22) (C8545)) |
| LM393_TI.pdf | 56 | `a1f63934820e27c1` | U7 (LM393DR2G) | https://www.ti.com/lit/ds/symlink/lm393.pdf (SHA-256 identical to the repo file (2026-11-22); the link always serves the latest revision) |
| MeanWell_DC_plug.pdf | 1 | `6184980f2c2a71a9` | off-board E4b DC PLUG-P1J-P1I | https://www.meanwell.com/upload/pdf/DC_plug.pdf (downloaded from this link on 2026-11-22) |
| MeanWell_GSM60B.pdf | 5 | `3b971459c945ec49` | off-board E4 GSM60B12-P1J | https://www.meanwell.com/Upload/PDF/GSM60B/GSM60B-SPEC.PDF (downloaded from this link on 2026-11-22) |
| Molex_90119-0111.pdf | 3 | `8fc089740fd993ef` | off-board E11 crimp tool 63825-8100 (tooling list) | https://www.farnell.com/datasheets/4507172.pdf (Farnell copy of the Molex part sheet; downloaded from this link on 2026-11-22) |
| Molex_90123-0102.pdf | 1 | `6df4a138a27048ea` | off-board E6 housing | https://static.rapidonline.com/pdf/05-4394_v1.pdf (Rapid copy of the Molex part sheet; downloaded from this link on 2026-11-22) |
| Molex_90123_catalog.pdf | 1 | `6c23cd319aa7b31d` | off-board E6, E8 housings (90123 series page) | https://www.farnell.com/datasheets/7011.pdf (Farnell copy of the Molex catalog page; downloaded from this link on 2026-11-22) |
| Molex_90142-0006.pdf | 1 | `d6119c4e7f087beb` | off-board E5b housing | https://static.rapidonline.com/pdf/05-4407_v1.pdf (Rapid copy of the Molex part sheet; downloaded from this link on 2026-11-22) |
| Molex_90156-0144.pdf | 1 | `e36780fa8b7c98bc` | off-board E8 housing | https://static.rapidonline.com/pdf/05-4422_v1.pdf (Rapid copy of the Molex part sheet; downloaded from this link on 2026-11-22) |
| Molex_90156-0146.pdf | 1 | `2affe2eb4a7a2d80` | off-board E5b, E7 housing | https://www.farnell.com/datasheets/2672904.pdf (Farnell copy of the Molex part sheet; downloaded from this link on 2026-11-22) |
| Molex_SD-90119.pdf | 2 | `cc5884db48038677` | off-board E5b-E8 crimp terminal 90119-2110 | https://datasheet.lcsc.com/datasheet/pdf/c0c1a1b7bfbe9e5a950cae27bf7867eb.pdf (LCSC copy of Molex drawing SD-90119 (C588663); downloaded from this link on 2026-11-22) |
| MPX5010_b.pdf | 12 | `796fbf22bebe0df7` | U2 U3 (MPXV5010GC6T1) | NOT VERIFIED: Motorola MPX5010/D (2001); download source was not recorded. https://www.nxp.com/docs/en/data-sheet/MPX5010.pdf returns 404 and nxp.com returns 403 from here (2026-11-22) |
| Murata_GRM21BR61H106KE43.pdf | 31 | `ff8de8901544f5f4` | C10 C13 C14 (GRM21BR61H106KE43L) | https://datasheet.lcsc.com/datasheet/pdf/8cca754305bc7d881114d4ba10b82bf2.pdf (LCSC copy; SHA-256 identical to the repo file (2026-11-22) (C440198)) |
| onsemi_LM393.pdf | 13 | `2fd8e714267f0539` | U7 (LM393DR2G) | https://www.onsemi.com/pdf/datasheet/lm393-d.pdf (downloaded from this link on 2026-11-22) |
| Panasonic_ERJP06.pdf | 9 | `441511405b982033` | R16 (ERJP06F3303V) | https://datasheet.lcsc.com/datasheet/pdf/3694598bef8f46f8e50ce38954aee850.pdf (LCSC copy; SHA-256 identical to the repo file (2026-11-22) (C441993)) |
| Panasonic_FC.pdf | 17 | `42d91ae29ab6122d` | C25 (EEU-FC1E221) | NOT VERIFIED: Panasonic "FC series catalog" (2018-10-17); download source was not recorded. industrial.panasonic.com returns 403 from here, and none of the 5 LCSC copies for C178666 matches (2026-11-22) |
| PEC11R.pdf | 4 | `ac88f657e611bc82` | U6 (PEC11R-4225F-S0024) | https://www.bourns.com/docs/Product-Datasheets/PEC11R.pdf (SHA-256 identical to the repo file (2026-11-22)) |
| PJ-102AH.pdf | 3 | `269ff7d1d8b7130a` | J1 (PJ-102AH) | https://www.sameskydevices.com/product/resource/pj-102ah.pdf (same text and date (09/12/2024) as the repo file; Same Sky stamps a new PDF on every download, so SHA-256 differs) |
| Samsung_CL21.pdf | 84 | `96baebe4bd0ce363` | C23 (CL21A226MAQNNNE); C1 C2 C18 C21 C27 (CL21A226MAQNNNE); C5 C6 C9 C26 (CL21B105KBFNNNE); C7 C11 (CL21B103KBANNNC) | https://datasheet.lcsc.com/datasheet/pdf/02336ea48ea44ca18c72517dd3cb7b47.pdf (LCSC copy; SHA-256 identical to the repo file (2026-11-22) (C45783, C28323, C1710)) |
| SFM3300.pdf | 9 | `f7c20973d6592903` | U4 (61300611121); off-board E2 SFM3300-D | https://sensirion.com/media/documents/8ECF0D28/62D13110/Sensirion_Datasheet_SFM3300-AW_SFM3300-D.pdf (SHA-256 identical to the repo file (2026-11-22); product page https://sensirion.com/products/catalog/SFM3300-D lists SFM3304-D as "Successor product") |
| Solomon_SSD1306_Adafruit.pdf | 65 | `d55f875357de96d8` | off-board E3 (SSD1306 controller) | https://cdn-shop.adafruit.com/datasheets/SSD1306.pdf (Adafruit copy of Solomon Systech SSD1306 Rev 1.1; downloaded from this link on 2026-11-22) |
| SRN6045TA.pdf | 4 | `a5144a82a215a917` | L1 (SRN6045TA-150M); L2 (SRN6045TA-100M) | https://www.bourns.com/docs/Product-Datasheets/SRN6045TA.pdf (SHA-256 identical to the repo file (2026-11-22)) |
| TDK_PS_buzzer.pdf | 8 | `a2e091ca64d6cfef` | BZ1 (PS1240P02BT) | NOT VERIFIED: TDK "PS series" (2001); download source was not recorded. product.tdk.com returns 403 from here and the LCSC page for C76871 offers only its own viewer (2026-11-22) |
| TPS54202.pdf | 31 | `297fae8754121e09` | U8 U9 (TPS54202DDCR) | https://www.ti.com/lit/ds/symlink/tps54202.pdf (SHA-256 identical to the repo file (2026-11-22); the link always serves the latest revision) |
| UNIROYAL_0805W8F.pdf | 9 | `11cd644d5d8a34a6` | R1 R3 R7 R8 R9 R11 R12 R13 R14 R17 R24 R26 R27 (0805W8F1002T5E); R2 R4 (0805W8F2002T5E); R5 R6 (0805W8F4701T5E); R10 (0805W8F3602T5E); R15 R18 (0805W8F1004T5E); R21 (0805W8F1003T5E); R23 (0805W8F1003T5E); R30 (0805W8F1003T5E); R22 (0805W8F2212T5E); R28 R29 (0805W8F1001T5E) | https://datasheet.lcsc.com/datasheet/pdf/0a975aaa49b7c97f38a963127be4a823.pdf (LCSC copy; SHA-256 identical to the repo file (2026-11-22) (C17414)) |
| Vishay_SMBJ.pdf | 6 | `1b65b57a43111280` | D2 (SMBJ13CA-E3/52) | https://www.vishay.com/docs/88392/smbj.pdf (downloaded from this link on 2026-11-22) |
| Wonsmart_WS4540-12-NZ03.pdf | 1 | `5808f25cb4e59d31` | off-board E1 WS4540-12-NZ03 | NO PUBLIC LINK: Wonsmart drawing dated 2023-07-12, supplied by the maker or seller; no copy found on the web (2026-11-22) |
| Wurth_61300411121.pdf | 6 | `ce759fb8caee2fd0` | U5 (61300411121) | https://www.we-online.com/components/products/datasheet/61300411121.pdf (SHA-256 identical to the repo file (2026-11-22) (checked in the browser; the site refuses curl)) |
| Wurth_61300421121.pdf | 6 | `4847ecc476b64c3a` | J2 (61300421121) | https://www.we-online.com/components/products/datasheet/61300421121.pdf (SHA-256 identical to the repo file (2026-11-22) (checked in the browser; the site refuses curl)) |
| Wurth_61300611121.pdf | 7 | `c54a02fba6e8df8b` | J3 (61300611121); U4 (61300611121) | https://www.we-online.com/components/products/datasheet/61300611121.pdf (SHA-256 identical to the repo file (2026-11-22) (checked in the browser; the site refuses curl)) |
| Yageo_CC0805_NP0.pdf | 23 | `495287fc811dc714` | C8 C12 (CC0805JRNPOYBN471) | https://datasheet.lcsc.com/datasheet/pdf/4eca2bb4b507f1633db47e922d632b95.pdf (LCSC copy; SHA-256 identical to the repo file (2026-11-22) (C277525)) |
| Yageo_CC0805_X7R.pdf | 29 | `61a606825ab314ea` | C3 C4 C15 C16 C17 C20 C24 (CC0805KRX7R9BB104) | https://datasheet.lcsc.com/datasheet/pdf/23ccee80ee542e7cf156a772bb589942.pdf (LCSC copy; SHA-256 identical to the repo file (2026-11-22) (C49678)) |
| Yageo_RT.pdf | 10 | `0a729d144519a702` | R19 (RT0805BRD07100KL); R20 (RT0805BRD0713K3L) | https://datasheet.lcsc.com/datasheet/pdf/46d7760c582640714d605308a25a220a.pdf (LCSC copy; SHA-256 identical to the repo file (2026-11-22) (C122537, C865182)) |


Off-board items still without a chosen part (E4c mains cord, E9 tubing and T-fitting, E10 knob) have no datasheet here; the reason is in the Status column of `BOM_RevC_v3_offboard.csv`.
Distributor-hosted copies (LCSC, Farnell, Rapid, Adafruit) are used where the maker's own site refuses automated downloads (molex.com timed out on 2026-11-22); each is the maker's own document (title block or "generated by Molex" header), unchanged.
