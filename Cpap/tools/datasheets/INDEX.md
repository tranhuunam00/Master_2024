# Datasheets — SiPAP Rev C v3

Each file and the parts it covers, from the `Datasheet` column of `hardware/CPAP_VSSM/Fab_RevC_v3/BOM_RevC_v3.csv` and `BOM_RevC_v3_offboard.csv`.
`tools/layout_v3/fab_v3.py` stops if a BOM line names a file that is not here.
Source: the URL when the file was downloaded on 2026-11-22. "earlier session" means it was already in the repo and no URL was recorded.

| File | Pages | SHA-256 (first 16) | Used for | Source |
|---|---|---|---|---|
| 2N7002_Diodes.pdf | 5 | `3984debf5b293762` | Q3 (2N7002) | earlier session |
| AO3401A.pdf | 5 | `0d8e3261ae280e00` | Q2 (AO3401A) | earlier session |
| AO4407A.pdf | 5 | `ef4b3686383ca726` | Q1 Q4 (AO4407A) | earlier session |
| Bourns_3224.pdf | 3 | `db9857638bb6b034` | RV1 (3224W-1-503E) | https://www.bourns.com/docs/Product-Datasheets/3224.pdf |
| Bourns_MF-R.pdf | 10 | `d22f0f06c8909784` | F1 (MF-R400) | earlier session |
| Bourns_MF-SM.pdf | 7 | `c55c1edf8cba275a` | **not used by the current BOM** | earlier session |
| Bourns_SMBJ.pdf | 6 | `3303bff29fd46a97` | D2 (SMBJ13CA-E3/52) | earlier session |
| FH_0805CG.pdf | 30 | `705023d3be045414` | C19 (0805CG750J500NT); C22 (0805CG560J500NT) | LCSC copy for C1838/C1828 |
| JSCJ_2N7002.pdf | 5 | `7941fb423af7c6c6` | Q3 (2N7002) | LCSC copy for C8545 |
| LM393_TI.pdf | 56 | `a1f63934820e27c1` | U7 (LM393DR2G) | earlier session |
| MPX5010_b.pdf | 12 | `796fbf22bebe0df7` | U2 U3 (MPXV5010GC6T1) | earlier session |
| Murata_GRM21BR61H106KE43.pdf | 31 | `ff8de8901544f5f4` | C10 C13 C14 (GRM21BR61H106KE43L) | LCSC copy for C440198 |
| PEC11R.pdf | 4 | `ac88f657e611bc82` | U6 (PEC11R-4225F-S0024) | earlier session |
| PJ-102AH.pdf | 3 | `269ff7d1d8b7130a` | J1 (PJ-102AH) | earlier session |
| Panasonic_ERJP06.pdf | 9 | `441511405b982033` | R16 (ERJP06F3303V) | LCSC copy for C441993 |
| Panasonic_FC.pdf | 17 | `42d91ae29ab6122d` | C25 (EEU-FC1E221) | earlier session |
| SFM3300.pdf | 9 | `f7c20973d6592903` | U4 (61300611121); off-board E2 SFM3300-D | earlier session |
| SRN6045TA.pdf | 4 | `a5144a82a215a917` | L1 (SRN6045TA-150M); L2 (SRN6045TA-100M) | earlier session |
| Samsung_CL21.pdf | 84 | `96baebe4bd0ce363` | C23 (CL21A226MAQNNNE); C1 C2 C18 C21 C27 (CL21A226MAQNNNE); C5 C6 C9 C26 (CL21B105KBFNNNE); C7 C11 (CL21B103KBANNNC) | LCSC copy for C45783/C28323/C1710 |
| TDK_PS_buzzer.pdf | 8 | `a2e091ca64d6cfef` | BZ1 (PS1240P02BT) | earlier session |
| TPS54202.pdf | 31 | `297fae8754121e09` | U8 U9 (TPS54202DDCR) | earlier session |
| UNIROYAL_0805W8F.pdf | 9 | `11cd644d5d8a34a6` | R1 R3 R7 R8 R9 R11 R12 R13 R14 R17 R24 R26 R27 (0805W8F1002T5E); R2 R4 (0805W8F2002T5E); R5 R6 (0805W8F4701T5E); R10 (0805W8F3602T5E); R15 R18 (0805W8F1004T5E); R21 (0805W8F1003T5E); R23 (0805W8F1003T5E); R30 (0805W8F1003T5E); R22 (0805W8F2212T5E); R28 R29 (0805W8F1001T5E) | LCSC copy for C17414 (same file for every 0805W8F code) |
| Vishay_SMBJ.pdf | 6 | `1b65b57a43111280` | D2 (SMBJ13CA-E3/52) | https://www.vishay.com/docs/88392/smbj.pdf |
| Wonsmart_WS4540-12-NZ03.pdf | 1 | `5808f25cb4e59d31` | off-board E1 WS4540-12-NZ03 | earlier session |
| Wurth_61300411121.pdf | 6 | `ce759fb8caee2fd0` | U5 (61300411121) | earlier session |
| Wurth_61300421121.pdf | 6 | `4847ecc476b64c3a` | J2 (61300421121) | earlier session |
| Wurth_61300611121.pdf | 7 | `c54a02fba6e8df8b` | J3 (61300611121); U4 (61300611121) | earlier session |
| Yageo_CC0805_NP0.pdf | 23 | `495287fc811dc714` | C8 C12 (CC0805JRNPOYBN471) | LCSC copy for C277525 |
| Yageo_CC0805_X7R.pdf | 29 | `61a606825ab314ea` | C3 C4 C15 C16 C17 C20 C24 (CC0805KRX7R9BB104) | LCSC copy for C49678 |
| Yageo_RT.pdf | 10 | `0a729d144519a702` | R19 (RT0805BRD07100KL); R20 (RT0805BRD0713K3L) | LCSC copy for C122537/C865182 (Yageo RT V.16) |
| esp32-s3-wroom-1_datasheet_v1.8.pdf | 53 | `27d71971da07c280` | U1 (ESP32-S3-WROOM-1-N8R8) | earlier session |
| onsemi_LM393.pdf | 13 | `2fd8e714267f0539` | U7 (LM393DR2G) | https://www.onsemi.com/pdf/datasheet/lm393-d.pdf |

Off-board items with no part chosen yet (E3 OLED, E4–E10) have no datasheet here; see `BOM_RevC_v3_offboard.csv`.
