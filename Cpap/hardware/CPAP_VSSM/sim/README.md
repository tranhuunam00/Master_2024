# Mô phỏng hoạt động bình thường — SiPAP Rev C (2026-11-22)

Chạy bằng Altium Mixed Sim (ngspice) qua eda-agent `sim_run_deck`. LM393 dùng model PSpice của TI (SLCJ016B, subckt LM2903B).

| File | Nội dung |
|---|---|
| `sim_normal_operation.cir` | Board như hiện tại: cắm adapter 12 V, soft-start 2 buck, reset ESP32, SFM3300, LM393 giữ Q1 dẫn, PWM blower 45 %, cảm biến áp 10 cmH2O |
| `sim_q1_slewcap_whatif.cir` | Thử thêm tụ 47 nF G–D cho Q1 để giảm dòng nạp C25 — **đã loại**: Q1 tự dẫn lúc cắm adapter, trước khi LM393 có nguồn |
| `sim_normal_overview.png` | Dạng sóng 0–25 ms |
| `sim_q1_turn_on.png` | Phóng to lúc Q1 đóng |

## Mô hình hoá
- TPS54202 (chưa có model SPICE): nguồn ổn áp theo hành vi. Soft-start 5 ms, UVLO bật 4.2 V và tắt 3.7 V (TI SLVSD26C tr.5). Hiệu suất giả định 90 % / 85 %.
- AO4407A, AO3401A, 2N7002 (AOS không công bố model):
  - Q1 mô hình theo luật bình phương, Vth 2 V, RDS(on) 17 mΩ (AO4407A DS tr.1), kèm Cgs/Cgd;
  - Q4, Q2, Q3 là công tắc theo ngưỡng Vgs trong datasheet.
- MPXV5010: Vout = Vs·(0.09·P + 0.04), dòng tiêu thụ 5 mA (NXP DS tr.2).
- Adapter: tụ ngõ ra 470 µF, cáp 0.15 Ω / 1 µH. Đây là giả định.
- Phụ tải giả định: 3V3 120 mA, SFM3300 10 mA (< 50 mW, Sensirion DS tr.3), blower 0.1 + 1.5·(VSR/3.3)² A.
- LM393 lúc chưa có nguồn: ngõ ra hở (cực thu hở). Thêm một công tắc theo VCC vì model TI không mô tả trạng thái này.

## Kết quả chính
- Adapter cắm vào: Q4 dẫn, RPP_G ≈ 6 V (Vgs −6 V). VIN12V lên 12 V sau ~1 ms.
- 5V và 3V3 bắt đầu khi VIN > 4.2 V, đạt 5.08 V / 3.29 V sau 5 ms soft-start.
- LM393 có nguồn (5V ≈ 2 V) lúc ~3 ms: COMP_OUT ≈ 0.04 V, GATE ≈ 6 V, Q1 dẫn.
- Q1 đóng vào C25 220 µF trống: dòng đỉnh 20 A trong ~145 µs, VIN12V sụt tới 8.4 V. Vẫn cao hơn UVLO tắt 3.7 V, nên không reset. AO4407A chịu được dòng xung 60 A (DS tr.1).
- Trong ~3 ms đầu, trước khi LM393 có nguồn, SAFETY_TRIP lên ~2.9 V. ESP32 lúc đó còn đang reset nên không đọc được, và không gây báo động.
- Áp 10 cmH2O: PRESS_ADC 0.43 V, COMP_REF 0.89 V (ngưỡng ~25 cmH2O), SAFETY_TRIP 0.01 V.
- PWM 45 %: VSR 1.48 V, gợn 40 mV p-p, ổn định sau ~3 ms (τ = 1 ms). FG ở IO11 dao động 0.30 ↔ 3.29 V.
- Trạng thái ổn định: VIN12V 11.93 V, BLOWER_V12 11.93 V, SFM_5V 5.08 V.

Không thay được bring-up thật: sai số linh kiện, adapter thật, blower thật và firmware chưa được mô phỏng.
