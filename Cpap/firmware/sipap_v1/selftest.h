// SiPAP Rev C — board bring-up self-test (Serial 115200, interactive).
// Entered when the encoder button is held during power-on, or by sending "TEST" on Serial
// within 2 s of boot. Prints PASS / FAIL / SKIP per check and a summary.
#pragma once
class U8G2;
void runSelfTest(U8G2 &oled);
