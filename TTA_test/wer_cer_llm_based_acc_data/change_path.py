import os

root="/your/path"

scp_files = [
    f"{root}/IIPL_Flitto/TTA_test/wer_cer_llm_based_acc_data/CN/CN_wav.scp",
    f"{root}/IIPL_Flitto/TTA_test/wer_cer_llm_based_acc_data/EN/EN_wav.scp",
    f"{root}/IIPL_Flitto/TTA_test/wer_cer_llm_based_acc_data/JP/JP_wav.scp",
    f"{root}/IIPL_Flitto/TTA_test/wer_cer_llm_based_acc_data/KR/KR_wav.scp",
]

for scp_path in scp_files:
    with open(scp_path, "r", encoding="utf-8") as infile:
        lines = infile.readlines()

    updated_lines = [line.replace("/path/to/your", f"{root}") for line in lines]

    with open(scp_path, "w", encoding="utf-8") as outfile:
        outfile.writelines(updated_lines)

print("All .scp files have been updated.")
