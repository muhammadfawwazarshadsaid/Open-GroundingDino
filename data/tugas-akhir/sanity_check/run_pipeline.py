import subprocess

steps = [
    "1_check_dataformat.py",
    "2_fix_dataformat.py",
    "3_check_alignment.py",
    "4_fix_alignment.py",
    # "5_class_distribution.py",  # skip
    "6_check_bbox.py",
    "7_fix_bbox.py",
    "8_setup_dataset.py"
]

print("=== AUTO PIPELINE START ===\n")
for step in steps:
    print(f"🚀 Running {step} ...\n")
    subprocess.run(["python3", step])
    print("\n" + "="*50 + "\n")

print("✅ Pipeline selesai, dataset final siap dipakai training.")
