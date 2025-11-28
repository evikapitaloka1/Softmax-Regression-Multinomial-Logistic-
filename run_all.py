import subprocess

files = [
    "proses_pertama.py",
    "proses_kedua.py",
    "proses_ketiga.py",
    "proses_keempat.py"
]

for f in files:
    print("\n====================================")
    print("MENJALANKAN :", f)
    print("====================================\n")
    subprocess.run(["python", f])
