import os

data_root = r"C:\Users\KIIT\Desktop\GT-NET\data"
for part in ("train","val"):
    print(f"\n== {part.upper()} ==")
    base = os.path.join(data_root, part)
    if not os.path.exists(base):
        print("  folder not found:", base)
        continue
    for cls in sorted(os.listdir(base)):
        p = os.path.join(base, cls)
        if os.path.isdir(p):
            cnt = len([f for f in os.listdir(p) if f.lower().endswith(('.jpg','.jpeg','.png'))])
            print(f"  {cls:12s}: {cnt}")
