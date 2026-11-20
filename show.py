import json
nb=json.load(open("Part_II_Deep_Learning.ipynb",encoding="utf-8"))
for i,c in enumerate(nb["cells"]):
    if c["cell_type"]!="code": continue
    for o in c.get("outputs",[]):
        t=o.get("text") or o.get("data",{}).get("text/plain")
        if t: print(f"--- cell {i} ---"); print("".join(t)[:2500])
