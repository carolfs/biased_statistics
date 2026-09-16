import pickle
import numpy as np
from scipy.stats import differential_entropy
import matplotlib.pyplot as plt

novel_weight_flnm = "three_stage_task_weights.bin"

wlow, whmid, whtop = [], [], []
with open(novel_weight_flnm, "rb") as inpf:
    while True:
        try:
            samples = pickle.load(inpf)
        except:
            break
        else:
            wlow.append(differential_entropy(samples["wlow"]))
            whmid.append(differential_entropy(samples["whmid"]))
            whtop.append(differential_entropy(samples["whtop"]))

for name, e in [("high top", whtop), ("high middle", whmid), ("low", wlow)]:
    e = np.array(e)
    print(f"{name}: median {np.median(e):.3f}  "
          f"above -0.25: {(e > -0.25).mean():.2f}")

plt.hist(wlow, density=True, bins="auto", alpha=0.5, label=f"Low-effort")
plt.hist(whmid, density=True, bins="auto", alpha=0.5, label=f"High-effort middle")
plt.hist(whtop, density=True, bins="auto", alpha=0.5, label=f"High-effort top")
plt.legend(loc="best", title="Choice")
plt.ylabel("Relative frequency")
plt.xlabel("Entropy")
plt.savefig("three_stage_entropy.pdf")
plt.close()
print(len(wlow))
