from scipy.io import loadmat
import pandas as pd

m = loadmat("mytst.mat")["d"]
keys = [k for k in m[0][0].view().dtype.fields]
df = {k: [] for k in keys}
part = 0
while True:
    try:
        partm = m[0][part]
        part += 1
        assert keys == [k for k in partm.view().dtype.fields]
        for i, k in enumerate(keys):
            try:
                df[k] += list(partm.view()[0][0][i].flatten())
            except:
                print("Error getting data", part, i, k)
                raise ValueError()
    except IndexError:
        break
df = pd.DataFrame(df)
df = df[df.abort == 0] # Eliminate aborted trials
df.to_csv("two_stage_task_data.csv", index=False)
