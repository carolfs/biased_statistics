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
# # Select only participants with both sessions
# subj = []
# for s, sdf in df.groupby("subj"):
#     if 1 in sdf.meas.unique() and 3 in sdf.meas.unique():
#         subj.append(s)
# df = df[df.subj.isin(subj)]
# newkeys = ["subject", "measurment", "trial", "choice1", "choice2", "RT1", "RT2", "reward", "transition", "second_stage_state"]
# df["subject"] = df["subj"]
# df["measurment"] = df["meas"]
# df["trial"] = df["trl"]
# df["choice1"] = df["ch1"]
# df["choice2"] = df["ch2"]
# df["RT1"] = df["rt1"]
# df["RT2"] = df["rt2"]
# df["reward"] = df["rw"]
# df["transition"] = df["tran"]
# df["second_stage_state"] = df["state"]
# for col in df.columns:
#     if col not in newkeys:
        # del df[col]
df.to_csv("two_stage_task_data.csv", index=False)
