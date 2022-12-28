import pandas as pd

text_arr = []
overview_arr = []
for i in range(0, 20, 2):
    df = pd.read_excel("scrum.xlsx", sheet_name=i)
    df2 = pd.read_excel("scrum.xlsx", sheet_name=i + 1)
    overview = list(df["Meeting Overview"].values)
    texts = list(
        df2["Manual Transcript (corrected transcript without guidelines)"].values
    )
    i = 0
    while i < len(texts):
        if not isinstance(texts[i], str):
            del texts[i]
        else:
            i += 1
    text_arr.append(". ".join(texts))
    overview_arr.append(overview[0])

df = pd.DataFrame()
df["text"] = text_arr
df["overview"] = overview_arr
df.to_csv("train.csv", index=False)
