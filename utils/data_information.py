import pandas as pd

# df_white = pd.read_csv("data/winequality-white.csv", sep=";")
df_red = pd.read_csv("data/winequality-red.csv", sep=";")

# print("White wine data info:", pd.DataFrame.info(df_white))
pd.DataFrame.info(df_red)
print(pd.DataFrame.describe(df_red))