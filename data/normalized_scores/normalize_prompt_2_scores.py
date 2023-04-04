import pandas as pd


df = pd.read_csv(filepath)
df.insert(10, column='normalized_score', value = '')
print(df)



max_final_score = (df['domain1_score'] + df['domain2_score']).max()
print(max_final_score)
min_final_score = (df['domain1_score'] + df['domain2_score']).min()
print(min_final_score)

df['normalized_score'] = ((df['domain1_score'] + df['domain2_score']) - min_final_score)/(max_final_score - min_final_score)

df.to_csv(filepath, index=False)

print(df)
