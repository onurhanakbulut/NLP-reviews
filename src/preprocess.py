import pandas as pd








#erkek dataframe
df1 = pd.read_csv('../data/raw/reviews_erkek_esofman.csv')

df2 = pd.read_csv('../data/raw/reviews_erkek_kazak.csv')

df3 = pd.read_csv('../data/raw/reviews_erkek_tshirt.csv')


df_m = pd.concat([df1, df2, df3], ignore_index=True)

df_m['gender'] = 1 


#kadın dataframe
df4 = pd.read_csv('../data/raw/reviews_kadin_jogger.csv')

df5 = pd.read_csv('../data/raw/reviews_kadin_kazak.csv')

df6 = pd.read_csv('../data/raw/reviews_kadin_tayt.csv')

df_f = pd.concat([df4, df5, df6], ignore_index=True)

df_f['gender'] = 0


#ana dataframe
df = pd.concat([df_m, df_f], ignore_index=True)
df.to_csv('all_rew.csv', index=False, encoding='utf-8')



#preprocessing
print(df.info())
print(df.describe())

##duplicate check
print(df.duplicated(subset=['comment']).sum())          #274
df = df.drop_duplicates(subset=['comment'], keep='first')   #13296 ---> 13022


#-------------sadece sembolden oluşan var mı---------------------

#df['only_sembol'] = df['comment'].str.match(r"^[^a-zA-ZçğıöşüÇĞİÖŞÜ]+$")

#print(df['only_sembol'].describe())


df.to_csv('all_rew_clean.csv', index=True, encoding='utf-8')











































