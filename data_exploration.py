import pandas as pd

df = pd.read_csv("spam.csv", encoding='latin-1')

df = df[['v1', 'v2']]
df.columns = ['label', 'text']
print(df.head())
print(df.shape)
print(df.columns)
print(df['label'].value_counts())

def get_data():
    return df

if __name__ == "__main__":
    df = get_data()
