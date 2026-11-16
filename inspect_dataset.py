
import pandas as pd

df = pd.read_csv("data/fake_job_postings.csv")

print("Dataset Shape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nFraud Distribution:")
print(df["fraudulent"].value_counts())

print("\nFraud Percentage:")
print(df["fraudulent"].value_counts(normalize=True) * 100)

print("\nSample Records:")
print(df[["title", "description", "fraudulent"]].head())
