import pandas as pd

df = pd.read_csv('../data/cell-count.csv')

result = df[
    (df['condition'] == 'melanoma') &
    (df['sex'] == 'M') &
    (df['response'] == 'yes') &
    (df['time_from_treatment_start'] == 0)
]['b_cell'].mean()

print(f"{result:.2f}")

# ANSWER: 10206.15