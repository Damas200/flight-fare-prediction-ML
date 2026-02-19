def clean_data(df):
    df = df.drop_duplicates()
    df = df.ffill()
    print("Data cleaned successfully.")
    return df
