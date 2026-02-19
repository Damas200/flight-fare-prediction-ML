import pandas as pd


def create_features(df):

    df.columns = df.columns.str.strip()

    if "Departure Date & Time" in df.columns:
        df["Departure Date & Time"] = pd.to_datetime(
            df["Departure Date & Time"]
        )

        df["Departure Month"] = df["Departure Date & Time"].dt.month
        df["Departure Day"] = df["Departure Date & Time"].dt.day
        df["Departure Weekday"] = df["Departure Date & Time"].dt.weekday

    if "Arrival Date & Time" in df.columns:
        df["Arrival Date & Time"] = pd.to_datetime(
            df["Arrival Date & Time"]
        )

        df["Arrival Hour"] = df["Arrival Date & Time"].dt.hour

    # Drop raw datetime columns
    df = df.drop(
        columns=[
            col for col in [
                "Departure Date & Time",
                "Arrival Date & Time"
            ] if col in df.columns
        ]
    )

    print("Feature engineering completed.")
    return df
