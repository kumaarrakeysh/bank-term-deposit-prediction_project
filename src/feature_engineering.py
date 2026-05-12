def add_features(df):

    df['age_balance'] = (
        df['age'] * df['balance']
    )

    df['campaign_duration'] = (
        df['campaign'] * df['duration']
    )

    return df