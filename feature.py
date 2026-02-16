def get_features_and_target(df):

    FEATURES = ["area", "bedrooms", "bathrooms"]
    TARGET = "price"

    X = df[FEATURES]
    y = df[TARGET]
    return X, y, FEATURES, TARGET