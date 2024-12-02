def get_splits(CSV_FILE,  test_split=0.2, val_split=0.1):
    df = pd.read_csv(CSV_FILE)
    X = df['filename'].to_list()
    y = df['speaker_id'].to_list()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_split, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_split, random_state=42)
    return [
        (X_train, y_train),
        (X_val, y_val),
        (X_test, y_test)
    ]