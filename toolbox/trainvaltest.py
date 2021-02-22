from sklearn.model_selection import train_test_split

def train_test_validation(X, y, test_ratio=0.8):
    """
    splits X, and y in to X_train, X_val, X_test, y_train, y_val, y_test
    Set ratio with ratio param. 
    """

    X_train, X_test_val, y_train, y_test_val = train_test_split(X, y, train_size=test_ratio, \
                                                                test_size=(1-test_ratio))
    X_val, X_test, y_val, y_test = train_test_split(X_test_val, y_test_val, train_size=0.5, test_size=0.5)
    return X_train, X_val, X_test, y_train, y_val, y_test