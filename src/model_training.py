import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from src.data_loader import load_data
from src.feature_engineering import add_features
from src.preprocessing import create_preprocessor

def train_model():

    df = load_data('data/raw/bank-full.csv')

    df['y'] = df['y'].map({'yes':1,'no':0})

    df = add_features(df)

    X = df.drop('y', axis=1)
    y = df['y']

    categorical_features = X.select_dtypes(include=['object']).columns.tolist()
    numeric_features = X.select_dtypes(exclude=['object']).columns.tolist()

    preprocessor = create_preprocessor(
        numeric_features,
        categorical_features
    )

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42
    )

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    pipeline.fit(X_train, y_train)

    score = pipeline.score(X_test, y_test)

    print(f'Accuracy: {score}')

    joblib.dump(pipeline, 'models/best_model.pkl')

    print('Model Saved!')