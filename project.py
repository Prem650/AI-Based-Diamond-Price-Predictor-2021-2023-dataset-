import pandas as pd
import numpy as np
import time

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

print("Loading data and training model...")
start = time.time()

# Load dataset
df = pd.read_csv("diamonds.csv")
df = df.drop(columns=['Unnamed: 0'])

# Create size feature
df['size'] = df['x'] * df['y'] * df['z']

# Remove invalid diamonds
df = df[
    (df['x'] > 0) &
    (df['y'] > 0) &
    (df['z'] > 0) &
    (df['depth'] > 50) &
    (df['depth'] < 75) &
    (df['table'] > 50) &
    (df['table'] < 70)
]

print("Dataset size:", len(df))

# Features
X = df[['carat','cut','color','clarity','depth','table','size']]
y = df['price']

# Log transform
y_log = np.log1p(y)

# Split data
X_train, X_test, y_train_log, y_test_log = train_test_split(
    X, y_log, test_size=0.2, random_state=42
)

categorical_features = ['cut','color','clarity']
numerical_features = ['carat','depth','table','size']

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
        ('num', StandardScaler(), numerical_features)
    ]
)

# More accurate model
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', GradientBoostingRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=4,
        random_state=42
    ))
])

print("Training model...")
model.fit(X_train, y_train_log)

# Predictions
y_pred_log = model.predict(X_test)
y_pred = np.expm1(y_pred_log)

mae = mean_absolute_error(np.expm1(y_test_log), y_pred)
r2 = r2_score(np.expm1(y_test_log), y_pred)

cv_r2 = cross_val_score(model, X_train, y_train_log, cv=5, scoring='r2').mean()

print("Model Ready")
print("MAE:", mae)
print("R2:", r2)
print("CV R2:", cv_r2)
print("Training time:", time.time() - start)

# FASTAPI SERVER

app = FastAPI(title="Diamond Price Predictor")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class DiamondInput(BaseModel):
    carat: float
    cut: str
    color: str
    clarity: str
    depth: float
    table: float
    x: float
    y: float
    z: float

@app.post("/predict")
def predict_price(diamond: DiamondInput):

    if diamond.x <= 0 or diamond.y <= 0 or diamond.z <= 0:
        raise HTTPException(status_code=400, detail="Invalid dimensions")

    size = diamond.x * diamond.y * diamond.z

    input_df = pd.DataFrame([{
        'carat': diamond.carat,
        'cut': diamond.cut,
        'color': diamond.color,
        'clarity': diamond.clarity,
        'depth': diamond.depth,
        'table': diamond.table,
        'size': size
    }])

    pred_log = model.predict(input_df)[0]
    prediction = np.expm1(pred_log)

    return {"predicted_price": round(prediction,2)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
