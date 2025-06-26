import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

def train_all_models(csv_path='data/human_readable_food_logs (1).csv'):
    # Load the food log data
    df = pd.read_csv(csv_path)

    # Encode categorical features
    le_day = LabelEncoder()
    le_slot = LabelEncoder()
    df['Day_enc'] = le_day.fit_transform(df['Day'])
    df['Slot_enc'] = le_slot.fit_transform(df['TimeSlot'])

    # Detect all food item columns dynamically
    food_items = [col for col in df.columns if col not in ['Day', 'TimeSlot', 'Day_enc', 'Slot_enc']]

    # Train one model per food item
    models = {}
    for item in food_items:
        X = df[['Day_enc', 'Slot_enc']]
        y = df[item]
        model = RandomForestRegressor()
        model.fit(X, y)
        models[item] = model

    # Return all models and encoders
    return models, le_day, le_slot, food_items
