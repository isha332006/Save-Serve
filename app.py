import streamlit as st
import pandas as pd
from model import train_all_models

@st.cache_resource
def load_models():
    return train_all_models()

models, le_day, le_slot, food_list = load_models()

# Load datasets
centers = pd.read_csv("data/donation_centers_expanded.csv")
distances = pd.read_csv("data/restaurant_center_distances.csv")

# UI: Header
st.title("🍛 Save & Serve - Surplus Food Donation Planner")
st.markdown("Help restaurants reduce waste by predicting surplus and recommending nearby donation centers.")

# UI: Input
restaurant = st.selectbox("🏢 Select Restaurant", distances['Restaurant'].unique())
day = st.selectbox("📅 Day", le_day.classes_)
slot = st.selectbox("⏰ Time Slot", le_slot.classes_)
food_selected = st.multiselect("🍲 Select Food Items", food_list)

# On submit
if st.button("Predict & Suggest Centers"):

    if not food_selected:
        st.warning("Please select at least one food item.")
    else:
        day_enc = le_day.transform([day])[0]
        slot_enc = le_slot.transform([slot])[0]

        st.subheader("📦 Predicted Surplus and Suggested Donation Plans:")

        for item in food_selected:
            model = models[item]
            predicted_qty = model.predict([[day_enc, slot_enc]])[0]

           # Define unit and scale for certain items
            unit_map = {
                'Papad': ('pcs', 30),
                'Chapati': ('pcs', 50),
                'Raita': ('bowls', 20),
                'Curd': ('bowls', 15),
                'Salad': ('plates', 20),
                'Pickle': ('pcs', 25)
            }

            # Predict the base quantity
            model = models[item]
            base_prediction = model.predict([[day_enc, slot_enc]])[0]

            # Get display unit and scale
            unit, scale = unit_map.get(item, ('kg', 1))

            # Apply scaling and round to realistic quantity
            final_prediction = int(round(base_prediction * scale))

            # Display the clean result
            st.markdown(f"### 🍽️ {item}: **{final_prediction} {unit}** predicted")



            # Match donation centers that accept this item
            accepted = centers[centers['AcceptedFoods'].str.contains(rf'\b{item}\b', case=False, na=False)]

            rest_dist = distances[distances['Restaurant'] == restaurant]
            merged = pd.merge(accepted, rest_dist, on='CenterName')
            top_centers = merged.sort_values(by='Distance_km')

            if top_centers.empty:
                st.warning(f"No donation centers found for **{item}**.")
            else:
                st.markdown(f"Suggested centers for **{item}**:")

                remaining = final_prediction  # how much we have left to send

                for _, row in top_centers.iterrows():
                    if remaining <= 0:
                        break

                    center_name = row['CenterName']
                    contact = row['Contact']
                    distance = row['Distance_km']
                    center_capacity = int(row['Capacity_kg'])

                    to_send = min(remaining, center_capacity)
                    remaining -= to_send

                    st.write(f"🏥 {center_name} — {distance} km away")
                    st.write(f"📦 Sending: **{to_send} {unit}**")
                    st.write(f"📞 Contact: {contact}")
                    st.write("---")

                    if remaining > 0:
                        st.info(f"⚠️ {remaining} {unit} of **{item}** still remaining to be allocated.")

