import json
import time
import streamlit as st
import pandas as pd

from app.rag.rag_pipeline import init_pipeline
from app.chains.day_plan_chains import build_chain
from app.tools.cost_utils import daily_budget_split

def main():
    
    st.set_page_config(page_title="AI Travel Planner", page_icon="🧭", layout="wide")
    st.title("🧭 AI Travel Planner — Booking-style")

    with st.sidebar:
        st.header("Trip inputs")
        destination = st.text_input("Destination city (e.g., Cairo, Alexandria)", value="Cairo")
        budget_egp = st.number_input("Total budget (EGP)", min_value=0.0, value=6000.0, step=500.0)
        days = st.number_input("Trip length (days)", min_value=1, value=3, step=1)
        prefs = st.multiselect("Preferred themes", ["history", "museum", "culture", "shopping", "food", "walking", "landmark"], default=["history","culture","food"])

    st.divider()

    pipeline = init_pipeline()
    chain, parser = build_chain()

    # RAG search
    st.subheader("Recommended places")
    places = pipeline.search_places(destination, budget_egp / max(days,1), query=",".join(prefs))

    if not places:
        st.warning("No places found in the local corpus for this destination. Add more data to data/places_sample.json.")
    else:
        col_cards = st.columns(3)
        for i, p in enumerate(places[:9]):
            with col_cards[i % 3]:
                st.markdown(f"**{p['name']}** ({p['type']})")
                st.caption(p["description"])
                st.badge(p["best_time"])
                st.write(f"Approx. cost: EGP {p.get('avg_cost_egp', '—')}")
                st.write(f"Tags: {', '.join(p.get('tags', []))}")

    st.divider()

    # Build itinerary
    st.subheader("Itinerary planner")
    activity_cap, cushion_cap = daily_budget_split(budget_egp, int(days))
    daily_caps = [activity_cap for _ in range(int(days))]

    if st.button("Generate plan"):
        places_json = json.dumps(places, ensure_ascii=False)
        with st.spinner("Generating your plan..."):
            raw = chain.run(
                destination=destination,
                days=int(days),
                daily_caps=daily_caps,
                places_json=places_json,
                format_instructions=parser.get_format_instructions()
            )
        try:
            parsed = parser.parse(raw)
        except Exception:
            parsed = {
                "itinerary": [{"day": d+1, "activity": "City walk + local market", "approx_cost_egp": activity_cap * 0.8} for d in range(int(days))],
                "notes": "Fallback plan. Consider adding more places and re-generating."
            }

        # Table view
        st.subheader("Travel table")
        df = pd.DataFrame(parsed["itinerary"])
        if "day" in df.columns and "activity" in df.columns and "approx_cost_egp" in df.columns:
            df["approx_cost_egp"] = df["approx_cost_egp"].apply(lambda x: round(float(x), 2) if isinstance(x, (int,float,str)) else x)
            st.dataframe(df, use_container_width=True)
        else:
            st.write(parsed["itinerary"])

        st.subheader("Notes")
        st.write(parsed.get("notes", ""))

        # Export JSON
        output = {
            "destination": destination,
            "days": int(days),
            "budget_egp": budget_egp,
            "daily_caps": daily_caps,
            "itinerary": parsed.get("itinerary", []),
            "notes": parsed.get("notes", ""),
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ")
        }
        st.download_button("Download itinerary JSON", data=json.dumps(output, ensure_ascii=False, indent=2), file_name="itinerary.json", mime="application/json")


if __name__ == "__main__":
    main()
