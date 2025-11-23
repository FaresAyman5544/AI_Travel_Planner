def daily_budget_split(total_budget_egp: float, days: int):
    # Simple split for activities vs food/transport cushion
    activity_per_day = total_budget_egp * 0.6 / max(days, 1)
    cushion_per_day = total_budget_egp * 0.4 / max(days, 1)
    return round(activity_per_day, 2), round(cushion_per_day, 2)

def within_budget(activities, daily_cap):
    total = sum(a.get("avg_cost_egp", 0) for a in activities)
    return total <= daily_cap
