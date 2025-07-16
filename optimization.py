from scipy.optimize import linprog
import numpy as np

def optimize_resources(admissions, icu_admissions, ventilators_available, beds_available):
    """
    Goal: allocate beds and ventilators to meet demand minimizing unmet needs.
    Variables: x = number of patients allocated beds
               y = number of patients allocated ventilators
    Constraints:
    - x >= icu_admissions (ICU patients need beds)
    - y >= ventilator demand
    - x <= beds_available
    - y <= ventilators_available
    Minimize unmet patients (not allocated)
    """

    # Objective: minimize unmet ICU patients and ventilator shortfall
    # For demo, simplified: just return if resources meet demands
    unmet_icu = max(0, icu_admissions - beds_available)
    unmet_vent = max(0, int(icu_admissions * 0.7) - ventilators_available)

    print(f"Unmet ICU bed demand: {unmet_icu}")
    print(f"Unmet ventilator demand: {unmet_vent}")

    if unmet_icu > 0 or unmet_vent > 0:
        print("Warning: Resources insufficient!")
    else:
        print("Resources sufficient to cover predicted demands.")

if __name__ == "__main__":
    # Sample numbers
    optimize_resources(admissions=100, icu_admissions=20, ventilators_available=15, beds_available=25)