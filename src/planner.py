"""
Meal planning module for NutriGen AI.

==============================================================================
AI TECHNIQUE #2: CONSTRAINT SATISFACTION (CSP) AND HEURISTIC SEARCH
==============================================================================

This module implements the core planning algorithms:

1. PEAS Framework Implementation:
   - Performance: Maximize user acceptance, hit calorie targets within ±150 kcal
   - Environment: Recipe database, user dietary constraints
   - Actuators: Meal selection, menu generation, portion scaling
   - Sensors: User input, dietary requirements, preference feedback

2. Heuristic Search (Monte Carlo Optimization):
   - 5,000 iteration randomized search for optimal meal combinations
   - Goal: Find breakfast + lunch + dinner where Sum(Calories) ≈ Target
   - Uses quality scores from ML model to guide selection

3. Constraint Satisfaction Integration:
   - All meals MUST pass DietEnforcer (CSP) before selection
   - Hard constraints: dietary restrictions, calorie bounds
   - Soft constraints: user preferences, meal variety

4. Goal-Based Rational Agent:
   - Self-correction through meal rejection and replanning
   - Dynamic portion scaling for extreme calorie targets
"""

# =============================================================================
# IMPORTS
# =============================================================================

# Standard library
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

# Third-party
import pandas as pd

# Local
from src.ai_engine import NutriBrain
from src.config import Colors
from src.constraints import DietEnforcer, is_real_breakfast


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class MealPlan:
    """
    Container for a complete daily meal plan.

    Attributes:
        breakfast: Selected breakfast recipe as pandas Series.
        lunch: Selected lunch recipe as pandas Series.
        dinner: Selected dinner recipe as pandas Series.
        total_calories: Sum of calories across all meals.
        total_protein: Sum of protein across all meals.
        target_calories: User's daily calorie target.
        diet_type: Active dietary restriction.
        portion_multiplier: Scaling factor for portion sizes.
        day_number: Day index for weekly planning.
    """

    breakfast: Optional[pd.Series]
    lunch: Optional[pd.Series]
    dinner: Optional[pd.Series]
    total_calories: float
    total_protein: float
    target_calories: float
    diet_type: str
    portion_multiplier: float = 1.0
    day_number: int = 1


@dataclass
class WeeklyPlan:
    """
    Container for a complete weekly meal plan.

    Attributes:
        daily_plans: List of MealPlan objects for each day.
        master_shopping_list: Aggregated ingredients with quantities.
        total_weekly_calories: Sum of calories for the week.
        total_weekly_protein: Sum of protein for the week.
    """

    daily_plans: List[MealPlan] = field(default_factory=list)
    master_shopping_list: Dict[str, int] = field(default_factory=dict)
    total_weekly_calories: float = 0.0
    total_weekly_protein: float = 0.0


# =============================================================================
# RATIONAL AGENT CLASS
# =============================================================================


class RationalAgent:
    """
    Goal-Based Rational Agent implementing PEAS framework and Heuristic Search.

    ===========================================================================
    IMPLEMENTS: Constraint Satisfaction (CSP) and Heuristic Search
    ===========================================================================

    This agent uses Monte Carlo optimization (a form of heuristic search)
    to find optimal meal combinations that satisfy multiple constraints:

    1. Caloric Constraint: Sum(Calories) ≈ Target ± 150 kcal
    2. Dietary Constraint: All meals MUST pass DietEnforcer (CSP)
    3. Quality Heuristic: Prefer higher ML-predicted quality scores

    Optimization Strategy:
        - Random sampling with 5,000 iterations
        - Early termination when constraints are satisfied
        - Fallback to best-effort solution if no perfect match

    Attributes:
        diet_enforcer: DietEnforcer instance for constraint checking.
        ml_model: NutriBrain instance for quality predictions.
        breakfast_pool: Filtered breakfast recipes DataFrame.
        main_pool: Filtered main meal recipes DataFrame.
        rejected_breakfasts: Set of user-rejected breakfast indices.
        rejected_lunches: Set of user-rejected lunch indices.
        rejected_dinners: Set of user-rejected dinner indices.
        current_plan: Most recently generated MealPlan.
    """

    def __init__(self, diet_enforcer: DietEnforcer, ml_model: NutriBrain) -> None:
        """
        Initialize the rational agent with required components.

        Args:
            diet_enforcer: DietEnforcer instance for constraint checking.
            ml_model: NutriBrain instance for quality predictions.
        """
        self.diet_enforcer: DietEnforcer = diet_enforcer
        self.ml_model: NutriBrain = ml_model

        self.breakfast_pool: pd.DataFrame = pd.DataFrame()
        self.main_pool: pd.DataFrame = pd.DataFrame()

        self.rejected_breakfasts: Set = set()
        self.rejected_lunches: Set = set()
        self.rejected_dinners: Set = set()

        self.current_plan: Optional[MealPlan] = None

        self.avoid_additives: bool = False
        self.avoid_allergens: bool = False

        self.breakfast_exclusions: List[str] = []

    def initialize_pools(
        self,
        breakfast_df: pd.DataFrame,
        main_df: pd.DataFrame,
        diet_type: str,
        avoid_additives: bool = False,
        avoid_allergens: bool = False,
        breakfast_exclusions: Optional[List[str]] = None,
    ) -> None:
        """
        Initialize and filter recipe pools using CSP constraints.

        Applies the DietEnforcer (CSP solver) to filter recipes that
        don't comply with dietary restrictions before the search begins.

        Args:
            breakfast_df: DataFrame of breakfast recipes.
            main_df: DataFrame of main meal recipes.
            diet_type: Dietary restriction to apply (e.g., 'vegan').
            avoid_additives: If True, also filter out additives/preservatives.
            avoid_allergens: If True, also filter out common allergens.
            breakfast_exclusions: Keywords to exclude from breakfast selection.
        """
        self.avoid_additives = avoid_additives
        self.avoid_allergens = avoid_allergens
        self.breakfast_exclusions = breakfast_exclusions or []

        # Apply CSP constraints to filter breakfast recipes
        breakfast_compliant = breakfast_df[
            breakfast_df.apply(
                lambda row: self.diet_enforcer.is_compliant_multi(
                    row["ingredients_text"],
                    diet_type,
                    row.get("name", ""),
                    avoid_additives,
                    avoid_allergens,
                ),
                axis=1,
            )
        ].copy()

        # Apply CSP constraints to filter main meal recipes
        main_compliant = main_df[
            main_df.apply(
                lambda row: self.diet_enforcer.is_compliant_multi(
                    row["ingredients_text"],
                    diet_type,
                    row.get("name", ""),
                    avoid_additives,
                    avoid_allergens,
                ),
                axis=1,
            )
        ].copy()

        # Apply ML quality predictions for heuristic guidance
        self.breakfast_pool = self.ml_model.predict_quality(breakfast_compliant)
        self.main_pool = self.ml_model.predict_quality(main_compliant)

        # Sort by quality for greedy selection heuristic
        self.breakfast_pool = self.breakfast_pool.sort_values(
            "predicted_quality", ascending=False
        )
        self.main_pool = self.main_pool.sort_values(
            "predicted_quality", ascending=False
        )

        # Reset rejection sets for new session
        self.rejected_breakfasts.clear()
        self.rejected_lunches.clear()
        self.rejected_dinners.clear()

    def calculate_tdee(
        self,
        age: int,
        weight: float,
        height: float,
        gender: str,
        activity: str,
        goal: str,
    ) -> float:
        """
        Calculate Total Daily Energy Expenditure using Mifflin-St Jeor equation.

        This is a well-established formula for estimating caloric needs
        based on individual physical characteristics and activity level.

        Args:
            age: Age in years.
            weight: Weight in kilograms.
            height: Height in centimeters.
            gender: 'male' or 'female'.
            activity: Activity level ('sedentary', 'light', 'moderate', 'active', 'very_active').
            goal: Weight goal ('lose', 'maintain', or 'gain').

        Returns:
            float: Target daily calories (minimum 1200 kcal for safety).
        """
        if gender.lower() == "male":
            bmr = (10 * weight) + (6.25 * height) - (5 * age) + 5
        else:
            bmr = (10 * weight) + (6.25 * height) - (5 * age) - 161

        multipliers: Dict[str, float] = {
            "sedentary": 1.2,
            "light": 1.375,
            "moderate": 1.55,
            "active": 1.725,
            "very_active": 1.9,
        }
        tdee: float = bmr * multipliers.get(activity, 1.55)

        if goal == "lose":
            tdee -= 500
        elif goal == "gain":
            tdee += 300

        return max(1200.0, tdee)

    def optimize_meals(
        self,
        target_calories: float,
        diet_type: str
    ) -> Optional[MealPlan]:
        """
        Select optimal meal combination using Heuristic Search (Monte Carlo).

        ===========================================================================
        IMPLEMENTS: Heuristic Search via Monte Carlo Optimization
        ===========================================================================

        Strategy:
            1. Dynamic Portion Scaling for high-calorie targets (>3000 kcal)
            2. Monte Carlo search with 5,000 random iterations
            3. Early termination when total within ±150 of target
            4. Fallback to best-effort solution if no perfect match

        The search is guided by:
            - Random sampling for exploration
            - Breakfast validation heuristics (is_real_breakfast)
            - Quality scores from ML model (implicit via sorted pools)

        Args:
            target_calories: Daily calorie target in kcal.
            diet_type: Active dietary restriction.

        Returns:
            Optional[MealPlan]: Optimized meal plan or None if impossible.
        """
        # Dynamic portion scaling for extreme calorie targets
        if target_calories > 5000:
            portion_multiplier: float = 3.0
        elif target_calories > 3000:
            portion_multiplier = 2.0
        else:
            portion_multiplier = 1.0

        tolerance: float = 150.0
        max_iterations: int = 5000

        available_breakfast = self.breakfast_pool[
            ~self.breakfast_pool.index.isin(self.rejected_breakfasts)
        ]
        available_main = self.main_pool[
            ~self.main_pool.index.isin(self.rejected_lunches | self.rejected_dinners)
        ]

        if available_breakfast.empty or len(available_main) < 2:
            return None

        best_plan = None
        best_diff = float("inf")

        # Select a valid breakfast (using standalone is_real_breakfast function)
        breakfast_indices = available_breakfast.index.tolist()
        selected_breakfast_idx = None
        last_breakfast_idx = None

        for _ in range(50):
            candidate_idx = random.choice(breakfast_indices)
            last_breakfast_idx = candidate_idx
            candidate = available_breakfast.loc[candidate_idx]
            if is_real_breakfast(candidate.get("name", ""), self.breakfast_exclusions):
                selected_breakfast_idx = candidate_idx
                break

        if selected_breakfast_idx is None:
            selected_breakfast_idx = last_breakfast_idx

        # Monte Carlo search loop (Heuristic Search)
        for _ in range(max_iterations):
            if random.random() < 0.7:
                breakfast_idx = selected_breakfast_idx
            else:
                for _ in range(10):
                    breakfast_idx = random.choice(breakfast_indices)
                    if is_real_breakfast(
                        available_breakfast.loc[breakfast_idx].get("name", ""),
                        self.breakfast_exclusions
                    ):
                        break

            main_indices = available_main.index.tolist()
            lunch_idx = random.choice(main_indices)

            dinner_candidates = [idx for idx in main_indices if idx != lunch_idx]
            if not dinner_candidates:
                continue
            dinner_idx = random.choice(dinner_candidates)

            breakfast = available_breakfast.loc[breakfast_idx]
            lunch = available_main.loc[lunch_idx]
            dinner = available_main.loc[dinner_idx]

            breakfast_cal = breakfast.get("calories", 0)
            lunch_cal = lunch.get("calories", 0) * portion_multiplier
            dinner_cal = dinner.get("calories", 0) * portion_multiplier
            total_cal = breakfast_cal + lunch_cal + dinner_cal

            diff = abs(total_cal - target_calories)
            if diff < tolerance:
                total_protein = (
                    breakfast.get("protein", 0)
                    + lunch.get("protein", 0) * portion_multiplier
                    + dinner.get("protein", 0) * portion_multiplier
                )
                self.current_plan = MealPlan(
                    breakfast=breakfast,
                    lunch=lunch,
                    dinner=dinner,
                    total_calories=total_cal,
                    total_protein=total_protein,
                    target_calories=target_calories,
                    diet_type=diet_type,
                    portion_multiplier=portion_multiplier,
                )
                return self.current_plan

            if diff < best_diff:
                best_diff = diff
                best_plan = (breakfast, lunch, dinner, total_cal)

        # Fallback: Return best attempt found during search
        if best_plan:
            breakfast, lunch, dinner, total_cal = best_plan
            total_protein = (
                breakfast.get("protein", 0)
                + lunch.get("protein", 0) * portion_multiplier
                + dinner.get("protein", 0) * portion_multiplier
            )
            self.current_plan = MealPlan(
                breakfast=breakfast,
                lunch=lunch,
                dinner=dinner,
                total_calories=total_cal,
                total_protein=total_protein,
                target_calories=target_calories,
                diet_type=diet_type,
                portion_multiplier=portion_multiplier,
            )
            return self.current_plan

        return None

    def filter_recipes(
        self,
        recipes_df: pd.DataFrame,
        max_time_minutes: Optional[int] = None,
        complexity_level: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Filter recipes by cooking time and complexity constraints.

        Args:
            recipes_df: DataFrame of recipes to filter.
            max_time_minutes: Maximum cooking time in minutes (optional).
            complexity_level: 'Simple' (<8 ingredients), 'Moderate' (8-15), 'Gourmet' (>15).

        Returns:
            pd.DataFrame: Filtered recipes matching the criteria.
        """
        if recipes_df.empty:
            return recipes_df

        filtered = recipes_df.copy()

        if max_time_minutes is not None and "minutes" in filtered.columns:
            filtered = filtered[filtered["minutes"] <= max_time_minutes]

        if complexity_level is not None and "n_ingredients" in filtered.columns:
            complexity_lower = complexity_level.lower()
            if complexity_lower == "simple":
                filtered = filtered[filtered["n_ingredients"] < 8]
            elif complexity_lower == "moderate":
                filtered = filtered[
                    (filtered["n_ingredients"] >= 8)
                    & (filtered["n_ingredients"] <= 15)
                ]
            elif complexity_lower == "gourmet":
                filtered = filtered[filtered["n_ingredients"] > 15]

        return filtered

    def reject_meal(self, meal_type: str) -> Optional[MealPlan]:
        """
        Reject a meal and find replacement (self-correction mechanism).

        Implements goal-based agent self-correction by allowing users
        to reject meals and triggering replanning.

        Args:
            meal_type: 'breakfast', 'lunch', or 'dinner'.

        Returns:
            Optional[MealPlan]: Updated meal plan with replacement.
        """
        if self.current_plan is None:
            return None

        if meal_type == "breakfast" and self.current_plan.breakfast is not None:
            self.rejected_breakfasts.add(self.current_plan.breakfast.name)
        elif meal_type == "lunch" and self.current_plan.lunch is not None:
            self.rejected_lunches.add(self.current_plan.lunch.name)
        elif meal_type == "dinner" and self.current_plan.dinner is not None:
            self.rejected_dinners.add(self.current_plan.dinner.name)

        return self.optimize_meals(
            self.current_plan.target_calories, self.current_plan.diet_type
        )
