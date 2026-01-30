"""
NutriGen AI - Streamlit Web Dashboard.

This module provides an interactive web interface for NutriGen AI,
allowing users to generate personalized meal plans through a
beautiful, clickable UI instead of the command-line interface.

Run with:
    streamlit run app.py

Features:
    1. Daily Menu - Generate a single day meal plan with shopping list
    2. Weekly Plan - Generate 7 days of meals with master shopping list
    3. Pantry Search - Find recipes using ingredients you have
    - Interactive user profile input
    - Visual macro charts
    - Downloadable meal plans
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
from typing import List, Dict, Optional
from collections import Counter

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import HybridDataLoader
from src.ai_engine import NutriBrain
from src.constraints import DietEnforcer
from src.planner import RationalAgent, WeeklyPlan


# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="NutriGen AI",
    page_icon="🥗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# CUSTOM CSS
# =============================================================================

st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #2E7D32;
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .day-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 0.5rem 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
        font-weight: bold;
    }
    .recipe-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #667eea;
    }
    .shopping-item {
        background: #e8f5e9;
        padding: 0.3rem 0.8rem;
        border-radius: 5px;
        margin: 0.2rem;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# SESSION STATE INITIALIZATION
# =============================================================================

@st.cache_resource
def load_system():
    """Load and cache the NutriGen AI system components."""
    with st.spinner("🧠 Initializing AI components..."):
        data_loader = HybridDataLoader()
        ml_model = NutriBrain()
        diet_enforcer = DietEnforcer()

        # Load training data
        training_data = data_loader.load_training_data(sample_size=10000)
        if not training_data.empty:
            ml_model.train(training_data)

        # Load recipes
        data_loader.load_recipes(max_recipes=5000)

        # Create agent
        agent = RationalAgent(diet_enforcer, ml_model)

        return data_loader, ml_model, diet_enforcer, agent


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def extract_ingredients(recipe) -> List[str]:
    """Extract ingredients from a recipe."""
    if recipe is None:
        return []

    ingredients_raw = recipe.get("ingredients", "")
    if pd.isna(ingredients_raw) or not ingredients_raw:
        return []

    # Handle string representation of list
    if isinstance(ingredients_raw, str):
        if ingredients_raw.startswith("["):
            try:
                import ast
                return ast.literal_eval(ingredients_raw)
            except:
                return [ingredients_raw]
        return [ing.strip() for ing in ingredients_raw.split(",")]

    return list(ingredients_raw) if ingredients_raw else []


def aggregate_shopping_list(all_ingredients: List[str]) -> Dict[str, int]:
    """Aggregate ingredients into a shopping list with counts."""
    cleaned = []
    for ing in all_ingredients:
        if isinstance(ing, str):
            cleaned.append(ing.lower().strip())
    return dict(Counter(cleaned))


def display_meal_card(meal, meal_type: str, emoji: str):
    """Display a single meal card."""
    st.markdown(f"#### {emoji} {meal_type}")
    if meal is not None:
        st.markdown(f"**{meal.get('name', 'Unknown')}**")
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"🔥 {meal.get('calories', 0):.0f} kcal")
            st.write(f"🥩 {meal.get('protein', 0):.1f}g protein")
        with col2:
            st.write(f"🧈 {meal.get('fat', 0):.1f}g fat")
            st.write(f"🍞 {meal.get('carbs', 0):.1f}g carbs")
        if 'minutes' in meal and pd.notna(meal.get('minutes')):
            st.write(f"⏱️ {int(meal['minutes'])} min")
    else:
        st.write("No suitable meal found")


def generate_daily_plan_text(plan, diet_type: str) -> str:
    """Generate downloadable text for daily plan."""
    diff = plan.total_calories - plan.target_calories

    # Helper to safely get meal info
    def get_meal_name(meal):
        if meal is None:
            return 'N/A'
        if isinstance(meal, dict):
            return meal.get('name', 'N/A')
        return 'N/A'

    def get_meal_calories(meal):
        if meal is None:
            return 0
        if isinstance(meal, dict):
            return meal.get('calories', 0)
        return 0

    breakfast_name = get_meal_name(plan.breakfast)
    breakfast_cal = get_meal_calories(plan.breakfast)
    lunch_name = get_meal_name(plan.lunch)
    lunch_cal = get_meal_calories(plan.lunch)
    dinner_name = get_meal_name(plan.dinner)
    dinner_cal = get_meal_calories(plan.dinner)

    return f"""
NUTRIGEN AI - DAILY MEAL PLAN
==========================================

Diet: {diet_type.upper()}
Target: {plan.target_calories:.0f} kcal

BREAKFAST
---------
{breakfast_name}
Calories: {breakfast_cal:.0f} kcal

LUNCH
-----
{lunch_name}
Calories: {lunch_cal:.0f} kcal

DINNER
------
{dinner_name}
Calories: {dinner_cal:.0f} kcal

TOTALS
------
Total Calories: {plan.total_calories:.0f} kcal
Total Protein: {plan.total_protein:.1f}g
Target Difference: {diff:+.0f} kcal

Generated by NutriGen AI
"""


def get_meal_name_safe(meal):
    """Safely get meal name handling None, dict, and pandas Series."""
    if meal is None:
        return 'N/A'
    if isinstance(meal, dict):
        return meal.get('name', 'N/A')
    return 'N/A'


def generate_weekly_plan_text(weekly_plan: WeeklyPlan, diet_type: str, target_calories: float, all_ingredients: List) -> str:
    """Generate downloadable text for weekly plan."""
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    text = f"""
NUTRIGEN AI - WEEKLY MEAL PLAN
==========================================

Diet: {diet_type.upper()}
Target per day: {target_calories:.0f} kcal

"""
    for i, (day_name, plan) in enumerate(zip(days, weekly_plan.daily_plans)):
        if plan:
            breakfast_name = get_meal_name_safe(plan.breakfast)
            lunch_name = get_meal_name_safe(plan.lunch)
            dinner_name = get_meal_name_safe(plan.dinner)
            text += f"""
{day_name.upper()}
{'-' * len(day_name)}
Breakfast: {breakfast_name}
Lunch: {lunch_name}
Dinner: {dinner_name}
Total: {plan.total_calories:.0f} kcal
"""

    # Add shopping list
    text += """

MASTER SHOPPING LIST
====================
"""
    shopping = aggregate_shopping_list(all_ingredients)
    for item, count in sorted(shopping.items()):
        text += f"- {item} (x{count})\n"

    text += f"""

Generated by NutriGen AI
"""
    return text


# =============================================================================
# HEADER
# =============================================================================

st.markdown('<h1 class="main-header">🥗 NutriGen AI</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Intelligent Diet Planning System</p>', unsafe_allow_html=True)

# AI Techniques Banner
with st.expander("🤖 AI Techniques Used", expanded=False):
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### 🧠 Machine Learning")
        st.markdown("""
        - **Model:** RandomForest Classifier
        - **Training:** 10,000 OpenFoodFacts samples
        - **Task:** Recipe quality prediction
        """)
    with col2:
        st.markdown("### ⚖️ Constraint Satisfaction")
        st.markdown("""
        - **Method:** Regex pattern matching
        - **Constraints:** Dietary restrictions
        - **Enforcement:** Hard constraints
        """)
    with col3:
        st.markdown("### 🎲 Heuristic Search")
        st.markdown("""
        - **Algorithm:** Monte Carlo
        - **Iterations:** 5,000
        - **Goal:** Optimal meal combinations
        """)


# =============================================================================
# SIDEBAR - USER PROFILE
# =============================================================================

st.sidebar.header("👤 Your Profile")

# Personal Info
st.sidebar.subheader("Personal Information")
age = st.sidebar.number_input("Age", min_value=10, max_value=100, value=30, step=1)
weight = st.sidebar.number_input("Weight (kg)", min_value=30, max_value=200, value=70, step=1)
height = st.sidebar.number_input("Height (cm)", min_value=100, max_value=250, value=170, step=1)
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])

# Activity Level
st.sidebar.subheader("Activity Level")
activity_options = {
    "Sedentary (office job)": "sedentary",
    "Light (1-2 days/week)": "light",
    "Moderate (3-5 days/week)": "moderate",
    "Active (6-7 days/week)": "active",
    "Very Active (athlete)": "very_active",
}
activity_display = st.sidebar.selectbox("Activity Level", list(activity_options.keys()))
activity = activity_options[activity_display]

# Goal
st.sidebar.subheader("Your Goal")
goal_options = {
    "🔥 Lose Weight": "lose",
    "⚖️ Maintain Weight": "maintain",
    "💪 Gain Muscle": "gain",
}
goal_display = st.sidebar.selectbox("Goal", list(goal_options.keys()))
goal = goal_options[goal_display]

# Dietary Restrictions
st.sidebar.subheader("Dietary Restrictions")
diet_options = {
    "None (eat everything)": "none",
    "🥬 Vegetarian": "vegetarian",
    "🌱 Vegan": "vegan",
    "🌾 Gluten-Free": "gluten-free",
    "🥑 Keto": "keto",
    "🧀 Dairy-Free": "dairy-free",
}
diet_display = st.sidebar.selectbox("Diet Type", list(diet_options.keys()))
diet_type = diet_options[diet_display]

# Additional Options
st.sidebar.subheader("Additional Options")
avoid_additives = st.sidebar.checkbox("🚫 Clean Label (No Additives)")
avoid_allergens = st.sidebar.checkbox("⚠️ Avoid Common Allergens")

# Filtering Options
st.sidebar.subheader("Recipe Filters")
max_time = st.sidebar.number_input("Max Cooking Time (min)", min_value=10, max_value=120, value=60, step=5)
complexity = st.sidebar.selectbox(
    "Complexity",
    ["Any", "Simple (< 8 ingredients)", "Moderate (8-15)", "Gourmet (> 15)"]
)
complexity_map = {
    "Any": None,
    "Simple (< 8 ingredients)": "Simple",
    "Moderate (8-15)": "Moderate",
    "Gourmet (> 15)": "Gourmet",
}


# =============================================================================
# MAIN CONTENT - TAB NAVIGATION
# =============================================================================

# Load system
try:
    data_loader, ml_model, diet_enforcer, agent = load_system()
    system_loaded = True
except Exception as e:
    st.error(f"Failed to load system: {e}")
    st.info("Make sure data files are in the data/ folder.")
    system_loaded = False

if system_loaded:
    # Calculate TDEE
    target_calories = agent.calculate_tdee(
        age=age,
        weight=weight,
        height=height,
        gender=gender.lower(),
        activity=activity,
        goal=goal,
    )

    # Display TDEE
    st.markdown("### 📊 Your Daily Target")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("🔥 Calories", f"{target_calories:.0f} kcal")
    with col2:
        st.metric("🥩 Protein", f"{target_calories * 0.25 / 4:.0f}g")
    with col3:
        st.metric("🧈 Fat", f"{target_calories * 0.30 / 9:.0f}g")
    with col4:
        st.metric("🍞 Carbs", f"{target_calories * 0.45 / 4:.0f}g")

    st.divider()

    # Create tabs for the three main features
    tab1, tab2, tab3 = st.tabs([
        "🍽️ Daily Menu",
        "📅 Weekly Plan",
        "🥘 Pantry Search"
    ])

    # =========================================================================
    # TAB 1: DAILY MENU
    # =========================================================================
    with tab1:
        st.markdown("### Generate Daily Menu")
        st.markdown("Get a personalized breakfast, lunch, and dinner for one day.")

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            generate_daily = st.button(
                "🍽️ Generate Daily Menu",
                type="primary",
                use_container_width=True,
                key="daily_btn"
            )

        if generate_daily:
            with st.spinner("🔄 AI is optimizing your meals..."):
                # Get recipe pools
                breakfast_pool = data_loader.breakfast_recipes.copy()
                main_pool = data_loader.main_recipes.copy()

                # Apply filters
                if max_time < 120:
                    breakfast_pool = agent.filter_recipes(breakfast_pool, max_time_minutes=max_time)
                    main_pool = agent.filter_recipes(main_pool, max_time_minutes=max_time)

                if complexity_map[complexity]:
                    breakfast_pool = agent.filter_recipes(breakfast_pool, complexity_level=complexity_map[complexity])
                    main_pool = agent.filter_recipes(main_pool, complexity_level=complexity_map[complexity])

                # Initialize pools
                agent.initialize_pools(
                    breakfast_pool, main_pool, diet_type,
                    avoid_additives, avoid_allergens,
                    data_loader.breakfast_exclusions,
                )

                # Generate plan
                plan = agent.optimize_meals(target_calories, diet_type)

            if plan:
                st.success("✅ Meal plan generated successfully!")

                # Display meals
                col1, col2, col3 = st.columns(3)
                with col1:
                    display_meal_card(plan.breakfast, "Breakfast", "🌅")
                with col2:
                    display_meal_card(plan.lunch, "Lunch", "🌞")
                with col3:
                    display_meal_card(plan.dinner, "Dinner", "🌙")

                st.divider()

                # Summary
                st.markdown("### 📈 Daily Summary")
                col1, col2, col3 = st.columns(3)
                with col1:
                    diff = plan.total_calories - plan.target_calories
                    st.metric("Total Calories", f"{plan.total_calories:.0f} kcal", f"{diff:+.0f}")
                with col2:
                    st.metric("Total Protein", f"{plan.total_protein:.1f}g")
                with col3:
                    accuracy = 100 - (abs(diff) / plan.target_calories * 100)
                    st.metric("Target Accuracy", f"{accuracy:.1f}%")

                # Shopping List
                st.markdown("### 🛒 Shopping List")
                all_ingredients = []
                for meal in [plan.breakfast, plan.lunch, plan.dinner]:
                    all_ingredients.extend(extract_ingredients(meal))

                if all_ingredients:
                    shopping = aggregate_shopping_list(all_ingredients)
                    cols = st.columns(4)
                    for i, (item, count) in enumerate(sorted(shopping.items())):
                        with cols[i % 4]:
                            st.markdown(f"• {item} ×{count}")
                else:
                    st.info("No ingredient data available for these recipes.")

                # Download
                st.divider()
                st.download_button(
                    label="📥 Download Daily Plan",
                    data=generate_daily_plan_text(plan, diet_type),
                    file_name="nutrigen_daily_plan.txt",
                    mime="text/plain",
                )
            else:
                st.error("❌ Could not generate a meal plan. Try relaxing your restrictions.")

    # =========================================================================
    # TAB 2: WEEKLY PLAN
    # =========================================================================
    with tab2:
        st.markdown("### Generate Weekly Plan")
        st.markdown("Get 7 days of meals with a master shopping list.")

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            generate_weekly = st.button(
                "📅 Generate 7-Day Plan",
                type="primary",
                use_container_width=True,
                key="weekly_btn"
            )

        if generate_weekly:
            days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            daily_plans = []
            all_ingredients = []

            progress_bar = st.progress(0, text="Generating weekly plan...")

            for i, day in enumerate(days):
                progress_bar.progress((i + 1) / 7, text=f"Planning {day}...")

                # Get fresh pools for each day
                breakfast_pool = data_loader.breakfast_recipes.copy()
                main_pool = data_loader.main_recipes.copy()

                # Apply filters
                if max_time < 120:
                    breakfast_pool = agent.filter_recipes(breakfast_pool, max_time_minutes=max_time)
                    main_pool = agent.filter_recipes(main_pool, max_time_minutes=max_time)

                if complexity_map[complexity]:
                    breakfast_pool = agent.filter_recipes(breakfast_pool, complexity_level=complexity_map[complexity])
                    main_pool = agent.filter_recipes(main_pool, complexity_level=complexity_map[complexity])

                # Initialize pools
                agent.initialize_pools(
                    breakfast_pool, main_pool, diet_type,
                    avoid_additives, avoid_allergens,
                    data_loader.breakfast_exclusions,
                )

                # Generate plan
                plan = agent.optimize_meals(target_calories, diet_type)
                daily_plans.append(plan)

                # Collect ingredients
                if plan:
                    for meal in [plan.breakfast, plan.lunch, plan.dinner]:
                        all_ingredients.extend(extract_ingredients(meal))

            progress_bar.empty()

            # Create WeeklyPlan object
            weekly_plan = WeeklyPlan(
                daily_plans=daily_plans,
                total_weekly_calories=sum(p.total_calories for p in daily_plans if p),
                total_weekly_protein=sum(p.total_protein for p in daily_plans if p),
            )

            st.success("✅ Weekly plan generated!")

            # Display each day
            for day, plan in zip(days, daily_plans):
                with st.expander(f"📆 {day}", expanded=(day == "Monday")):
                    if plan:
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            display_meal_card(plan.breakfast, "Breakfast", "🌅")
                        with col2:
                            display_meal_card(plan.lunch, "Lunch", "🌞")
                        with col3:
                            display_meal_card(plan.dinner, "Dinner", "🌙")

                        st.markdown(f"**Daily Total:** {plan.total_calories:.0f} kcal | {plan.total_protein:.1f}g protein")
                    else:
                        st.warning("Could not generate plan for this day.")

            # Weekly summary
            st.divider()
            st.markdown("### 📊 Weekly Summary")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Weekly Calories", f"{weekly_plan.total_weekly_calories:.0f} kcal")
            with col2:
                st.metric("Daily Average", f"{weekly_plan.total_weekly_calories/7:.0f} kcal")
            with col3:
                st.metric("Total Protein", f"{weekly_plan.total_weekly_protein:.1f}g")

            # Master shopping list
            st.markdown("### 🛒 Master Shopping List")
            if all_ingredients:
                shopping = aggregate_shopping_list(all_ingredients)

                # Display in multiple columns
                cols = st.columns(4)
                sorted_items = sorted(shopping.items())
                for i, (item, count) in enumerate(sorted_items):
                    with cols[i % 4]:
                        st.markdown(f"• {item} ×{count}")

                st.info(f"📝 Total unique items: {len(shopping)}")
            else:
                st.info("No ingredient data available.")

            # Download
            st.divider()
            st.download_button(
                label="📥 Download Weekly Plan",
                data=generate_weekly_plan_text(weekly_plan, diet_type, target_calories, all_ingredients),
                file_name="nutrigen_weekly_plan.txt",
                mime="text/plain",
            )

    # =========================================================================
    # TAB 3: PANTRY SEARCH
    # =========================================================================
    with tab3:
        st.markdown("### Pantry Search")
        st.markdown("Find recipes using ingredients you already have!")

        # Ingredient input
        st.markdown("#### Enter your ingredients")
        ingredients_input = st.text_area(
            "Type ingredients separated by commas",
            placeholder="e.g., chicken, rice, garlic, onion, tomato",
            height=100,
        )

        # Common ingredients quick-add
        st.markdown("#### Quick Add Common Ingredients")
        common_ingredients = [
            "chicken", "beef", "pork", "fish", "eggs",
            "rice", "pasta", "bread", "potato", "flour",
            "onion", "garlic", "tomato", "carrot", "broccoli",
            "milk", "cheese", "butter", "olive oil", "salt"
        ]

        # Create checkbox grid
        cols = st.columns(5)
        selected_common = []
        for i, ing in enumerate(common_ingredients):
            with cols[i % 5]:
                if st.checkbox(ing, key=f"ing_{ing}"):
                    selected_common.append(ing)

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            search_btn = st.button(
                "🔍 Find Recipes",
                type="primary",
                use_container_width=True,
                key="pantry_btn"
            )

        if search_btn:
            # Combine typed and selected ingredients
            user_ingredients = []

            if ingredients_input.strip():
                user_ingredients.extend([
                    ing.strip().lower()
                    for ing in ingredients_input.split(",")
                    if ing.strip()
                ])

            user_ingredients.extend([ing.lower() for ing in selected_common])
            user_ingredients = list(set(user_ingredients))  # Remove duplicates

            if not user_ingredients:
                st.warning("Please enter or select at least one ingredient.")
            else:
                st.info(f"🔍 Searching for recipes with: {', '.join(user_ingredients)}")

                # Search in both breakfast and main recipes
                all_recipes = pd.concat([
                    data_loader.breakfast_recipes,
                    data_loader.main_recipes
                ], ignore_index=True)

                # Score recipes by ingredient match
                matches = []
                for idx, row in all_recipes.iterrows():
                    recipe_ingredients = extract_ingredients(row)
                    recipe_ing_lower = [ing.lower() for ing in recipe_ingredients]

                    # Count matches
                    match_count = sum(
                        1 for user_ing in user_ingredients
                        if any(user_ing in ring for ring in recipe_ing_lower)
                    )

                    if match_count > 0:
                        matches.append({
                            'recipe': row,
                            'match_count': match_count,
                            'match_percent': match_count / len(user_ingredients) * 100
                        })

                # Sort by match count
                matches.sort(key=lambda x: x['match_count'], reverse=True)

                if matches:
                    st.success(f"✅ Found {len(matches)} matching recipes!")

                    # Display top 10
                    st.markdown("### 🥘 Best Matches")
                    for i, match in enumerate(matches[:10]):
                        recipe = match['recipe']
                        with st.expander(
                            f"#{i+1} {recipe.get('name', 'Unknown')} "
                            f"({match['match_count']}/{len(user_ingredients)} ingredients matched)"
                        ):
                            col1, col2 = st.columns(2)
                            with col1:
                                st.markdown("**Nutrition:**")
                                st.write(f"🔥 {recipe.get('calories', 0):.0f} kcal")
                                st.write(f"🥩 {recipe.get('protein', 0):.1f}g protein")
                                st.write(f"🧈 {recipe.get('fat', 0):.1f}g fat")
                                st.write(f"🍞 {recipe.get('carbs', 0):.1f}g carbs")
                            with col2:
                                st.markdown("**Details:**")
                                if 'minutes' in recipe and pd.notna(recipe.get('minutes')):
                                    st.write(f"⏱️ {int(recipe['minutes'])} minutes")
                                if 'n_ingredients' in recipe and pd.notna(recipe.get('n_ingredients')):
                                    st.write(f"📝 {int(recipe['n_ingredients'])} ingredients")
                                st.write(f"✅ Match: {match['match_percent']:.0f}%")

                            # Show ingredients
                            st.markdown("**Ingredients:**")
                            recipe_ings = extract_ingredients(recipe)
                            if recipe_ings:
                                for ing in recipe_ings[:15]:  # Limit to 15
                                    st.write(f"• {ing}")
                                if len(recipe_ings) > 15:
                                    st.write(f"... and {len(recipe_ings) - 15} more")
                else:
                    st.warning("❌ No matching recipes found. Try different ingredients.")


# =============================================================================
# FOOTER
# =============================================================================

st.divider()
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <p>🥗 <strong>NutriGen AI</strong></p>
    <p>Built with ❤️ by Klea Hila</p>
</div>
""", unsafe_allow_html=True)
