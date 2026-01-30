# NutriGen AI

A meal planning system that generates personalized daily and weekly meal plans based on user-defined dietary restrictions and caloric targets. The system combines a Random Forest classifier for recipe quality scoring with rule-based constraint enforcement and randomized search optimization.

## How It Works

The system follows a three-stage pipeline:

1. **Constraint Filtering** – The `DietEnforcer` module applies regex-based pattern matching against banned ingredient lists to filter recipes that violate dietary restrictions (vegan, vegetarian, keto, gluten-free, dairy-free). Recipes containing any banned ingredient are rejected immediately.

2. **Quality Scoring** – The `NutriBrain` module uses a Random Forest classifier trained on OpenFoodFacts nutritional data to assign quality scores (Low/Medium/High) to compliant recipes based on 24 engineered features including macro ratios, caloric density, and nutrient balance.

3. **Meal Optimization** – The `RationalAgent` uses Monte Carlo sampling (5,000 iterations) to find breakfast + lunch + dinner combinations that minimize the distance from the user's caloric target (within ±150 kcal). The search is guided by the ML-predicted quality scores.

Caloric targets are calculated using the Mifflin-St Jeor equation based on user profile data (age, weight, height, activity level, goal).

## Features

- Daily and weekly meal plan generation
- Dietary restriction enforcement: vegan, vegetarian, gluten-free, keto, dairy-free
- TDEE calculation with goal-based calorie adjustment (lose, maintain, gain weight)
- Ingredient aggregation for shopping lists
- Pantry-based recipe search
- Downloadable meal plans
- Minimum 1,200 kcal safety floor

## Tech Stack

- **Python 3.9+**
- **Streamlit** – Web interface
- **scikit-learn** – Random Forest classifier
- **pandas / NumPy** – Data processing
- **OpenFoodFacts** – Training data for ML model
- **Food.com recipes** – Recipe database for meal selection

## Project Structure

```
nutrigen/
├── app.py              # Streamlit web application
├── src/
│   ├── ai_engine.py    # NutriBrain – ML classifier for recipe quality
│   ├── planner.py      # RationalAgent – Monte Carlo meal optimizer
│   ├── constraints.py  # DietEnforcer – Dietary constraint filtering
│   ├── data_loader.py  # HybridDataLoader – Dual-source data loading
│   └── config.py       # Path and styling configuration
├── requirements.txt
└── Procfile            # Heroku deployment config
```

## Setup

```bash
# Clone and enter directory
git clone <repository-url>
cd nutrigen

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
```

The app runs at `http://localhost:8501`.

**Note:** The application expects data files (`openfoodfacts.tsv`, `RAW_recipes.csv`, `banned_ingredients.json`) in a `data/` directory. These are not included in the repository due to size constraints.

## Limitations

- Recipe quality predictions depend on training data distribution; edge cases may score unexpectedly
- Constraint matching uses substring search and may produce false positives on ingredient names
- Monte Carlo search is stochastic; repeated runs may yield different meal combinations
- No persistent user profiles or meal history

## Future Extensions

- User authentication and saved preferences
- Nutritional tracking over time
- Integration with grocery delivery APIs
- Expanded dietary restriction support (halal, low-sodium, etc.)
- Recipe rating feedback to improve recommendations
