"""
Data loading module for NutriGen AI.

This module implements the Hybrid Intelligence data layer by combining:
    - OpenFoodFacts TSV for ML training (nutritional ground truth)
    - Food.com recipes CSV for user-facing meal suggestions

Uses chunking for performance optimization when ingesting large CSV files.

Key Components:
    - HybridDataLoader: Dual-source data loader for training and menu data
    - _get_data_path: Dynamic path resolution for data files
"""

# =============================================================================
# IMPORTS
# =============================================================================

# Standard library
import ast
import json
import os
from typing import Dict, List, Optional, Tuple

# Third-party
import pandas as pd

# Local
from src.config import Colors, DATA_DIR
from src.constraints import DietEnforcer


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


def _get_data_path(filename: str) -> str:
    """
    Resolve data file path dynamically.

    Searches multiple locations to find data files, providing
    flexibility in project structure.

    Args:
        filename: Name of the data file (e.g., 'openfoodfacts.tsv').

    Returns:
        str: Resolved absolute path to the data file.
    """
    primary_path: str = os.path.join(DATA_DIR, filename)
    if os.path.exists(primary_path):
        return primary_path

    script_dir: str = os.path.dirname(os.path.abspath(__file__))
    search_paths: List[str] = [
        os.path.join(script_dir, "..", "data", filename),
        os.path.join(script_dir, "data", filename),
        os.path.join(os.getcwd(), "data", filename),
        os.path.join(os.getcwd(), filename),
        filename,
    ]

    for path in search_paths:
        if os.path.exists(path):
            return os.path.abspath(path)

    return primary_path


# =============================================================================
# DATA LOADER CLASS
# =============================================================================


class HybridDataLoader:
    """
    Dual-source data loader following proposal architecture.

    Source A: OpenFoodFacts → TRAINING DATA for ML model
    Source B: Food.com Recipes → ACTUAL MENU items for user selection

    This separation ensures:
        1. ML model learns from large nutritional database (OpenFoodFacts)
        2. Users receive REALISTIC MEALS from curated recipe database

    Attributes:
        off_path: Path to OpenFoodFacts TSV file.
        recipes_path: Path to Food.com recipes CSV file.
        training_data: Cached training data DataFrame.
        breakfast_recipes: Filtered breakfast recipes DataFrame.
        main_recipes: Filtered main meal recipes DataFrame.
        breakfast_exclusions: Keywords to exclude from breakfast.
        diet_enforcer: DietEnforcer instance for filtering.
    """

    BREAKFAST_KEYWORDS: List[str] = [
        "egg", "omelet", "omelette", "pancake", "waffle", "bacon",
        "scramble", "scrambled", "oatmeal", "toast", "avocado toast",
        "french toast", "breakfast", "morning", "brunch", "hash brown",
        "frittata", "quiche", "muffin", "granola", "smoothie bowl",
        "eggs benedict", "huevos", "crepe", "porridge", "bagel",
        "cereal", "yogurt", "fruit bowl", "breakfast burrito",
        "breakfast sandwich", "breakfast bowl", "breakfast casserole",
    ]

    MAIN_MEAL_KEYWORDS: List[str] = [
        "chicken", "beef", "pasta", "salad", "soup", "steak", "salmon",
        "shrimp", "rice", "curry", "stir fry", "roast", "grilled",
        "baked", "casserole", "tacos", "burrito", "bowl", "noodle",
        "pork", "lamb", "turkey", "fish", "tofu", "veggie", "vegetable",
        "wrap", "sandwich", "burger", "pizza", "lasagna", "risotto",
        "stew", "chili", "teriyaki", "bbq", "mediterranean", "asian",
    ]

    DESSERT_KEYWORDS: List[str] = [
        "cookie", "cake", "pie", "brownie", "ice cream", "dessert",
        "candy", "chocolate bar", "fudge", "cupcake", "tart", "pudding",
        "cheesecake", "mousse", "donut", "doughnut", "frosting", "icing",
    ]

    def __init__(
        self,
        off_path: Optional[str] = None,
        recipes_path: Optional[str] = None
    ) -> None:
        """
        Initialize the hybrid data loader.

        Args:
            off_path: Path to OpenFoodFacts TSV file (auto-resolved if None).
            recipes_path: Path to Food.com recipes CSV (auto-resolved if None).
        """
        self.off_path: str = off_path or _get_data_path("openfoodfacts.tsv")
        self.recipes_path: str = recipes_path or _get_data_path("RAW_recipes.csv")

        self.training_data: Optional[pd.DataFrame] = None
        self.breakfast_recipes: Optional[pd.DataFrame] = None
        self.main_recipes: Optional[pd.DataFrame] = None
        self.all_recipes: Optional[pd.DataFrame] = None

        self.breakfast_exclusions: List[str] = self._load_breakfast_exclusions()
        self.diet_enforcer: DietEnforcer = DietEnforcer()

    def _load_breakfast_exclusions(self) -> List[str]:
        """
        Load breakfast exclusion keywords from JSON file.

        Returns:
            List[str]: Words that should NOT appear in breakfast recipes.
        """
        json_path: str = os.path.join(DATA_DIR, "banned_ingredients.json")

        if os.path.exists(json_path):
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    data: Dict = json.load(f)
                    exclusions: List[str] = data.get("breakfast_exclusions", [])
                    if exclusions:
                        print(f"{Colors.INFO}[DATA] Loaded {len(exclusions)} breakfast exclusion rules{Colors.END}")
                        return exclusions
            except Exception as e:
                print(f"{Colors.WARNING}[DATA] Failed to load breakfast exclusions: {e}{Colors.END}")

        return [
            "soup", "stew", "chili", "curry", "pizza",
            "burger", "pasta", "lasagna", "dinner", "lunch",
        ]

    def load_data(self) -> bool:
        """
        Load all data sources (training data and recipes).

        Returns:
            bool: True if data loaded successfully, False otherwise.
        """
        training_data = self.load_training_data(sample_size=10000)
        if training_data.empty:
            print(f"{Colors.ERROR}[DATA] Failed to load training data{Colors.END}")
            return False

        breakfast, mains = self.load_recipes(max_recipes=5000)
        if breakfast.empty and mains.empty:
            print(f"{Colors.ERROR}[DATA] Failed to load recipes{Colors.END}")
            return False

        if not self.breakfast_exclusions:
            self.breakfast_exclusions = self._load_breakfast_exclusions()

        return True

    def load_training_data(self, sample_size: int = 10000) -> pd.DataFrame:
        """
        Load OpenFoodFacts data for ML TRAINING ONLY.

        Uses chunked reading for memory efficiency on large datasets.

        Args:
            sample_size: Number of samples to load.

        Returns:
            pd.DataFrame: Training data with nutritional features.
        """
        print(f"{Colors.INFO}[DATA] Loading OpenFoodFacts for ML training...{Colors.END}")

        # Check if file exists before attempting to load
        if not os.path.exists(self.off_path):
            print(f"{Colors.ERROR}[DATA] Dataset not found: {self.off_path}{Colors.END}")
            print(f"{Colors.WARNING}[DATA] Please download 'openfoodfacts.tsv' and place it in the 'data/' folder.{Colors.END}")
            return pd.DataFrame()

        try:
            chunks: List[pd.DataFrame] = []
            for chunk in pd.read_csv(
                self.off_path,
                sep="\t",
                low_memory=False,
                chunksize=20000,
                usecols=[
                    "product_name", "energy_100g", "proteins_100g",
                    "fat_100g", "carbohydrates_100g", "nutrition_grade_fr",
                ],
                on_bad_lines="skip",
            ):
                chunks.append(chunk)
                if sum(len(c) for c in chunks) >= sample_size:
                    break

            df: pd.DataFrame = pd.concat(chunks, ignore_index=True)

            df = df.dropna(subset=[
                "energy_100g", "proteins_100g", "fat_100g", "nutrition_grade_fr"
            ])

            df = df.rename(columns={
                "energy_100g": "calories",
                "proteins_100g": "protein",
                "fat_100g": "fat",
                "carbohydrates_100g": "carbs",
            })

            df = df[(df["calories"] > 0) & (df["calories"] < 1000)]
            df = df[(df["protein"] >= 0) & (df["protein"] < 100)]
            df = df[(df["fat"] >= 0) & (df["fat"] < 100)]

            grade_map: Dict[str, int] = {"a": 5, "b": 4, "c": 3, "d": 2, "e": 1}
            df["quality_score"] = df["nutrition_grade_fr"].str.lower().map(grade_map)
            df = df.dropna(subset=["quality_score"])

            self.training_data = df.head(sample_size)
            print(f"{Colors.SUCCESS}[DATA] Loaded {len(self.training_data):,} training samples{Colors.END}")

            return self.training_data

        except FileNotFoundError:
            print(f"{Colors.ERROR}[DATA] Dataset not found: {self.off_path}{Colors.END}")
            print(f"{Colors.WARNING}[DATA] Please download 'openfoodfacts.tsv' and place it in the 'data/' folder.{Colors.END}")
            return pd.DataFrame()
        except Exception as e:
            print(f"{Colors.ERROR}[ERROR] Failed to load OpenFoodFacts: {e}{Colors.END}")
            return pd.DataFrame()

    def load_recipes(self, max_recipes: int = 5000) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load and split Food.com recipes into breakfast and main meals.

        Parses recipe nutrition data and categorizes recipes based on
        name matching against keyword lists.

        Args:
            max_recipes: Maximum number of recipes to load.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: (breakfast_recipes, main_recipes).
        """
        print(f"{Colors.INFO}[DATA] Loading Food.com recipes for menu...{Colors.END}")

        # Check if file exists before attempting to load
        if not os.path.exists(self.recipes_path):
            print(f"{Colors.ERROR}[DATA] Dataset not found: {self.recipes_path}{Colors.END}")
            print(f"{Colors.WARNING}[DATA] Please download 'RAW_recipes.csv' and place it in the 'data/' folder.{Colors.END}")
            return pd.DataFrame(), pd.DataFrame()

        try:
            df: pd.DataFrame = pd.read_csv(self.recipes_path, nrows=max_recipes * 2)

            def parse_nutrition(nutrition_str: str) -> Optional[Dict[str, float]]:
                """Parse nutrition string into dictionary."""
                try:
                    if pd.isna(nutrition_str):
                        return None
                    values = ast.literal_eval(nutrition_str)
                    if len(values) >= 7:
                        return {
                            "calories": float(values[0]),
                            "fat": float(values[1]),
                            "protein": float(values[4]),
                            "carbs": float(values[6]),
                        }
                except Exception:
                    pass
                return None

            def parse_ingredients(ing_str: str) -> str:
                """Parse ingredients list into comma-separated string."""
                try:
                    if pd.isna(ing_str):
                        return ""
                    ingredients = ast.literal_eval(ing_str)
                    if isinstance(ingredients, list):
                        return ", ".join(ingredients)
                    return str(ingredients)
                except Exception:
                    return str(ing_str)

            df["parsed_nutrition"] = df["nutrition"].apply(parse_nutrition)
            df = df.dropna(subset=["parsed_nutrition"])

            nutrition_df = pd.DataFrame(df["parsed_nutrition"].tolist(), index=df.index)
            df = pd.concat([df, nutrition_df], axis=1)

            df["ingredients_text"] = df["ingredients"].apply(parse_ingredients)

            df = df[(df["calories"] >= 100) & (df["calories"] <= 1500)]
            df = df[(df["protein"] >= 0) & (df["protein"] <= 150)]

            df["name"] = df["name"].fillna("Unknown Recipe")
            df["name_lower"] = df["name"].str.lower()

            breakfast_pattern: str = "|".join(self.BREAKFAST_KEYWORDS)
            breakfast_mask = df["name_lower"].str.contains(breakfast_pattern, na=False)

            main_pattern: str = "|".join(self.MAIN_MEAL_KEYWORDS)
            main_mask = df["name_lower"].str.contains(main_pattern, na=False)

            dessert_pattern: str = "|".join(self.DESSERT_KEYWORDS)
            dessert_mask = df["name_lower"].str.contains(dessert_pattern, na=False)

            self.breakfast_recipes = df[breakfast_mask & ~dessert_mask].copy()
            self.main_recipes = df[main_mask & ~breakfast_mask & ~dessert_mask].copy()

            self.breakfast_recipes = self.breakfast_recipes[
                self.breakfast_recipes["calories"] >= 200
            ]
            self.main_recipes = self.main_recipes[
                (self.main_recipes["calories"] >= 300)
                & (self.main_recipes["protein"] >= 10)
            ]

            print(f"{Colors.SUCCESS}[DATA] Breakfast recipes: {len(self.breakfast_recipes):,}{Colors.END}")
            print(f"{Colors.SUCCESS}[DATA] Main meal recipes: {len(self.main_recipes):,}{Colors.END}")

            return self.breakfast_recipes, self.main_recipes

        except FileNotFoundError:
            print(f"{Colors.ERROR}[DATA] Dataset not found: {self.recipes_path}{Colors.END}")
            print(f"{Colors.WARNING}[DATA] Please download 'RAW_recipes.csv' and place it in the 'data/' folder.{Colors.END}")
            return pd.DataFrame(), pd.DataFrame()
        except Exception as e:
            print(f"{Colors.ERROR}[ERROR] Failed to load recipes: {e}{Colors.END}")
            return pd.DataFrame(), pd.DataFrame()

    def get_compliant_recipes(
        self,
        diet_type: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Filter recipes by dietary restrictions using DietEnforcer (CSP).

        Args:
            diet_type: Diet restriction to apply (e.g., 'vegan', 'keto').

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: (compliant_breakfast, compliant_mains).
        """
        if self.breakfast_recipes is None or self.main_recipes is None:
            return pd.DataFrame(), pd.DataFrame()

        breakfast_mask = self.breakfast_recipes.apply(
            lambda row: self.diet_enforcer.is_compliant(
                row["ingredients_text"], diet_type, row.get("name", "")
            ),
            axis=1,
        )
        compliant_breakfast: pd.DataFrame = self.breakfast_recipes[breakfast_mask].copy()

        main_mask = self.main_recipes.apply(
            lambda row: self.diet_enforcer.is_compliant(
                row["ingredients_text"], diet_type, row.get("name", "")
            ),
            axis=1,
        )
        compliant_mains: pd.DataFrame = self.main_recipes[main_mask].copy()

        return compliant_breakfast, compliant_mains
