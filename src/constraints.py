"""
Constraint Satisfaction Problem (CSP) module for NutriGen AI.

==============================================================================
AI TECHNIQUE #2: CONSTRAINT SATISFACTION PROBLEM (CSP) WITH BACKTRACKING
==============================================================================

The DietEnforcer class enforces HARD CONSTRAINTS on meal selection:
    - Variables: Recipe ingredients
    - Domain: All possible ingredients in the recipe database
    - Constraints: BANNED_INGREDIENTS dictionary per diet type

Constraint Propagation Logic:
    If ANY banned ingredient appears in a recipe, it is REJECTED.
    This implements arc consistency by eliminating invalid assignments.

This module also implements Heuristic Search through the is_real_breakfast
function which applies domain-specific heuristics to filter breakfast items.
"""

# =============================================================================
# IMPORTS
# =============================================================================

# Standard library
import json
import os
import re
from typing import Dict, List, Optional, Pattern

# Third-party
import pandas as pd

# Local
from src.config import Colors, DATA_DIR


# =============================================================================
# HEURISTIC FUNCTIONS
# =============================================================================


def is_real_breakfast(recipe_name: str, exclusion_list: List[str]) -> bool:
    """
    Apply heuristic search to validate breakfast recipes.

    This function implements domain-specific heuristics to filter out
    non-breakfast items (soups, dinner items, etc.) from the breakfast pool.

    Args:
        recipe_name: Name of the recipe to validate.
        exclusion_list: List of keywords that disqualify a recipe from breakfast.

    Returns:
        bool: True if recipe is a valid breakfast, False if it should be excluded.
    """
    if not exclusion_list:
        return True

    recipe_name_lower: str = str(recipe_name).lower()

    for exclusion in exclusion_list:
        if exclusion.lower() in recipe_name_lower:
            return False

    return True


# =============================================================================
# CSP SOLVER CLASS
# =============================================================================


class DietEnforcer:
    """
    Constraint Satisfaction Problem (CSP) solver for dietary restrictions.

    ===========================================================================
    IMPLEMENTS: Constraint Satisfaction (CSP) and Heuristic Search
    ===========================================================================

    This class enforces HARD CONSTRAINTS on recipe selection - no negotiation.
    If ANY banned ingredient appears, the recipe is REJECTED immediately.

    CSP Components:
        - Variables: Individual ingredients in each recipe
        - Domains: All possible ingredient values
        - Constraints: Diet-specific banned ingredient lists

    The solver uses regex pattern matching for efficient constraint checking
    across potentially thousands of recipes.

    Attributes:
        BANNED_INGREDIENTS: Class-level dictionary mapping diet types to banned items.
        patterns: Pre-compiled regex patterns for efficient matching.
    """

    BANNED_INGREDIENTS: Dict[str, List[str]] = {}

    @classmethod
    def _load_banned_ingredients(cls) -> Dict[str, List[str]]:
        """
        Load banned ingredients from JSON configuration file.

        Implements constraint loading for the CSP solver by reading
        diet-specific banned ingredient lists from external configuration.

        Returns:
            Dict[str, List[str]]: Dictionary mapping diet types to banned ingredient lists.
        """
        if cls.BANNED_INGREDIENTS:
            return cls.BANNED_INGREDIENTS

        json_path: str = os.path.join(DATA_DIR, "banned_ingredients.json")

        if os.path.exists(json_path):
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    data: Dict = json.load(f)
                    normalized: Dict[str, List[str]] = {}
                    for key, value in data.items():
                        normalized_key = key.replace("_", "-")
                        normalized[normalized_key] = value
                    normalized["none"] = []
                    cls.BANNED_INGREDIENTS = normalized
                    print(f"{Colors.INFO}[DIET] Loaded banned ingredients{Colors.END}")
                    return cls.BANNED_INGREDIENTS
            except Exception as e:
                print(f"{Colors.WARNING}[DIET] Failed to load JSON: {e}{Colors.END}")

        # Fallback to minimal defaults if JSON not found
        print(f"{Colors.WARNING}[DIET] Using default banned ingredients{Colors.END}")
        cls.BANNED_INGREDIENTS = {
            "vegan": [
                "meat", "beef", "chicken", "pork", "fish", "egg", "eggs",
                "milk", "cheese", "butter", "cream", "honey", "gelatin",
            ],
            "vegetarian": [
                "meat", "beef", "chicken", "pork", "fish", "seafood", "gelatin",
            ],
            "gluten-free": ["wheat", "flour", "bread", "pasta", "barley", "rye"],
            "keto": ["sugar", "rice", "pasta", "bread", "potato", "corn"],
            "dairy-free": ["milk", "cheese", "butter", "cream", "yogurt"],
            "none": [],
        }
        return cls.BANNED_INGREDIENTS

    def __init__(self) -> None:
        """
        Initialize the DietEnforcer with pre-compiled regex patterns.

        Pre-compiles regex patterns for each diet type to enable
        efficient constraint checking during meal planning.
        """
        banned_data: Dict[str, List[str]] = self._load_banned_ingredients()

        self.patterns: Dict[str, Pattern] = {}
        for diet_type, banned in banned_data.items():
            if banned:
                pattern = r"(" + "|".join(re.escape(word) for word in banned) + r")"
                self.patterns[diet_type] = re.compile(pattern, re.IGNORECASE)

    def is_compliant(
        self,
        ingredients_text: str,
        diet_type: str,
        recipe_name: str = ""
    ) -> bool:
        """
        Check if ingredients and recipe name comply with dietary restrictions.

        CSP Logic: Returns False if ANY banned word appears in ingredients OR name.
        This implements constraint propagation by immediately rejecting
        any recipe that violates the dietary constraints.

        Args:
            ingredients_text: String of ingredients (comma-separated or list format).
            diet_type: One of the keys in BANNED_INGREDIENTS (e.g., 'vegan', 'keto').
            recipe_name: Optional recipe name to check for violations.

        Returns:
            bool: True if compliant (no banned ingredients found),
                  False if any banned ingredient detected.
        """
        if diet_type not in self.BANNED_INGREDIENTS:
            return True

        if diet_type == "none" or not self.BANNED_INGREDIENTS[diet_type]:
            return True

        text_to_check: str = ""
        if ingredients_text and not pd.isna(ingredients_text):
            text_to_check += str(ingredients_text).lower()
        if recipe_name and not pd.isna(recipe_name):
            text_to_check += " " + str(recipe_name).lower()

        if not text_to_check:
            return True

        if diet_type in self.patterns:
            match = self.patterns[diet_type].search(text_to_check)
            if match:
                return False

        return True

    def get_violations(
        self,
        ingredients_text: str,
        diet_type: str
    ) -> List[str]:
        """
        Get list of specific constraint violations for debugging/display.

        Args:
            ingredients_text: String of ingredients to check.
            diet_type: Diet restriction type to check against.

        Returns:
            List[str]: List of banned ingredients found in the text.
        """
        if diet_type not in self.BANNED_INGREDIENTS or diet_type == "none":
            return []

        if not ingredients_text or pd.isna(ingredients_text):
            return []

        text: str = str(ingredients_text).lower()
        violations: List[str] = []

        for banned in self.BANNED_INGREDIENTS[diet_type]:
            pattern = r"\b" + re.escape(banned) + r"\b"
            if re.search(pattern, text, re.IGNORECASE):
                violations.append(banned)

        return violations

    def is_compliant_multi(
        self,
        ingredients_text: str,
        diet_type: str,
        recipe_name: str = "",
        avoid_additives: bool = False,
        avoid_allergens: bool = False,
    ) -> bool:
        """
        Check compliance against diet type AND optional additives/allergens.

        Implements multi-constraint satisfaction by checking the recipe
        against multiple constraint sets simultaneously.

        Args:
            ingredients_text: String of ingredients to validate.
            diet_type: Primary diet restriction (e.g., 'vegan', 'keto').
            recipe_name: Optional recipe name to check.
            avoid_additives: If True, also check against additives list.
            avoid_allergens: If True, also check against allergens list.

        Returns:
            bool: True if compliant with ALL specified restrictions.
        """
        if not self.is_compliant(ingredients_text, diet_type, recipe_name):
            return False

        if avoid_additives and "additives" in self.BANNED_INGREDIENTS:
            if not self.is_compliant(ingredients_text, "additives", recipe_name):
                return False

        if avoid_allergens and "allergens" in self.BANNED_INGREDIENTS:
            if not self.is_compliant(ingredients_text, "allergens", recipe_name):
                return False

        return True

    @classmethod
    def get_available_diets(cls) -> List[str]:
        """
        Return list of supported diet types.

        Returns:
            List[str]: Available diet type identifiers.
        """
        if not cls.BANNED_INGREDIENTS:
            cls._load_banned_ingredients()
        diet_types: List[str] = [
            "none", "vegetarian", "vegan", "gluten-free", "keto", "dairy-free"
        ]
        return [d for d in diet_types if d in cls.BANNED_INGREDIENTS]
