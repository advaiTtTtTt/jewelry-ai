"""
Budget-Aware Substitution Advisor
==================================
Calculates jewelry design costs and suggests budget-friendly substitutions
with visual similarity scoring.

Prices are approximate retail estimates (per-carat for gems, per-gram for metals).
Visual similarity is a heuristic based on color proximity, transparency match, and
overall appearance similarity.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════
# PRICE DATABASE — approximate retail prices (USD)
# ═══════════════════════════════════════════════════════════════════════

# Per-carat prices (gemstones) — mid-range quality estimates
GEMSTONE_PRICES = {
    "diamond":        500.0,   # 1ct, VS2, G color
    "ruby":           200.0,   # Good quality, heated
    "sapphire":       150.0,   # Ceylon, heated
    "emerald":        180.0,   # Colombian, minor oil
    "amethyst":        20.0,   # Abundant
    "cubic_zirconia":  10.0,   # Synthetic
}

# Per-gram prices (metals)
METAL_PRICES = {
    "yellow_gold":   65.0,    # 18K
    "white_gold":    70.0,    # 18K + rhodium plating
    "rose_gold":     63.0,    # 18K
    "platinum":     100.0,    # 950 Pt
    "silver":         2.5,    # Sterling 925
}

# Default component weights/sizes for cost estimation when no mesh data available
DEFAULT_COMPONENT_SPECS = {
    "gemstone": {"carats": 1.0},       # Assume 1 carat center stone
    "metal":    {"grams": 5.0},        # Assume 5g metal weight
    "prong":    {"grams": 0.5},        # Small prongs
    "setting":  {"grams": 1.0},        # Setting mount
    "bail":     {"grams": 0.8},        # Pendant bail
    "clasp":    {"grams": 0.5},        # Necklace/bracelet clasp
}

# ═══════════════════════════════════════════════════════════════════════
# VISUAL SIMILARITY MATRIX
# ═══════════════════════════════════════════════════════════════════════
# Pre-computed similarity scores between material pairs.
# 1.0 = identical appearance, 0.0 = completely different.
# Based on color proximity, transparency, and sparkle characteristics.

GEMSTONE_SIMILARITY = {
    ("diamond", "cubic_zirconia"):  0.95,  # Nearly identical appearance
    ("diamond", "amethyst"):        0.20,  # Different color
    ("diamond", "sapphire"):        0.15,  # Blue vs clear
    ("diamond", "ruby"):            0.10,  # Red vs clear
    ("diamond", "emerald"):         0.10,  # Green vs clear
    ("ruby", "amethyst"):           0.35,  # Both warm tones
    ("ruby", "sapphire"):           0.45,  # Same crystal, different color
    ("ruby", "emerald"):            0.30,  # Both colored stones
    ("ruby", "cubic_zirconia"):     0.15,  # Red vs clear
    ("sapphire", "amethyst"):       0.55,  # Blue/purple similarity
    ("sapphire", "emerald"):        0.35,  # Both cool-toned gems
    ("sapphire", "cubic_zirconia"): 0.15,  # Blue vs clear
    ("emerald", "amethyst"):        0.30,  # Green vs purple
    ("emerald", "cubic_zirconia"):  0.10,  # Green vs clear
    ("amethyst", "cubic_zirconia"): 0.25,  # Purple vs clear
}

METAL_SIMILARITY = {
    ("yellow_gold", "rose_gold"):   0.75,  # Both warm gold tones
    ("yellow_gold", "white_gold"):  0.50,  # Warm vs cool
    ("yellow_gold", "platinum"):    0.45,  # Warm vs neutral
    ("yellow_gold", "silver"):      0.40,  # Warm vs cool
    ("white_gold", "platinum"):     0.90,  # Nearly identical appearance
    ("white_gold", "silver"):       0.80,  # Similar color, different weight
    ("white_gold", "rose_gold"):    0.55,  # Cool vs warm
    ("rose_gold", "platinum"):      0.40,  # Pink vs grey
    ("rose_gold", "silver"):        0.35,  # Pink vs grey
    ("platinum", "silver"):         0.85,  # Very similar appearance
}


def _get_similarity(mat_a: str, mat_b: str, table: dict) -> float:
    """Look up visual similarity between two materials (order-independent)."""
    if mat_a == mat_b:
        return 1.0
    key = (mat_a, mat_b) if (mat_a, mat_b) in table else (mat_b, mat_a)
    return table.get(key, 0.1)  # Default low similarity for unknown pairs


class BudgetAdvisor:
    """
    Analyzes jewelry design costs and suggests budget-friendly substitutions.

    Usage:
        advisor = BudgetAdvisor()
        cost = advisor.calculate_cost(design_config)
        suggestions = advisor.suggest_substitutions(design_config, budget=200)
    """

    def calculate_cost(self, design_config: dict) -> dict:
        """
        Calculate total cost of a jewelry design.

        Args:
            design_config: {
                "gemstone": {"material": "diamond", "carats": 1.0},
                "metal": {"material": "yellow_gold", "grams": 5.0},
                "prong": {"material": "yellow_gold", "grams": 0.5},
                "setting": {"material": "yellow_gold", "grams": 1.0},
                ...
            }

        Returns:
            {
                "total": float,
                "breakdown": {
                    "gemstone": {"material": "diamond", "cost": 500.0, "quantity": "1.0 carats"},
                    "metal": {"material": "yellow_gold", "cost": 325.0, "quantity": "5.0 grams"},
                    ...
                },
                "currency": "USD"
            }
        """
        breakdown = {}
        total = 0.0

        for component, config in design_config.items():
            material = config.get("material", "")
            cost = 0.0
            quantity_str = ""

            if material in GEMSTONE_PRICES:
                carats = config.get("carats", DEFAULT_COMPONENT_SPECS.get(component, {}).get("carats", 1.0))
                cost = GEMSTONE_PRICES[material] * carats
                quantity_str = f"{carats} carats"

            elif material in METAL_PRICES:
                grams = config.get("grams", DEFAULT_COMPONENT_SPECS.get(component, {}).get("grams", 5.0))
                cost = METAL_PRICES[material] * grams
                quantity_str = f"{grams} grams"

            breakdown[component] = {
                "material": material,
                "cost": round(cost, 2),
                "quantity": quantity_str,
            }
            total += cost

        return {
            "total": round(total, 2),
            "breakdown": breakdown,
            "currency": "USD",
        }

    def suggest_substitutions(
        self,
        design_config: dict,
        budget: float,
        min_similarity: float = 0.0,
    ) -> dict:
        """
        Suggest material substitutions that fit within the given budget.

        Args:
            design_config: Same format as calculate_cost() input
            budget: Maximum total budget in USD
            min_similarity: Minimum visual similarity score (0.0–1.0) for suggestions

        Returns:
            {
                "current_total": float,
                "budget": float,
                "over_budget": bool,
                "suggestions": [
                    {
                        "component": "gemstone",
                        "replace": "diamond",
                        "with": "cubic_zirconia",
                        "savings": 490.0,
                        "new_cost": 10.0,
                        "visual_similarity": 0.95,
                        "new_total": 345.0
                    },
                    ...
                ],
                "best_combo": {
                    "changes": [...],       # Optimal set of swaps to fit budget
                    "new_total": float,
                    "total_savings": float,
                }
            }
        """
        current = self.calculate_cost(design_config)
        current_total = current["total"]

        if current_total <= budget:
            return {
                "current_total": current_total,
                "budget": budget,
                "over_budget": False,
                "suggestions": [],
                "best_combo": None,
                "message": "Design is already within budget!",
            }

        # Generate all possible single-component substitutions
        suggestions = []

        for component, config in design_config.items():
            current_material = config.get("material", "")
            current_cost = current["breakdown"].get(component, {}).get("cost", 0)

            # Try all cheaper alternatives
            if current_material in GEMSTONE_PRICES:
                alternatives = {
                    k: v for k, v in GEMSTONE_PRICES.items()
                    if v < GEMSTONE_PRICES[current_material] and k != current_material
                }
                similarity_table = GEMSTONE_SIMILARITY
                carats = config.get("carats", 1.0)
                for alt_name, alt_price_per in sorted(alternatives.items(), key=lambda x: x[1]):
                    alt_cost = alt_price_per * carats
                    sim = _get_similarity(current_material, alt_name, similarity_table)
                    if sim >= min_similarity:
                        savings = current_cost - alt_cost
                        suggestions.append({
                            "component": component,
                            "replace": current_material,
                            "with": alt_name,
                            "savings": round(savings, 2),
                            "new_cost": round(alt_cost, 2),
                            "visual_similarity": sim,
                            "new_total": round(current_total - savings, 2),
                        })

            elif current_material in METAL_PRICES:
                alternatives = {
                    k: v for k, v in METAL_PRICES.items()
                    if v < METAL_PRICES[current_material] and k != current_material
                }
                similarity_table = METAL_SIMILARITY
                grams = config.get("grams", DEFAULT_COMPONENT_SPECS.get(component, {}).get("grams", 5.0))
                for alt_name, alt_price_per in sorted(alternatives.items(), key=lambda x: x[1]):
                    alt_cost = alt_price_per * grams
                    sim = _get_similarity(current_material, alt_name, similarity_table)
                    if sim >= min_similarity:
                        savings = current_cost - alt_cost
                        suggestions.append({
                            "component": component,
                            "replace": current_material,
                            "with": alt_name,
                            "savings": round(savings, 2),
                            "new_cost": round(alt_cost, 2),
                            "visual_similarity": sim,
                            "new_total": round(current_total - savings, 2),
                        })

        # Sort by best savings-to-similarity ratio (prefer high savings + high similarity)
        suggestions.sort(
            key=lambda s: s["savings"] * (0.5 + s["visual_similarity"]),
            reverse=True,
        )

        # Find the best combination of swaps that brings total under budget
        best_combo = self._find_best_combo(design_config, suggestions, budget, current_total)

        return {
            "current_total": current_total,
            "budget": budget,
            "over_budget": True,
            "suggestions": suggestions,
            "best_combo": best_combo,
        }

    @staticmethod
    def _find_best_combo(
        design_config: dict,
        suggestions: list,
        budget: float,
        current_total: float,
    ) -> Optional[dict]:
        """
        Find the optimal set of substitutions (one per component max)
        that minimizes visual change while staying under budget.

        Uses a greedy approach: pick the highest-similarity swap per component
        that provides enough cumulative savings.
        """
        # Group suggestions by component
        by_component = {}
        for s in suggestions:
            comp = s["component"]
            if comp not in by_component:
                by_component[comp] = []
            by_component[comp].append(s)

        # Sort each component's options by similarity (highest first)
        for comp in by_component:
            by_component[comp].sort(key=lambda x: x["visual_similarity"], reverse=True)

        # Greedy: try high-similarity swaps first, accumulate savings
        chosen = []
        remaining_excess = current_total - budget

        # First pass: try single highest-similarity swap per component
        for comp, options in sorted(by_component.items(), key=lambda x: -x[1][0]["savings"]):
            if remaining_excess <= 0:
                break
            # Pick the option with best similarity that provides meaningful savings
            for option in options:
                if option["savings"] > 0:
                    chosen.append(option)
                    remaining_excess -= option["savings"]
                    break

        if not chosen:
            return None

        total_savings = sum(c["savings"] for c in chosen)
        return {
            "changes": chosen,
            "new_total": round(current_total - total_savings, 2),
            "total_savings": round(total_savings, 2),
            "fits_budget": (current_total - total_savings) <= budget,
        }
