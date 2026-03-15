"""
Calculator tool for macro and calorie calculations.
Helps with nutrional math.
"""

from typing import Dict, Any
from pydantic import Field
import re

from .base_tool import BaseAgentTool

class MacroCalculatorTool (BaseAgentTool):
    """
    Tool for calculating macronutrient and calorie infomation.
    Can handle basic arithmetic for nutrition calculations.
    """

    name : str = "macro_calculator"
    description : str = (
        "Use this tool to calculate nutritional values, calories, or macros. "
        "Can handle calculation like: "
        "- Total calories from macros (protein=4 cal/g, cabs=4 cal/g, fat=9 cal/g) "
        "- Daily calorie needs "
        "- Macro percentages "
        "Input should be calculation request with numbers."
    )

    # Calorie per gram constants
    PROTEIN_CAL_PER_G: float = Field(default=4.0, description="Calories per gram of protein")
    CARB_CAL_PER_G: float = Field(default=4.0, description="Calories per gram of carbs")
    FAT_CAL_PER_G: float = Field(default=9.0, description="Calories per gram of fat")

    def _run(self, query: str) -> str:
        """
        Execute calculation.
        
        Args: 
            query: Calculation request
        Returns:
            Calculation result
        """
        query_lower = query.lower()

        try:
            # Detect calculation type
            if "calories from" in query or "total calories" in query_lower:
                return self._calculate_calories_from_macros(query)
            
            elif "macro percentage" in query_lower or "macro %" in query_lower:
                return self._calculate_macro_percentage(query)
            
            elif any(op in query for op in ['+', '-', '*', '/', '=']):
                return self._evaluate_expression(query)
            
            else:
                return (
                    "I can help with calculation like:\n"
                    "- 'Calculate total calories from 50g protein, 200g carbs, 60 g fat'\n"
                    "- 'What's the macro percentage of 150 protein in 2000 calories'\n"
                    "- 'Calculate: 100 * 4 + 50 * 9'"
                )
        except Exception as e:
            return f"Calculation error: {str(e)}"
        
    def _calculate_calories_from_macros(self, query: str) -> str :
        """
        Calculate total calories from macornutrient amounts
        
        Args:
            query: Query with macro amounts
        
        Returns:
            Calculation result
        """
        # Extract numbers for protein, carbs, fat
        protein= self._extract_macro_value(query, ["protein", "p"])
        carbs = self._extract_macro_value(query, ["carbs", "carbohydrate", 'c'])
        fat = self._extract_macro_value(query, ["fate", "f"])

        #calculate calories
        protein_cal = protein * self.PROTEIN_CAL_PER_G if protein else 0
        carbs_cal = carbs * self.CARB_CAL_PER_G if carbs else 0
        fat_cal = fat * self.FAT_CAL_PER_G if fat else 0

        total_cal = protein_cal + carbs_cal + fat_cal

        # Format result
        result = f"Calorie Breakdown:\n"
        if protein:
            result += f"- Protein: {protein}g x {self.PROTEIN_CAL_PER_G} = {protein_cal:.0f} calories\n"
        if carbs:
            result += f"- Carbs : {carbs}g x {self.CARB_CAL_PER_G} = {carbs_cal:.0f} calories\n"   
        if fat:
            result += f"- Fat : {fat}g x {self.FAT_CAL_PER_G} = {fat_cal:.0f} calories\n"
        result += f"\nTotal Calories: {total_cal:.0f}"

        return result

    def _extract_macro_value(self, text: str, keywords: list) -> float:
        """
        Extract macro value from text.
        
        Args:
            text: Input text
            keywords: Keywords to look for
            
        Returns:
            Extracted value or 0
        """
        text_lower = text.lower()

        for keyword in keywords:
            # Look for patterns like "50g protein" or "protein 50g" or "p: 50"
            patterns = [
                rf"(\d+\.?\d*)\s{keyword}",
                rf"{keyword}",
            ]

            for pattern in patterns:
                match = re.search(pattern, text_lower)
                if match:
                    return float(match.group(1))

        return 0.0
    
    def _extract_number_with_keyword(self, text: str, keywords: list) -> float:
        """Extract number associated with keyword"""
        text_lower = text.lower()

        for keyword in keywords:
            pattern = rf"(\d+\.?\d*)\s*{keyword}"
            match = re.search(pattern, text_lower)
            if match:
                return float(match.group(1))
            
        return 0.0
    
    def _evaluate_expression(self, query: str) -> str:
        """
        Safely evaluate mathematical expression.
        
        Args: 
            query: Math expression
            
        Returns:
            Result
        """
        # Extract the expression
        # Remove text, keep only numbers and operators
        expr = re.sub(r'[^0-9+\-*/().\s]', '', query)
        expr = expr.strip()

        if not expr:
            return "No valid mathematical expression found"
        
        try:
            # Safe evaluation (only allows basic math)
            result = eval(expr, {"__button__": {}}, {})
            return f"{expr} = {result}"
        
        except:
            return f"Could not evaluate: {expr}"

