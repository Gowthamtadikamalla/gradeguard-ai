"""
AI Agent for providing personalized academic guidance to students.
Uses Hugging Face Inference API to generate actionable recommendations for at-risk students
and encouragement/growth guidance for passing students.
"""
import os
import json
from typing import Dict, List, Optional
from huggingface_hub import InferenceClient


class AcademicGuidanceAgent:
    """
    Analyzes student features and provides personalized recommendations
    for improving academic performance (at-risk students) or maintaining/excelling (passing students).
    """
    
    def __init__(self, token: Optional[str] = None, model_id: str = "meta-llama/Llama-3.1-8B-Instruct"):
        """
        Initialize the AI agent.
        
        Args:
            token: Hugging Face API token. If None, will try HF_API_TOKEN env var.
        """
        api_key = token or os.getenv("HF_API_TOKEN")
        if not api_key:
            raise ValueError(
                "Hugging Face token not provided. Set HF_API_TOKEN environment variable "
                "or pass token parameter."
            )
        
        self.client = InferenceClient(model=model_id, token=api_key)
        self.model_id = model_id
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt for the AI agent."""
        return """You are an expert academic advisor helping at-risk students improve their academic performance.
Your role is to analyze student profiles and provide personalized, actionable recommendations.

Guidelines:
1. Be encouraging and supportive - focus on what students CAN do
2. Provide specific, actionable steps (not vague advice)
3. Prioritize recommendations (most impactful first)
4. Consider the student's context (age, grade, resources available)
5. Be realistic about timelines
6. Focus on areas where the student is struggling

Always structure your response as a JSON object with this format:
{
    "recommendations": [
        {
            "priority": 1,
            "category": "Attendance|TestScore|GPA|General",
            "action": "Specific actionable step",
            "reason": "Why this helps",
            "timeline": "Expected timeline for improvement"
        }
    ],
    "summary": "Brief encouraging summary"
}

Provide 3-5 recommendations, prioritized by impact."""
    
    def _format_failed_conditions(self, failed_conditions: Dict[str, bool]) -> str:
        """Format failed conditions for the prompt."""
        conditions = []
        if failed_conditions.get("gpa", False):
            conditions.append("GPA is below 2.0 threshold")
        if failed_conditions.get("test_score", False):
            conditions.append("Average test score is below 73 threshold")
        if failed_conditions.get("attendance", False):
            conditions.append("Attendance rate is below 85% threshold")
        
        if not conditions:
            return "General academic improvement needed"
        
        return "Areas needing improvement:\n- " + "\n- ".join(conditions)
    
    def _format_shap_factors(self, shap_factors: Optional[List[Dict]]) -> str:
        """Format SHAP top factors for the prompt."""
        if not shap_factors:
            return ""
        
        lines = ["\nModel explanation (top contributing factors):"]
        for factor in shap_factors:
            direction = factor.get("direction", "negative")
            impact = factor.get("impact", 0.0)
            desc = factor.get("description") or ("Increases pass probability" if impact >= 0 else "Decreases pass probability")
            lines.append(f"- {factor.get('feature')}: {direction} impact ({impact:+.3f}) - {desc}")
        return "\n".join(lines)
    
    def _build_prompt(self, features: Dict, failed_conditions: Dict[str, bool], shap_factors: Optional[List[Dict]] = None) -> str:
        """Build the user prompt with student information."""
        attendance_pct = features.get('AttendanceRate', 0) * 100
        shap_section = self._format_shap_factors(shap_factors)
        
        prompt = f"""Analyze this at-risk student's profile and provide personalized guidance:

Student Profile:
- Age: {features.get('Age', 'N/A')}, Grade: {features.get('Grade', 'N/A')}
- Attendance Rate: {attendance_pct:.1f}% (Target: 85%)
- Study Hours per day: {features.get('StudyHours', 0):.1f} hours
- Extracurricular Activities: {'Yes' if features.get('Extracurricular', 0) == 1 else 'No'}
- Part-time Job: {'Yes' if features.get('PartTimeJob', 0) == 1 else 'No'}
- Parent Support: {'Yes' if features.get('ParentSupport', 0) == 1 else 'No'}
- Internet Access: {'Yes' if features.get('InternetAccess', 0) == 1 else 'No'}
- Free Time Level: {features.get('FreeTime', 'N/A')}/5
- Social Activity Level: {features.get('GoOut', 'N/A')}/5
- Parental Education: {features.get('ParentalEducation', 'N/A')}
- School Type: {features.get('SchoolType', 'N/A')}
- Location: {features.get('Locale', 'N/A')}
- Socioeconomic Status: Quartile {features.get('SES_Quartile', 'N/A')}/4

{self._format_failed_conditions(failed_conditions)}
{shap_section}

Provide personalized recommendations that:
1. Address the specific areas where the student is struggling
2. Are realistic given their circumstances
3. Include actionable steps they can take immediately
4. Consider their available resources (internet, parent support, etc.)

Return your response as a valid JSON object following the format specified in the system prompt."""
        
        return prompt
    
    def _parse_response(self, response) -> Dict:
        """Parse the Hugging Face API response and extract recommendations."""
        try:
            # Handle different response formats from Hugging Face API
            if isinstance(response, str):
                content = response.strip()
            elif isinstance(response, dict):
                if "choices" in response:
                    choice = response["choices"][0]
                    if isinstance(choice, dict):
                        if "message" in choice:
                            message = choice["message"]
                            content = message.get("content", "") if isinstance(message, dict) else getattr(message, "content", "")
                        else:
                            content = choice.get("text", "")
                    else:
                        content = getattr(choice, "content", "")
                elif "generated_text" in response:
                    content = response["generated_text"].strip()
                elif "text" in response:
                    content = response["text"].strip()
                else:
                    content = str(response).strip()
            else:
                # Try to get content attribute (for object responses)
                content = getattr(response, "content", str(response)).strip()
            
            if not content:
                raise ValueError("Empty response from API")
            
            # Try to extract JSON from the response
            # Sometimes the model wraps JSON in markdown code blocks
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            
            # Parse JSON
            parsed = json.loads(content)
            
            # Validate structure
            if "recommendations" not in parsed:
                raise ValueError("Response missing 'recommendations' field")
            
            # Ensure recommendations are properly formatted
            recommendations = parsed.get("recommendations", [])
            if not isinstance(recommendations, list):
                raise ValueError("Recommendations must be a list")
            
            # Validate each recommendation
            validated_recommendations = []
            for rec in recommendations:
                if not all(key in rec for key in ["priority", "category", "action", "reason", "timeline"]):
                    continue  # Skip invalid recommendations
                validated_recommendations.append(rec)
            
            return {
                "recommendations": validated_recommendations,
                "summary": parsed.get("summary", "Focus on improving the areas identified above.")
            }
            
        except json.JSONDecodeError as e:
            # If JSON parsing fails, create a fallback response
            return {
                "recommendations": [
                    {
                        "priority": 1,
                        "category": "General",
                        "action": "Review your study habits and attendance patterns",
                        "reason": "Improving these fundamental areas will have the greatest impact",
                        "timeline": "2-4 weeks"
                    }
                ],
                "summary": "Focus on consistent attendance and dedicated study time."
            }
        except Exception as e:
            # Fallback on any error
            return {
                "recommendations": [
                    {
                        "priority": 1,
                        "category": "General",
                        "action": "Consult with your academic advisor for personalized guidance",
                        "reason": "Professional guidance can help identify specific improvement areas",
                        "timeline": "Immediate"
                    }
                ],
                "summary": "Seek support from academic advisors and teachers."
            }
    
    def generate_recommendations(
        self,
        student_features: Dict,
        prediction: int,
        confidence: float,
        failed_conditions: Dict[str, bool],
        shap_factors: Optional[List[Dict]] = None
    ) -> Dict:
        """
        Generate personalized recommendations based on student's weak areas.
        
        Args:
            student_features: Student's input features
            prediction: 0 (Fail) or 1 (Pass)
            confidence: Model's confidence in prediction (0.0-1.0)
            failed_conditions: Which conditions failed (GPA, TestScore, Attendance)
        
        Returns:
            dict with recommendations, priority actions, and improvement plan
        """
        # Only generate recommendations for at-risk students
        if prediction == 1:
            return None
        
        try:
            # Build context-aware prompt
            prompt = self._build_prompt(student_features, failed_conditions, shap_factors)
            
            # Call Hugging Face API
            response = self.client.chat_completion(
                messages=[
                    {"role": "system", "content": self._get_system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=800,
            )
            return self._parse_response(response)
            
        except Exception as e:
            raise RuntimeError("AI agent unavailable") from e
    
    def generate_encouragement(
        self,
        student_features: Dict,
        confidence: float,
        shap_factors: Optional[List[Dict]] = None
    ) -> Dict:
        """
        Generate personalized encouragement and growth guidance for passing students.
        
        Args:
            student_features: Student's input features
            confidence: Model's confidence in prediction (0.0-1.0)
            shap_factors: SHAP explanation factors showing what's working well
        
        Returns:
            dict with recommendations for maintaining excellence and growth opportunities
        """
        try:
            # Build context-aware prompt for passing students
            attendance_pct = student_features.get('AttendanceRate', 0) * 100
            shap_section = self._format_shap_factors(shap_factors)
            
            prompt = f"""Analyze this successful student's profile and provide personalized encouragement and growth guidance:

Student Profile:
- Age: {student_features.get('Age', 'N/A')}, Grade: {student_features.get('Grade', 'N/A')}
- Attendance Rate: {attendance_pct:.1f}% (Excellent: above 85%)
- Study Hours per day: {student_features.get('StudyHours', 0):.1f} hours
- Extracurricular Activities: {'Yes' if student_features.get('Extracurricular', 0) == 1 else 'No'}
- Part-time Job: {'Yes' if student_features.get('PartTimeJob', 0) == 1 else 'No'}
- Parent Support: {'Yes' if student_features.get('ParentSupport', 0) == 1 else 'No'}
- Internet Access: {'Yes' if student_features.get('InternetAccess', 0) == 1 else 'No'}
- Free Time Level: {student_features.get('FreeTime', 'N/A')}/5
- Social Activity Level: {student_features.get('GoOut', 'N/A')}/5
- Parental Education: {student_features.get('ParentalEducation', 'N/A')}
- School Type: {student_features.get('SchoolType', 'N/A')}
- Location: {student_features.get('Locale', 'N/A')}
- Socioeconomic Status: Quartile {student_features.get('SES_Quartile', 'N/A')}/4

{shap_section}

This student is currently in good academic standing (GPA >= 2.0 and Average Test Score >= 73).

Provide personalized guidance that:
1. Celebrates their current success and strengths
2. Offers specific tips to maintain their excellent performance
3. Suggests opportunities for further growth and excellence
4. Warns about common pitfalls that successful students might face
5. Is encouraging and motivating

Return your response as a valid JSON object following the format specified in the system prompt."""
            
            system_prompt = """You are an expert academic advisor helping successful students maintain and excel in their academic performance.
Your role is to analyze student profiles and provide personalized, encouraging guidance for continued success.

Guidelines:
1. Be positive and celebratory - acknowledge their current success
2. Provide specific, actionable steps for maintaining excellence and growth
3. Focus on prevention - help them avoid common pitfalls
4. Suggest opportunities for further growth and excellence
5. Consider their context (age, grade, resources available)
6. Keep recommendations concise and encouraging

Always structure your response as a JSON object with this format:
{
    "recommendations": [
        {
            "priority": 1,
            "category": "Maintenance|Growth|Prevention|General",
            "action": "Specific actionable step",
            "reason": "Why this helps maintain/excel",
            "timeline": "When to implement this"
        }
    ],
    "summary": "Brief encouraging summary celebrating their success"
}

Provide 2-4 concise recommendations focused on maintenance and growth."""
            
            # Call Hugging Face API
            response = self.client.chat_completion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=600,  # Shorter for passing students
            )
            return self._parse_response(response)
            
        except Exception as e:
            raise RuntimeError("AI agent unavailable") from e

