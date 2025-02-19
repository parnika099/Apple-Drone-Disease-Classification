import google.generativeai as genai
from typing import Dict, List, Optional
import datetime
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure the Gemini API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") 
model = None

if not GEMINI_API_KEY:
    print("Warning: GEMINI_API_KEY environment variable is not set")
else:
    genai.configure(api_key=GEMINI_API_KEY)
    
    # Create a Gemini model instance - use current available models
    try:
        # Use gemini-1.5-flash which is currently available and free
        model = genai.GenerativeModel('gemini-1.5-flash')
        
    except Exception as e:
        print(f"Failed to initialize Gemini model: {str(e)}")
        model = None

class AppleHarvestChatbot:
    def __init__(self):
        self.context_data = {}
        self.chat_history = []
        self.model = model # Use the module level model
        
        # Define system prompts for different scenarios
        self.SYSTEM_PROMPTS = {
            "general": """You are an expert agricultural advisor specializing in apple cultivation. 
            Your role is to assist farmers with:
            1. Analyzing harvest data and quality metrics
            2. Providing disease identification and treatment recommendations
            3. Offering market insights and pricing strategies
            4. Suggesting best practices for apple cultivation
            
            Base your responses on the provided harvest data and maintain a professional yet friendly tone.""",
            
            "disease_analysis": """Focus on analyzing disease patterns in the harvest data.
            Consider:
            - Disease types and their prevalence
            - Potential causes and environmental factors
            - Treatment recommendations
            - Prevention strategies""",
            
            "market_analysis": """Analyze market implications of the harvest quality.
            Consider:
            - Current market trends
            - Price expectations for different apple grades
            - Storage recommendations
            - Marketing strategies"""
        }
    
    def update_context(self, new_context: Dict):
        """Update the context data used for generating responses"""
        self.context_data.update(new_context)
    
    def _format_context(self) -> str:
        """Format the context data for inclusion in prompts"""
        context = []
        
        if "session_info" in self.context_data:
            session = self.context_data["session_info"]
            context.append(f"Session ID: {session.get('session_id', 'N/A')}")
            context.append(f"Timestamp: {session.get('timestamp', 'N/A')}")
            context.append(f"Total Apples: {session.get('total_apples', 0)}")
        
        if "condition_counts" in self.context_data:
            counts = self.context_data["condition_counts"]
            context.append("\nApple Conditions:")
            for condition, count in counts.items():
                context.append(f"- {condition}: {count}")
        
        if "quality_results" in self.context_data:
            context.append("\nRecent Quality Analysis:")
            for result in self.context_data["quality_results"][:3]:
                context.append(f"- {result}")
        
        return "\n".join(context)
    
    def _create_prompt(self, user_question: str, prompt_type: str = "general") -> str:
        """Create a context-aware prompt for the model"""
        system_prompt = self.SYSTEM_PROMPTS[prompt_type]
        context = self._format_context()
        
        return f"""{system_prompt}

Current Context:
{context}

User Question: {user_question}

Please provide a helpful, informative response based on the available data and your expertise. 
If specific data is not available, make general recommendations based on best practices."""
    
    def get_response(self, 
                   question: str, 
                   prompt_type: str = "general", 
                   language: str = "english") -> str:
        """Generate a response to the user's question"""
        # First, add the user question to chat history
        self.chat_history.append({
            "role": "user",
            "content": question,
            "timestamp": datetime.datetime.now().isoformat()
        })
        
        # Check if model is available
        if self.model is None:
            # Use rule-based fallback responses
            fallback_response = self._get_fallback_response(question, prompt_type)
            
            # Add the fallback response to chat history
            self.chat_history.append({
                "role": "assistant",
                "content": fallback_response,
                "timestamp": datetime.datetime.now().isoformat()
            })
            
            return fallback_response
        
        try:
            # Create the prompt
            prompt = self._create_prompt(question, prompt_type)
            
            # Generate response
            response = self.model.generate_content(prompt)
            
            if not response or not hasattr(response, 'text') or not response.text.strip():
                # Handle empty responses
                fallback_response = self._get_fallback_response(question, prompt_type)
                
                # Add fallback response to chat history
                self.chat_history.append({
                    "role": "assistant",
                    "content": fallback_response,
                    "timestamp": datetime.datetime.now().isoformat()
                })
                
                return fallback_response
            
            # Add successful response to chat history
            self.chat_history.append({
                "role": "assistant",
                "content": response.text,
                "timestamp": datetime.datetime.now().isoformat()
            })
            
            return response.text
            
        except Exception as e:
            # Get appropriate fallback response
            fallback_response = self._get_fallback_response(question, prompt_type)
            
            # Add error context if it's not a production environment
            error_message = f"I apologize, but I encountered an error. {fallback_response}"
            
            # Add fallback response to chat history
            self.chat_history.append({
                "role": "assistant",
                "content": error_message,
                "timestamp": datetime.datetime.now().isoformat()
            })
            
            return error_message
    
    def _get_fallback_response(self, question: str, prompt_type: str) -> str:
        """Get an appropriate fallback response based on question type"""
        # Basic question categorization for hardcoded responses
        question_lower = question.lower()
        
        # Disease related questions
        if prompt_type == "disease_analysis" or any(word in question_lower for word in ["disease", "infection", "treatment", "cure", "symptoms"]):
            return self._get_disease_fallback()
            
        # Market related questions
        elif prompt_type == "market_analysis" or any(word in question_lower for word in ["market", "price", "sell", "value", "cost"]):
            return self._get_market_fallback()
            
        # Cultivation questions
        elif any(word in question_lower for word in ["grow", "cultivate", "plant", "soil", "water", "fertilizer"]):
            return self._get_cultivation_fallback()
            
        # Harvest questions
        elif any(word in question_lower for word in ["harvest", "pick", "collect", "yield", "production"]):
            return self._get_harvest_fallback()
            
        # General fallback
        else:
            return """
Thank you for your question about apple cultivation. Based on general best practices:

1. Regular monitoring of your orchard is key to early detection of issues
2. Integrated Pest Management (IPM) can reduce chemical inputs while maintaining quality
3. Proper pruning and tree spacing improves air circulation and reduces disease pressure
4. Soil health directly impacts apple quality - consider regular soil testing
5. Post-harvest handling significantly affects storage life and marketability

If you have more specific questions about disease management, harvesting techniques, or market considerations, please let me know.
            """
    
    def _get_disease_fallback(self) -> str:
        """Provide fallback response for disease-related questions"""
        counts = self.context_data.get("condition_counts", {})
        total = sum(counts.values()) if counts else 0
        
        if not total:
            return """
Common apple diseases include:
- Apple Scab: Appears as olive-green to black spots on leaves and fruit
- Apple Blotch: Causes dark, irregular spots on fruit
- Bitter Rot: Forms sunken lesions with concentric rings
- Apple Rot: Causes soft, brown decay of fruit tissue

For management, consider fungicide treatments, proper pruning for airflow, and orchard sanitation.
            """
        
        # If we have condition data, create a focused response
        highest_disease = max(
            [k for k in counts.keys() if k != "Normal_Apple"], 
            key=lambda k: counts.get(k, 0),
            default=None
        )
        
        if highest_disease == "Blotch_Apple":
            return """
Blotch Management Recommendations:
1. Apply protective fungicides during the growing season
2. Improve air circulation through proper tree spacing and pruning
3. Remove fallen leaves and fruit to reduce disease spread
4. Consider copper-based sprays as a preventive measure
            """
        elif highest_disease == "Rot_Apple":
            return """
Rot Management Recommendations:
1. Handle fruit carefully during harvest to prevent wounds
2. Cool harvested fruit promptly to slow pathogen development
3. Remove infected fruit from the orchard regularly
4. Apply appropriate fungicides before harvest
5. Ensure proper storage conditions with good ventilation
            """
        elif highest_disease == "Scab_Apple":
            return """
Scab Management Recommendations:
1. Apply preventive fungicides from green tip through petal fall
2. Space trees to improve air circulation and reduce leaf wetness
3. Remove leaf litter from the orchard floor
4. Consider resistant apple varieties for future plantings
            """
        else:
            return """
General Disease Management:
1. Implement a regular fungicide spray program
2. Practice good orchard sanitation
3. Prune trees for proper air circulation
4. Monitor for early signs of disease
5. Consider weather conditions when planning sprays
            """
    
    def _get_market_fallback(self) -> str:
        """Provide fallback response for market-related questions"""
        counts = self.context_data.get("condition_counts", {})
        total = sum(counts.values()) if counts else 0
        
        if not total:
            return """
Market Value by Apple Quality:
- Premium (Normal): Highest value, suitable for direct retail and export
- Grade A (Minor blemishes): Good for local retail
- Processing Grade (Disease affected): Suitable for juice, sauce, or other processing
            """
        
        # Calculate percentages
        normal_pct = (counts.get("Normal_Apple", 0) / total * 100) if total else 0
        
        if normal_pct > 70:
            return """
Market Recommendation for High-Quality Harvest:
1. Target premium fresh markets for maximum returns
2. Consider export opportunities for high-grade fruit
3. Implement proper storage to extend selling period
4. Explore direct-to-consumer channels for premium pricing
            """
        elif normal_pct > 40:
            return """
Market Recommendation for Mixed-Quality Harvest:
1. Grade and sort fruit by quality
2. Sell premium quality to fresh market
3. Direct lower grades to processing or local markets
4. Consider value-added products for disease-affected fruit
            """
        else:
            return """
Market Recommendation for Processing-Quality Harvest:
1. Contact processing facilities (juice, sauce, dried products)
2. Consider bulk selling to minimize handling costs
3. Explore alternative markets like cider production
4. Implement improved practices for better quality next season
            """
    
    def _get_cultivation_fallback(self) -> str:
        """Provide fallback response for cultivation-related questions"""
        return """
Apple Cultivation Best Practices:

1. Site Selection:
   - Choose sites with good air drainage to reduce frost risk
   - Well-drained soil with pH 6.0-7.0 is ideal
   - Full sun exposure produces better quality fruit

2. Planting:
   - Space trees 15-20 feet apart depending on rootstock
   - Plant in early spring or late fall when dormant
   - Ensure proper root flare positioning at planting

3. Fertilization:
   - Base applications on soil test results
   - Apply nitrogen in spring, avoid late summer applications
   - Consider foliar applications for micronutrients

4. Irrigation:
   - Young trees need 5-10 gallons per week
   - Mature trees require 15-20 gallons during fruit development
   - Drip or micro-sprinkler systems are most efficient
        """

    def _get_harvest_fallback(self) -> str:
        """Provide fallback response for harvest-related questions"""
        return """
Apple Harvesting Guidelines:

1. Timing:
   - Harvest when fruit has reached full color for variety
   - Check starch conversion using iodine test
   - Fruit should separate easily from spur with slight twist

2. Techniques:
   - Handle fruit carefully to avoid bruising
   - Use picking bags to prevent damage
   - Harvest during cool parts of day when possible

3. Post-Harvest:
   - Cool fruit promptly after harvest (32-36°F ideal)
   - Sort by quality immediately
   - Maintain high humidity (90-95%) in storage

4. Quality Indicators:
   - Firmness measured with penetrometer
   - Sugar content (Brix) using refractometer
   - Background color change from green to yellow
        """
    
    def get_chat_history(self) -> List[Dict]:
        """Return the chat history"""
        return self.chat_history
    
    def clear_chat_history(self):
        """Clear the chat history"""
        self.chat_history = []
    
    def analyze_question(self, question: str) -> str:
        """Determine the most appropriate prompt type based on the question"""
        question_lower = question.lower()
        
        if any(word in question_lower for word in ["disease", "infection", "treatment", "symptoms"]):
            return "disease_analysis"
        elif any(word in question_lower for word in ["market", "price", "sell", "storage", "value"]):
            return "market_analysis"
        return "general"

# Helper functions for common chat operations
def format_message_for_display(message: Dict) -> str:
    """Format a chat message for display"""
    role = message["role"].capitalize()
    content = message["content"]
    timestamp = datetime.datetime.fromisoformat(message["timestamp"]).strftime("%H:%M:%S")
    return f"[{timestamp}] {role}: {content}"

def get_disease_recommendations(condition_counts: Dict) -> str:
    """Generate specific recommendations based on disease prevalence"""
    total = sum(condition_counts.values())
    recommendations = []
    
    for condition, count in condition_counts.items():
        percentage = (count / total) * 100 if total > 0 else 0
        if percentage > 10:  # Only give recommendations for significant issues
            if "Blotch" in condition:
                recommendations.append(
                    "Consider applying fungicides and improving air circulation for blotch control."
                )
            elif "Rot" in condition:
                recommendations.append(
                    "Implement better storage practices and handle fruits carefully to prevent rot."
                )
            elif "Scab" in condition:
                recommendations.append(
                    "Review your fungicide program and consider resistant varieties for future plantings."
                )
    
    return "\n".join(recommendations) if recommendations else "No specific disease recommendations needed."
