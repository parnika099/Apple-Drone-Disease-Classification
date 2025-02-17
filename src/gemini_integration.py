import google.generativeai as genai
import json
import os
import datetime
from typing import Dict, List, Optional, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure the Gemini API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
else:
    print("Warning: GEMINI_API_KEY or GOOGLE_API_KEY not found in environment variables.")

# Initialize model variable
model = None

def get_model():
    """Get or initialize the Gemini model"""
    global model
    if model is None and GEMINI_API_KEY:
        try:
            # Use gemini-1.5-flash as it is more cost effective and faster
            model = genai.GenerativeModel('gemini-1.5-flash')
        except Exception as e:
            print(f"Error initializing Gemini model: {e}")
    return model

# Common insights about apple diseases that can be used in responses
COMMON_INSIGHTS = {
    "Normal_Apple": [
        "Normal apples are in good condition and suitable for premium markets.",
        "They typically fetch the highest prices and can be stored for longer periods.",
        "Focus on maintaining current growing practices for these high-quality fruits."
    ],
    "Blotch_Apple": [
        "Apple blotch is caused by the fungus Phyllosticta solitaria.",
        "It appears as irregularly shaped dark spots on the fruit surface.",
        "Consider applying fungicides like captan or strobilurin-based products.",
        "Early detection and treatment is crucial to prevent spread."
    ],
    "Rot_Apple": [
        "Apple rot can be caused by various pathogens, including Botryosphaeria, Colletotrichum, and Penicillium.",
        "Symptoms include soft, brown or black areas that may have a distinct smell.",
        "Control measures include proper orchard sanitation, prompt removal of infected fruit, and fungicide treatments.",
        "Improve air circulation in the orchard to reduce humidity."
    ],
    "Scab_Apple": [
        "Apple scab is caused by the fungus Venturia inaequalis.",
        "It appears as olive-green to black spots on the fruit surface, often with a corky texture.",
        "Preventive fungicide applications early in the growing season are most effective.",
        "Consider resistant varieties for future plantings."
    ],
    "market_trends": [
        "Premium quality apples (Normal) typically command 40-60% higher prices than diseased fruit.",
        "Blotch-affected apples can still be marketed for processing at 60-70% of premium prices.",
        "Rot-affected apples have limited marketability and are often only suitable for juice processing.",
        "Scab-affected apples can be sold at discount markets at 40-50% of premium prices."
    ],
    "management_recommendations": [
        "Implement an integrated pest management (IPM) program that combines cultural, biological, and chemical controls.",
        "Regular pruning to improve air circulation can reduce disease pressure.",
        "Consider weather-based disease forecasting to optimize fungicide applications.",
        "Maintain proper nutrition, especially calcium levels, to increase disease resistance.",
        "Harvest at optimal maturity to reduce postharvest diseases."
    ]
}

# Hindi translations for UI elements
HINDI_TRANSLATIONS = {
    "Apple Harvest Dashboard": "सेब फसल डैशबोर्ड",
    "Harvest Summary": "फसल का सारांश",
    "Condition Analysis": "स्थिति विश्लेषण",
    "Apple Gallery": "सेब गैलरी",
    "Timeline": "समय रेखा",
    "Model Showcase": "मॉडल प्रदर्शन",
    "AI Analysis": "एआई विश्लेषण",
    "Chat": "चैट",
    "Select Harvest Session": "फसल सत्र चुनें",
    "Session Information": "सत्र जानकारी",
    "Harvest Date": "फसल की तारीख",
    "Harvest Time": "फसल का समय",
    "Total Apples Detected": "कुल सेब पहचाने गए",
    "Analyzing apple conditions...": "सेब की स्थिति का विश्लेषण...",
    "Apple Harvest Summary": "सेब फसल का सारांश",
    "Total Apples": "कुल सेब",
    "Normal": "सामान्य",
    "Blotch": "धब्बा",
    "Rot": "सड़न",
    "Scab": "खुरंट",
    "Apple Condition Distribution": "सेब की स्थिति का वितरण",
    "Harvest Recommendations": "फसल की सिफारिशें",
    "Disease Information": "रोग जानकारी",
    "Apple Condition Analysis": "सेब की स्थिति का विश्लेषण",
    "Condition Scores for Each Apple": "प्रत्येक सेब के लिए स्थिति स्कोर",
    "Prediction Confidence": "पूर्वानुमान विश्वास",
    "Prediction Confidence by Apple": "सेब के अनुसार पूर्वानुमान विश्वास",
    "No condition analysis data available for this session.": "इस सत्र के लिए कोई स्थिति विश्लेषण डेटा उपलब्ध नहीं है।",
    "Apple Condition Gallery": "सेब स्थिति गैलरी",
    "Filter by Condition": "स्थिति के अनुसार फ़िल्टर करें",
    "No apple images match the selected filters.": "कोई सेब छवि चयनित फिल्टर से मेल नहीं खाती।",
    "Condition Scores": "स्थिति स्कोर",
    "Failed to load image": "छवि लोड करने में विफल",
    "Harvest Timeline": "फसल समय रेखा",
    "Time": "समय",
    "Apple Count": "सेब की संख्या",
    "Apple Count Over Time": "समय के साथ सेब की संख्या",
    "Session Start": "सत्र प्रारंभ",
    "Session End": "सत्र समाप्त",
    "No timeline data available for this session": "इस सत्र के लिए कोई समय रेखा डेटा उपलब्ध नहीं है",
    "Apple Disease Detection Model": "सेब रोग पहचान मॉडल",
    "About the Model": "मॉडल के बारे में",
    "How the Model Works": "मॉडल कैसे काम करता है",
    "Model Architecture": "मॉडल आर्किटेक्चर",
    "Sample Predictions": "नमूना पूर्वानुमान",
    "No prediction data available yet. Run the detection first to see sample predictions.": "अभी तक कोई पूर्वानुमान डेटा उपलब्ध नहीं है। नमूना पूर्वानुमान देखने के लिए पहले डिटेक्शन चलाएं।",
    "No Normal apple examples in this session": "इस सत्र में कोई सामान्य सेब उदाहरण नहीं है",
    "No Blotch apple examples in this session": "इस सत्र में कोई धब्बा सेब उदाहरण नहीं है",
    "No Rot apple examples in this session": "इस सत्र में कोई सड़न सेब उदाहरण नहीं है",
    "No Scab apple examples in this session": "इस सत्र में कोई खुरंट सेब उदाहरण नहीं है",
    "Technical Details": "तकनीकी विवरण",
    "Key Features for Disease Detection": "रोग पहचान के लिए प्रमुख विशेषताएं",
    "AI Harvest Analysis": "एआई फसल विश्लेषण",
    "Loading AI analysis...": "एआई विश्लेषण लोड हो रहा है...",
    "हिंदी में अनुवाद हो रहा है...": "हिंदी में अनुवाद हो रहा है...",
    "Predicted Market Value Distribution": "अनुमानित बाजार मूल्य वितरण",
    "Seasonal Disease Trend Analysis": "मौसमी रोग प्रवृत्ति विश्लेषण",
    "Chat with Harvest Assistant": "फसल सहायक से चैट करें",
    "Ask a question...": "एक सवाल पूछें...",
    "Thinking...": "सोच रहा हूँ...",
    "Refresh Data": "डेटा रिफ्रेश करें"
}

def analyze_harvest_data(data):
    """
    Generate AI analysis of the harvest data using Gemini
    
    Args:
        data: Dictionary containing harvest data including apple counts and conditions
        
    Returns:
        String containing AI analysis and recommendations
    """
    model = get_model()
    if not model:
        return generate_fallback_analysis(data)

    try:
        # Format the data for the prompt
        total_apples = data.get("total_apples", 0)
        condition_counts = data.get("condition_counts", {})
        condition_percentages = data.get("condition_percentages", {})
        
        # Format timestamp if available
        timestamp = data.get("timestamp", "")
        if timestamp:
            try:
                dt = datetime.datetime.fromisoformat(timestamp)
                formatted_date = dt.strftime("%B %d, %Y")
            except:
                formatted_date = timestamp
        else:
            formatted_date = "Unknown date"
        
        # Create a prompt for Gemini
        prompt = f"""
        You are an expert agricultural advisor specializing in apple cultivation. Analyze the following harvest data and provide valuable insights, recommendations, and predictions for the farmer. Include information about potential causes of any diseases detected, treatment options, market implications, and preventive measures for future harvests.
        
        Harvest Data:
        - Date: {formatted_date}
        - Total Apples: {total_apples}
        - Normal Apples: {condition_counts.get('Normal_Apple', 0)} ({condition_percentages.get('Normal', 0):.1f}%)
        - Blotch Apples: {condition_counts.get('Blotch_Apple', 0)} ({condition_percentages.get('Blotch', 0):.1f}%)
        - Rot Apples: {condition_counts.get('Rot_Apple', 0)} ({condition_percentages.get('Rot', 0):.1f}%)
        - Scab Apples: {condition_counts.get('Scab_Apple', 0)} ({condition_percentages.get('Scab', 0):.1f}%)
        
        Structure your analysis with these sections:
        1. Harvest Overview
        2. Disease Analysis and Causes
        3. Market Implications
        4. Treatment Recommendations
        5. Preventive Measures for Future Harvests
        
        Format your response with clear headings and bullet points where appropriate. Make your advice practical and actionable for a farmer.
        """
        
        # Generate the analysis
        response = model.generate_content(prompt)
        return response.text
        
    except Exception as e:
        print(f"Error in analyze_harvest_data: {e}")
        # Fallback to a template response if Gemini API fails
        return generate_fallback_analysis(data)

def generate_fallback_analysis(data):
    """Generate a fallback analysis if the Gemini API fails"""
    condition_counts = data.get("condition_counts", {})
    condition_percentages = data.get("condition_percentages", {})
    
    # Calculate which condition is most prevalent
    max_condition = max(condition_counts.items(), key=lambda x: x[1]) if condition_counts else ("Unknown", 0)
    
    # Format the analysis based on the condition percentages
    normal_percentage = condition_percentages.get("Normal", 0)
    
    if normal_percentage > 70:
        quality_assessment = "excellent"
        recommendations = [
            "Continue your current orchard management practices",
            "Focus on premium markets for your high-quality produce",
            "Consider implementing cold storage to extend shelf life"
        ]
    elif normal_percentage > 40:
        quality_assessment = "moderate"
        recommendations = [
            "Review your fungicide application schedule",
            "Improve orchard sanitation by removing fallen fruit",
            "Consider pruning to improve air circulation"
        ]
    else:
        quality_assessment = "concerning"
        recommendations = [
            "Consult with a plant pathologist immediately",
            "Review your entire disease management program",
            "Consider testing for soil-borne pathogens"
        ]
    
    # Create the fallback analysis
    analysis = f"""
    # Harvest Overview
    
    Your harvest shows **{quality_assessment} quality** with {normal_percentage:.1f}% of apples in normal condition.
    
    ## Disease Analysis and Causes
    
    The most prevalent condition is **{max_condition[0].replace('_Apple', '')}** affecting {condition_counts.get(max_condition[0], 0)} apples.
    
    {COMMON_INSIGHTS.get(max_condition[0], ["No specific information available for this condition."])[0]}
    
    ## Market Implications
    
    {COMMON_INSIGHTS.get("market_trends", ["Market information not available."])[0]}
    
    ## Treatment Recommendations
    
    {' '.join(COMMON_INSIGHTS.get(max_condition[0], ["No specific recommendations available."])[1:3])}
    
    ## Preventive Measures for Future Harvests
    
    * {recommendations[0]}
    * {recommendations[1]}
    * {recommendations[2]}
    
    {COMMON_INSIGHTS.get("management_recommendations", ["No general recommendations available."])[0]}
    """
    
    return analysis

def ask_question(question, context_data, language="english"):
    """
    Ask a question about the harvest data using Gemini for context-aware answers
    
    Args:
        question: String containing the user's question
        context_data: Dictionary containing the harvest data for context
        language: String indicating the desired response language
        
    Returns:
        String containing the answer to the question
    """
    model = get_model()
    if not model:
        if language.lower() == "hindi":
            return "मुझे क्षमा करें, मैं अभी आपके प्रश्न का उत्तर नहीं दे सकता। कृपया बाद में पुनः प्रयास करें।"
        else:
            return "I'm sorry, I couldn't process your question at the moment. Please try again later."

    try:
        # Format the context data for the prompt
        context_json = json.dumps(context_data)
        
        # Determine language preference for the prompt
        lang_instruction = "Respond in Hindi." if language.lower() == "hindi" else "Respond in English."
        
        # Create a prompt for Gemini
        prompt = f"""
        You are a helpful agricultural AI assistant specializing in apple farming. Answer the following question based on the provided context data. {lang_instruction}
        
        Context Data: {context_json}
        
        User Question: {question}
        
        If the question can't be answered from the context, use your knowledge about apple cultivation, diseases, and best practices to provide a helpful response. Keep your answer concise and practical for farmers.
        """
        
        # Generate the response
        response = model.generate_content(prompt)
        return response.text
        
    except Exception as e:
        print(f"Error in ask_question: {e}")
        # Fallback to a simple response if Gemini API fails
        if language.lower() == "hindi":
            return "मुझे क्षमा करें, मैं अभी आपके प्रश्न का उत्तर नहीं दे सकता। कृपया बाद में पुनः प्रयास करें।"
        else:
            return "I'm sorry, I couldn't process your question at the moment. Please try again later."

def translate_to_hindi(text):
    """
    Translate text to Hindi using Gemini
    
    Args:
        text: String to translate
        
    Returns:
        String containing the Hindi translation
    """
    model = get_model()
    if not model:
        return text + "\n\n(Translation to Hindi failed. Gemini is not connected.)"

    try:
        # Create a prompt for Gemini
        prompt = f"""
        Translate the following English text to Hindi. Maintain the formatting including headings, bullet points, and paragraphs.
        
        Text to translate:
        {text}
        """
        
        # Generate the translation
        response = model.generate_content(prompt)
        return response.text
        
    except Exception as e:
        print(f"Error in translate_to_hindi: {e}")
        # Return original text with error message if translation fails
        return text + "\n\n(Translation to Hindi failed. Showing original text.)"

def classify_apple_image(image_path):
    """
    Classify an apple image using Gemini (from src/apple_detection.py)
    """
    model = get_model()
    if not model:
        return "Gemini model not initialized"

    try:
        import PIL.Image
        sample_file_1 = PIL.Image.open(image_path)
        
        generation_config = {
            "temperature": 1,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 8192,
        }
        
        # Create a new model instance with specific config if needed, or reuse global
        # Reusing global for simplicity and better quota management
        
        prompt = "You are a agriculture expert.Comment on the given image as an expert, keep it short. REPLY IN HINDI"
        
        response = model.generate_content([sample_file_1, prompt])
        return response.text
    except Exception as e:
        return f"Error classifying image: {str(e)}"

def get_fertilizers(disease):
    """
    Get fertilizer recommendations for a disease (from src/apple_detection.py)
    """
    model = get_model()
    if not model:
        return "Gemini model not initialized"
        
    try:
        prompt = f"You are a agriculture expert. Given the disease:{disease} on apple tree suggest some fertilizers. Keep it short and only return fertilizers names"
        response = model.generate_content([prompt])
        return response.text
    except Exception as e:
        return f"Error getting fertilizers: {str(e)}"

def get_qwt(readings):
    """
    Get soil/weather analysis (from src/apple_detection.py)
    """
    model = get_model()
    if not model:
        return "Gemini model not initialized"

    try:
        prompt = f"You are a agriculture expert. Given the soil and temperature conditions:{readings} for an apple tree give some suggestion. Keep it short and descriptive"
        response = model.generate_content([prompt])
        return response.text
    except Exception as e:
        return f"Error getting QWT analysis: {str(e)}"
