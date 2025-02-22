import os
import base64
import json
from dotenv import load_dotenv
import google.generativeai as genai
from PIL import Image
from io import BytesIO
import typing_extensions as typing
import time

# Load environment variables from .env file
load_dotenv()

# Set the GOOGLE_API_KEY environment variable
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY') or os.getenv('GEMINI_API_KEY')
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)
else:
    print("Warning: GOOGLE_API_KEY not found in .env file.")

class Disease(typing.TypedDict):
    disease_name: str


class Fertilizer(typing.TypedDict):
    fertilizer_name: str


class Recomendation(typing.TypedDict):
    recomendation_name: str


# Define a schema that combines all three types
class ImageAnalysisResponse(typing.TypedDict):
    disease: list[Disease]
    fertilizers: list[Fertilizer]
    recommendations: list[Recomendation]

def get_model():
    try:
        # Choose a Gemini model
        return genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            generation_config={
                "response_mime_type": "application/json",
                "response_schema": ImageAnalysisResponse
            }
        )
    except Exception as e:
        print(f"Error initializing model: {e}")
        return None

class AppleDiseaseDetectorGemini:
    def __init__(self, api_key, images_path):
        """
        Initialize the detector with the Gemini API key and image directory.
        """
        self.api_key = api_key
        self.images_path = images_path
        self.model = get_model()

    def preprocess_image(self, img_path):
        """
        Preprocess the image before sending to the Gemini API.
        """
        try:
            img = Image.open(img_path)
            img = img.convert('RGB')  # Ensure the image is in RGB mode
            img = img.resize((224, 224))  # Resize to fit the model input size
            img_byte_arr = BytesIO()
            img.save(img_byte_arr, format='JPEG')
            return img_byte_arr.getvalue()
        except Exception as e:
            print(f"Error preprocessing image {img_path}: {e}")
            return None

    def use_gemini(self, image_path):
        """
        Send the image to Gemini API for analysis with a 1-second delay between requests.
        """
        if not self.model:
            return [], [], []

        try:
            with open(image_path, "rb") as f:
                image_bytes = f.read()

            # Encode the image in base64
            image_base64 = base64.b64encode(image_bytes).decode("utf-8")

            # Create a structured prompt for the image analysis
            prompt = (
                "Analyze the given image of an apple and provide a comprehensive analysis in JSON format. "
                "The analysis should include the following details:\n\n"
                "1. A list of diseases that may affect the apple in the image, based on its appearance, "
                "including common symptoms that indicate the presence of these diseases.\n\n"
                "2. A list of fertilizers that could potentially help address the nutritional needs of the apple, "
                "based on the condition of the apple shown in the image.\n\n"
                "3. Additional relevant recommendations related to the health of the apple, including any "
                "preventive measures, care tips, or other suggested actions to improve the condition of the apple.\n\n"
                "Please ensure that the disease names, fertilizers, and recommendations are clear, accurate, and concise.\n\n"
                "Recommendations should be very short and crisp."
                "Return the analysis in the following structured JSON format:\n"
                "{\n"
                "  'disease': ['list of diseases'],\n"
                "  'fertilizers': ['list of suitable fertilizers'],\n"
                "  'recommendations': ['other relevant recommendations']\n"
                "}"
            )

            # Generate content using the Gemini model
            response = self.model.generate_content([{"mime_type": "image/jpeg", "data": image_base64}, prompt])
            
            try:
                response_data = json.loads(response.text)
                
                disease_names = [disease["disease_name"] for disease in response_data.get('disease', [])]
                fertilizer_names = [fertilizer["fertilizer_name"] for fertilizer in response_data.get('fertilizers', [])]
                recomendation_names = [recomendation["recomendation_name"] for recomendation in response_data.get('recommendations', [])]

                return disease_names, fertilizer_names, recomendation_names
            except json.JSONDecodeError:
                print(f"Error parsing JSON response: {response.text}")
                return [], [], []

        except Exception as e:
            print(f"Error in API call: {e}")
            return [], [], []

    def check_apples(self):
        """
        Check all apples in the images directory for diseases and return tagged names with a 1-second delay between requests.
        """
        tagged_apples = []

        if not os.path.exists(self.images_path):
            print(f"Image directory not found: {self.images_path}")
            return []

        for folder_name, subfolders, files in os.walk(self.images_path):  # Using os.walk for subfolder traversal
            for img_name in files:
                img_path = os.path.join(folder_name, img_name)

                # Check if the file is an image (jpg, jpeg, png)
                if img_name.lower().endswith(('jpg', 'jpeg', 'png')):
                    print(f"Analyzing {img_name}...")
                    # Preprocess the image and get the image data
                    disease_names, fertilizer_names, recomendation_names = self.use_gemini(img_path)

                    # If the apple has a disease, tag it with its folder and image name
                    if disease_names:  # Assuming disease_names will be non-empty if diseased
                        tagged_apples.append(
                            (folder_name, img_name, disease_names, fertilizer_names, recomendation_names))

                    # Add a 1-second delay to avoid exceeding the quota
                    time.sleep(1)

        return tagged_apples

    def analyze_orchard_health(self, tagged_apples):
        """
        Analyze all the tagged apples and print the optimal steps to improve the entire orchard health.
        """
        if not tagged_apples:
            print("No diseased apples to analyze.")
            return

        all_diseases = {}
        all_fertilizers = {}
        all_recommendations = {}

        for _, _, disease_names, fertilizer_names, recomendation_names in tagged_apples:
            for disease in disease_names:
                all_diseases[disease] = all_diseases.get(disease, 0) + 1
            for fertilizer in fertilizer_names:
                all_fertilizers[fertilizer] = all_fertilizers.get(fertilizer, 0) + 1
            for recommendation in recomendation_names:
                all_recommendations[recommendation] = all_recommendations.get(recommendation, 0) + 1

        # Find the most common disease, fertilizer, and recommendation
        optimal_disease = max(all_diseases, key=all_diseases.get, default=None)
        optimal_fertilizer = max(all_fertilizers, key=all_fertilizers.get, default=None)
        optimal_recommendation = max(all_recommendations, key=all_recommendations.get, default=None)

        # Print the optimal steps for the orchard
        print("\nOptimal Steps for Orchard Health:")

        if optimal_disease:
            print(
                f"Diseases: Focus on disease '{optimal_disease}', which is the most prevalent. Apply the necessary treatment.")

        if optimal_fertilizer:
            print(
                f"Fertilizers: Use fertilizer '{optimal_fertilizer}', which is the most recommended to address the orchard's nutritional needs.")

        if optimal_recommendation:
            print(
                f"Recommendations: Implement recommendation: '{optimal_recommendation}'. This is crucial for improving apple health.")


if __name__ == "__main__":
    # Get Gemini API Key from environment variables
    API_KEY = os.getenv('GOOGLE_API_KEY') or os.getenv('GEMINI_API_KEY')

    if not API_KEY:
        raise ValueError("GOOGLE_API_KEY not found in .env file.")

    # Directory containing the images of apples
    IMAGES_PATH = "downloaded_apples"  # Directory where the images are stored
    
    # Create directory if it doesn't exist for testing
    if not os.path.exists(IMAGES_PATH):
        os.makedirs(IMAGES_PATH, exist_ok=True)
        print(f"Created directory {IMAGES_PATH}. Please add images to analyze.")

    # Initialize the apple disease detector with Gemini API
    detector = AppleDiseaseDetectorGemini(API_KEY, IMAGES_PATH)

    # Step 1: Check apples for disease
    diseased_apples = detector.check_apples()

    # Step 2: Print the tagged diseased apples
    if diseased_apples:
        # Analyze orchard health
        detector.analyze_orchard_health(diseased_apples)
    else:
        print("No diseased apples found.")
