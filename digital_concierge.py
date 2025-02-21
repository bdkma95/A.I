# main.py
import logging
from recommender import ItineraryRecommender
from safety import SafetyAnalyzer
from translation import TranslationSystem
from social import SocialConnector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # 1. Test Itinerary Recommender
    recommender = ItineraryRecommender()
    preferences = [4, 3, 5]  # [adventure, culture, budget]
    recommendations = recommender.recommend(preferences)
    logger.info(f"Recommendations: {recommendations}")
    
    # 2. Test Safety Analyzer
    safety_analyzer = SafetyAnalyzer()
    safety_data = safety_analyzer.get_safety_data(lat=35.6895, lon=139.6917)  # Example: Tokyo
    risk = safety_analyzer.analyze_risk("This city is amazing but sometimes feels unsafe at night.")
    logger.info(f"Safety Data: {safety_data}")
    logger.info(f"Risk Analysis: {risk}")
    
    # 3. Test Translation System
    translator = TranslationSystem()
    translated_text = translator.translate_text("Hello, how are you?", dest_lang='es')
    logger.info(f"Translated Text: {translated_text}")
    # Uncomment to test image translation (ensure a valid image path)
    # image_translation = translator.translate_image("sample_image.jpg")
    # logger.info(f"Image Translation: {image_translation}")
    
    # 4. Test Social Connector
    social_connector = SocialConnector()
    sample_interests = [
        "adventure travel", "cultural experiences", "budget trips",
        "luxury travel", "family travel"
    ]
    matching_result = social_connector.match_travelers(sample_interests)
    logger.info(f"Matching Result: {matching_result}")

if __name__ == "__main__":
    main()
