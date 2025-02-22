# main.py
import logging
import asyncio
from pathlib import Path
from typing import Dict, Any
from config import config
from recommender import ItineraryRecommender
from safety import SafetyAnalyzer
from translation import TranslationSystem, TranslationError
from social import SocialConnector, ClusterResult
from exceptions import SafetyAPIError, RecommendationError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

async def main():
    """Main async entry point for the travel assistant application"""
    try:
        # 1. Test Enhanced Itinerary Recommender
        await test_recommendation_system()

        # 2. Test Safety Analysis with Async Support
        await test_safety_system()

        # 3. Test Translation System with Caching
        await test_translation_system()

        # 4. Test Social Connector with Advanced Clustering
        await test_social_system()

    except Exception as e:
        logger.error(f"Application failed: {str(e)}", exc_info=True)
        raise

async def test_recommendation_system():
    """Test the enhanced recommendation system"""
    try:
        logger.info("Testing Recommendation System...")
        
        recommender = ItineraryRecommender()
        preferences = {'adventure': 5, 'culture': 3, 'budget': 4}
        
        # Test with different parameters
        recommendations = recommender.recommend(
            preferences,
            k=5,
            max_distance=0.4
        )
        
        logger.info(f"Top Recommendations (n={len(recommendations)}):")
        for i, rec in enumerate(recommendations, 1):
            logger.info(
                f"#{i}: Adventure: {rec['adventure']} "
                f"Culture: {rec['culture']} Budget: {rec['budget']} "
                f"(Similarity: {rec['similarity']:.2f})"
            )

    except RecommendationError as e:
        logger.error(f"Recommendation system error: {str(e)}")
    except Exception as e:
        logger.error("Unexpected recommendation error occurred")
        raise

async def test_safety_system():
    """Test the enhanced safety analysis system"""
    try:
        logger.info("\nTesting Safety Analysis System...")
        
        safety_analyzer = SafetyAnalyzer()
        
        # Test with Tokyo coordinates
        safety_data = await safety_analyzer.get_safety_data(35.6895, 139.6917)
        logger.info(f"Safety Data: {safety_data}")
        
        # Analyze risk with different text samples
        texts = [
            "This area is perfectly safe during daylight hours",
            "Avoid this neighborhood after dark, frequent incidents reported",
            "Mixed reviews about safety, generally okay with precautions"
        ]
        
        for text in texts:
            analysis = safety_analyzer.analyze_risk(text)
            logger.info(
                f"Risk Analysis: '{text[:30]}...' => "
                f"{analysis.risk_level} (Confidence: {analysis.confidence:.2%})"
            )

    except SafetyAPIError as e:
        logger.error(f"Safety API error: {str(e)}")
    except Exception as e:
        logger.error("Unexpected safety analysis error")
        raise

async def test_translation_system():
    """Test the enhanced translation system"""
    try:
        logger.info("\nTesting Translation System...")
        
        translator = TranslationSystem()
        
        # Text translation with caching
        texts = [
            ("Hello, where is the nearest hospital?", 'es'),
            ("Emergency exit on the right side", 'fr'),
            ("Vegetarian food options available", 'de')
        ]
        
        for text, lang in texts:
            translated = translator.translate_text(text, lang)
            logger.info(f"Translated ({lang}): {translated}")
        
        # Image translation with preprocessing
        image_path = Path("sample_sign.jpg")
        if image_path.exists():
            translated_image = translator.translate_image(image_path, 'es')
            logger.info(f"Image Translation: {translated_image}")
        else:
            logger.warning("Sample image not found, skipping image translation")

    except TranslationError as e:
        logger.error(f"Translation error: {str(e)}")
    except Exception as e:
        logger.error("Unexpected translation error")
        raise

async def test_social_system():
    """Test the enhanced social connector system"""
    try:
        logger.info("\nTesting Social Connector System...")
        
        social_connector = SocialConnector()
        
        interests = [
            "hiking mountain trails",
            "visiting art museums",
            "budget backpacking trips",
            "luxury hotel stays",
            "family-friendly activities"
        ]
        
        # Test different clustering algorithms
        for algorithm in ['kmeans', 'dbscan']:
            result = social_connector.match_travelers(
                interests,
                algorithm=algorithm,
                n_clusters=2 if algorithm == 'kmeans' else None
            )
            
            logger.info(f"\nClustering with {algorithm.upper()}:")
            logger.info(f"Cluster Distribution: {result['cluster_counts']}")
            logger.info(f"Silhouette Score: {result.get('silhouette_score', 'N/A')}")
            
            for cluster_id, terms in result['top_terms'].items():
                logger.info(f"Cluster {cluster_id} Top Terms: {', '.join(terms)}")

    except Exception as e:
        logger.error(f"Social connection error: {str(e)}")
        raise
    finally:
        await social_connector.close()

if __name__ == "__main__":
    asyncio.run(main())
