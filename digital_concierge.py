# main.py
import logging
import asyncio
from pathlib import Path
from typing import Dict, Any, List, Tuple
from config import config
from itinerary import ItineraryRecommender
from safety import SafetyAnalyzer
from translation import TranslationSystem, TranslationError
from social import SocialConnector, ClusterResult
from exceptions import SafetyAPIError, RecommendationError

# Configure structured logging
logging.basicConfig(
    level=config.LOG_LEVEL,
    format='%(asctime)s - %(module)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

async def main():
    """Main async entry point for the travel assistant application"""
    test_cases: List[Tuple[str, callable]] = [
        ("Recommendation System", test_recommendation_system),
        ("Safety Analysis System", test_safety_system),
        ("Translation System", test_translation_system),
        ("Social Connector System", test_social_system)
    ]

    for system_name, test_func in test_cases:
        try:
            logger.info(f"\n{'='*40}\nStarting {system_name} Tests\n{'='*40}")
            await test_func()
            logger.info(f"\n{'✓'*5} {system_name} Tests Completed Successfully {'✓'*5}")
        except Exception as e:
            logger.error(f"\n{'✗'*5} {system_name} Tests Failed: {str(e)}", exc_info=True)
    
    logger.info("\nAll system tests completed. Exiting application.")

async def test_recommendation_system():
    """Test the enhanced recommendation system with async support"""
    try:
        recommender = ItineraryRecommender()
        preferences = config.DEFAULT_PREFERENCES
        
        logger.info("Generating recommendations with parameters:\n"
                   f"Preferences: {preferences}\n"
                   f"Top K: {config.RECOMMENDATION_TOP_K}\n"
                   f"Max Distance: {config.RECOMMENDATION_MAX_DISTANCE}")
        
        recommendations = await recommender.recommend(
            preferences,
            k=config.RECOMMENDATION_TOP_K,
            max_distance=config.RECOMMENDATION_MAX_DISTANCE
        )
        
        logger.info(f"Generated {len(recommendations)} recommendations:")
        for i, rec in enumerate(recommendations, 1):
            logger.info(
                f"#{i}: {rec['name']}\n"
                f"Adventure: {rec['adventure']} | Culture: {rec['culture']} | Budget: {rec['budget']}\n"
                f"Similarity: {rec['similarity']:.2f}\n{'-'*40}"
            )

    except RecommendationError as e:
        logger.error(f"Recommendation system error: {str(e)}")
    except Exception as e:
        logger.error("Unexpected error in recommendation system")
        raise

async def test_safety_system():
    """Test the safety analysis system with retry logic"""
    try:
        async with SafetyAnalyzer() as safety_analyzer:
            logger.info("Testing safety analysis with Tokyo coordinates")
            safety_data = await safety_analyzer.get_safety_data(
                lat=35.6895,
                lon=139.6917,
                retries=config.SAFETY_API_RETRIES
            )
            logger.info(f"Safety Data Analysis:\n{safety_data.to_formatted_string()}")
            
            logger.info("Running text risk analysis:")
            for text in config.SAFETY_TEST_TEXTS:
                analysis = await safety_analyzer.analyze_risk(text)
                logger.info(
                    f"Text: '{text[:config.TEXT_PREVIEW_LENGTH]}...'\n"
                    f"Risk Level: {analysis.risk_level} | Confidence: {analysis.confidence:.2%}\n"
                    f"Flags: {', '.join(analysis.flags)}"
                )

    except SafetyAPIError as e:
        logger.error(f"Safety API error: {str(e)}")
    except Exception as e:
        logger.error("Critical error in safety analysis system")
        raise

async def test_translation_system():
    """Test the translation system with enhanced error handling"""
    try:
        async with TranslationSystem() as translator:
            logger.info("Testing text translations:")
            for text, lang in config.TRANSLATION_TEST_CASES:
                try:
                    translated = await translator.translate_text(text, lang)
                    logger.info(f"Original: {text}\nTranslated ({lang.upper()}): {translated}\n{'-'*40}")
                except TranslationError as e:
                    logger.warning(f"Translation failed for {lang}: {str(e)}")
                    continue

            logger.info("Testing image translation:")
            if config.TEST_IMAGE_PATH.exists():
                try:
                    translated_image = await translator.translate_image(
                        config.TEST_IMAGE_PATH,
                        config.TARGET_LANGUAGE
                    )
                    logger.info(f"Image translation saved to: {translated_image}")
                except TranslationError as e:
                    logger.error(f"Image translation failed: {str(e)}")
            else:
                logger.warning("Sample image not found, skipping image translation")

    except Exception as e:
        logger.error("Critical error in translation system")
        raise

async def test_social_system():
    """Test the social connector system with dynamic clustering"""
    try:
        async with SocialConnector() as social_connector:
            logger.info(f"Testing traveler matching with {config.SOCIAL_CLUSTERING_ALGORITHM} algorithm")
            
            result = await social_connector.match_travelers(
                interests=config.SOCIAL_TEST_INTERESTS,
                algorithm=config.SOCIAL_CLUSTERING_ALGORITHM,
                **config.SOCIAL_ALGORITHM_PARAMS
            )
            
            logger.info(f"Clustering Results ({result.algorithm.upper()}):\n"
                       f"Clusters Found: {result.cluster_count}\n"
                       f"Silhouette Score: {result.quality_metrics.silhouette_score:.2f}\n"
                       f"Cluster Distribution: {result.cluster_distribution}")
            
            for cluster_id, details in result.cluster_details.items():
                logger.info(
                    f"\nCluster #{cluster_id}:\n"
                    f"Members: {details.member_count}\n"
                    f"Top Interests: {', '.join(details.top_terms)}\n"
                    f"Representative User: {details.representative_user}"
                )

    except Exception as e:
        logger.error(f"Social connection error: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\nApplication interrupted by user. Shutting down gracefully...")
    except Exception as e:
        logger.critical(f"Critical application failure: {str(e)}", exc_info=True)
        raise
