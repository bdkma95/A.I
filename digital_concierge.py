# main.py
import logging
import asyncio
from pathlib import Path
from typing import Dict, Any, List, Tuple
from config import get_config
from itinerary import ItineraryRecommender, RecommendationPreferences
from safety import SafetyAnalyzer, SafetyRequest
from translation import TranslationSystem, TranslationRequest
from social import SocialConnector, ClusterRequest
from exceptions import (SafetyAPIError, RecommendationError,
                       TranslationError, ClusteringError)

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(module)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

async def main():
    """Main async entry point with comprehensive error handling"""
    test_cases: List[Tuple[str, callable]] = [
        ("Recommendation System", test_recommendation_system),
        ("Safety Analysis System", test_safety_system),
        ("Translation System", test_translation_system),
        ("Social Connector System", test_social_system)
    ]

    async with get_config() as config:
        for system_name, test_func in test_cases:
            try:
                logger.info(f"\n{'='*40}\n🚀 Starting {system_name} Tests\n{'='*40}")
                await test_func(config)
                logger.info(f"\n✅ {system_name} Tests Completed Successfully")
            except Exception as e:
                logger.error(f"\n❌ {system_name} Tests Failed: {str(e)}", exc_info=True)
                continue
    
        logger.info("\n🌈 All system tests completed. Exiting application.")

async def test_recommendation_system(config: AsyncConfigManager):
    """Test recommendation system with async context management"""
    try:
        async with ItineraryRecommender(config) as recommender:
            prefs = RecommendationPreferences(**config.settings.default_preferences)
            
            logger.info("Generating recommendations with parameters:\n"
                       f"Preferences: {prefs.dict()}\n"
                       f"Top K: {config.settings.recommendation_top_k}\n"
                       f"Max Distance: {config.settings.recommendation_max_distance}")
            
            recommendations = await recommender.recommend(prefs.dict())
            
            logger.info(f"📋 Generated {len(recommendations)} recommendations:")
            for i, rec in enumerate(recommendations, 1):
                logger.info(
                    f"🏅 #{i}: {rec.get('name', 'Unnamed Recommendation')}\n"
                    f"   Adventure: {rec.get('adventure', 'N/A')} | "
                    f"Culture: {rec.get('culture', 'N/A')} | "
                    f"Budget: {rec.get('budget', 'N/A')}\n"
                    f"   Similarity: {rec.get('similarity', 0):.2f}\n"
                    f"{'-'*40}"
                )

    except RecommendationError as e:
        logger.error(f"🚨 Recommendation error: {str(e)}")
    except Exception as e:
        logger.error("💥 Unexpected recommendation system error")
        raise

async def test_safety_system(config: AsyncConfigManager):
    """Test safety analysis with geo-context and text processing"""
    try:
        async with SafetyAnalyzer(config) as analyzer:
            logger.info("📍 Testing safety analysis with Tokyo coordinates")
            safety_request = SafetyRequest(
                latitude=35.6895,
                longitude=139.6917,
                text="Nighttime street lighting is poor in some areas"
            )
            
            result = await analyzer.get_safety_data(safety_request)
            logger.info(f"🛡️ Safety Analysis:\n{result.to_formatted_string()}")
            
            logger.info("🔍 Running text risk analysis:")
            for text in config.settings.safety_test_texts:
                analysis = await analyzer.analyze_risk(text)
                logger.info(
                    f"📜 Text: '{text[:config.settings.text_preview_length]}...'\n"
                    f"   Risk Level: {analysis.risk_level} | "
                    f"Confidence: {analysis.confidence:.2%}\n"
                    f"   Flags: {', '.join(analysis.flags)}"
                )

    except SafetyAPIError as e:
        logger.error(f"🌐 Safety API error: {str(e)}")
    except Exception as e:
        logger.error("💥 Critical safety analysis error")
        raise

async def test_translation_system(config: AsyncConfigManager):
    """Test translation system with multimodal support"""
    try:
        async with TranslationSystem(config) as translator:
            logger.info("🌍 Testing text translations:")
            for text, lang in config.settings.translation_test_cases:
                try:
                    request = TranslationRequest(
                        text=text,
                        target_lang=lang,
                        source_lang='auto'
                    )
                    result = await translator.translate_text(request)
                    logger.info(
                        f"📝 Original: {result.source_text}\n"
                        f"   Translated ({result.target_lang.upper()}): "
                        f"{result.translated_text}\n"
                        f"   Confidence: {result.confidence:.2%}\n{'-'*40}"
                    )
                except TranslationError as e:
                    logger.warning(f"⚠️ Translation failed for {lang}: {str(e)}")
                    continue

            logger.info("🖼️ Testing image translation:")
            if config.settings.test_image_path.exists():
                try:
                    result = await translator.translate_image(
                        config.settings.test_image_path
                    )
                    logger.info(
                        f"📸 Image translation completed\n"
                        f"   Path: {result.translated_text}\n"
                        f"   OCR Confidence: {result.metadata['ocr_confidence']:.2%}"
                    )
                except TranslationError as e:
                    logger.error(f"❌ Image translation failed: {str(e)}")
            else:
                logger.warning("⚠️ Sample image not found, skipping image translation")

    except Exception as e:
        logger.error("💥 Critical translation system error")
        raise

async def test_social_system(config: AsyncConfigManager):
    """Test social connector with dynamic clustering"""
    try:
        async with SocialConnector(config) as connector:
            logger.info(f"🤝 Testing traveler matching with {
                config.settings.social_clustering_algorithm.upper()} algorithm")
            
            request = ClusterRequest(
                interests=config.settings.social_test_interests,
                algorithm=config.settings.social_clustering_algorithm,
                params=config.settings.social_algorithm_params
            )
            
            result = await connector.match_travelers(request)
            
            logger.info(
                f"📊 Clustering Results ({result.algorithm.upper()}):\n"
                f"   Clusters Found: {len(result.cluster_distribution)}\n"
                f"   Silhouette Score: {result.quality_metrics.get('silhouette', 0):.2f}\n"
                f"   Cluster Distribution: {result.cluster_distribution}"
            )
            
            for cluster_id, terms in result.top_terms.items():
                logger.info(
                    f"\n🧩 Cluster #{cluster_id}:\n"
                    f"   Members: {result.cluster_distribution.get(cluster_id, 0)}\n"
                    f"   Top Interests: {', '.join(terms[:5])}"
                )

    except ClusteringError as e:
        logger.error(f"🔀 Clustering error: {str(e)}")
    except Exception as e:
        logger.error("💥 Social connection error")
        raise

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\n🛑 Application interrupted by user. Shutting down gracefully...")
    except Exception as e:
        logger.critical(f"💀 Critical application failure: {str(e)}", exc_info=True)
        raise
