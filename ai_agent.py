import asyncio
import threading
import logging
import time
from typing import List, Dict
from config import Config
from twitter_client import TwitterClient
from dashboard import log_engagement, log_sentiment, log_api_call
from solana_client import SolanaClientError

logger = logging.getLogger(__name__)

class MainApp:
    def __init__(self):
        self.twitter_client = TwitterClient()
        self.solana_client = None
        self.running = True
        self.loop = None

    async def process_tweets(self, query: str) -> None:
        """Process tweets with enhanced error handling and metrics"""
        try:
            tweets: List[Dict] = self.twitter_client.search_tweets(query)
            logger.info(f"Found {len(tweets)} tweets for processing")
            
            tasks = []
            for tweet in tweets:
                try:
                    if self.twitter_client.is_spam(tweet):
                        logger.info(f"Skipping spam from @{tweet['author']}")
                        log_engagement('spam_detected')
                        continue

                    sentiment = self.twitter_client.analyze_sentiment(tweet['text'])
                    log_sentiment(sentiment)
                    logger.debug(f"Tweet from @{tweet['author']}: Sentiment {sentiment:.2f}")

                    if sentiment >= Config.SENTIMENT_THRESHOLD:
                        tasks.append(
                            self._execute_engagement(tweet)
                        )
                        log_engagement('positive_processed')
                    else:
                        logger.info(f"Skipping negative tweet from @{tweet['author']}")
                        log_engagement('negative_skipped')

                except Exception as e:
                    logger.error(f"Error processing tweet: {str(e)}", exc_info=True)
                    log_api_call('processing', False)

            if tasks:
                await asyncio.gather(*tasks)
                logger.info(f"Completed engagement on {len(tasks)} tweets")

        except Exception as e:
            logger.error(f"Tweet processing failed: {str(e)}", exc_info=True)
            log_api_call('twitter', False)

    async def _execute_engagement(self, tweet: Dict) -> None:
        """Execute engagement with rate limiting"""
        try:
            # Check cooldown before engagement
            if self.twitter_client._action_cooldown(tweet['author']):
                logger.debug(f"Cooldown active for @{tweet['author']}")
                return

            # Run in thread pool executor for blocking IO
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None, 
                self.twitter_client.engage_with_tweet, 
                tweet
            )
            log_engagement('tweet_engaged')
            log_api_call('twitter', True)

        except Exception as e:
            logger.error(f"Engagement failed: {str(e)}", exc_info=True)
            log_api_call('twitter', False)

    async def process_airdrops(self) -> None:
        """Process airdrop claims with retries"""
        try:
            logger.info("Processing airdrop claims...")
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                self.twitter_client.process_airdrop_claims
            )
        except SolanaClientError as e:
            logger.error(f"Airdrop processing failed: {str(e)}")
            log_api_call('solana', False)
        except Exception as e:
            logger.error(f"Unexpected airdrop error: {str(e)}", exc_info=True)

    async def main_loop(self) -> None:
        """Main application loop with error recovery"""
        backoff = 1
        query = f"${Config.MEME_COIN_SYMBOL} OR #{Config.MEME_COIN_SYMBOL}"
        
        while self.running:
            try:
                start_time = time.monotonic()
                
                await self.process_tweets(query)
                await self.process_airdrops()
                
                # Dynamic sleep based on processing time
                elapsed = time.monotonic() - start_time
                sleep_time = max(Config.MAIN_LOOP_INTERVAL - elapsed, 60)
                logger.info(f"Next cycle in {sleep_time/60:.1f} minutes")
                backoff = 1  # Reset backoff on success
                
                await asyncio.sleep(sleep_time)
                
            except KeyboardInterrupt:
                self.running = False
                logger.info("Shutting down gracefully...")
            except Exception as e:
                logger.error(f"Main loop error (retry in {backoff}s): {str(e)}")
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, Config.MAX_BACKOFF)

def start_dashboard():
    """Start monitoring dashboard with error handling"""
    try:
        from dashboard import app, socketio
        socketio.run(app, 
                   host=Config.DASHBOARD_HOST,
                   port=Config.DASHBOARD_PORT,
                   use_reloader=False)
    except Exception as e:
        logger.critical(f"Dashboard failed to start: {str(e)}")
        raise

if __name__ == "__main__":
    # Configure structured logging
    logging.basicConfig(
        level=Config.LOG_LEVEL,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(Config.LOG_FILE)
        ]
    )
    
    # Initialize application
    app = MainApp()
    
    try:
        # Start dashboard thread
        dashboard_thread = threading.Thread(
            target=start_dashboard,
            daemon=True
        )
        dashboard_thread.start()
        
        # Run main async loop
        asyncio.run(app.main_loop())
        
    except Exception as e:
        logger.critical(f"Fatal initialization error: {str(e)}")
    finally:
        logger.info("Application shutdown complete")
