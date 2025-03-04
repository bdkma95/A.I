import asyncio
import threading
import logging
import time
import signal
from typing import List, Dict, Optional
from config import Config
from twitter_client import TwitterClient
from solana_client import SolanaClient, SolanaClientError
from dashboard import log_engagement, log_sentiment, log_api_call, metrics

logger = logging.getLogger(__name__)

class MainApp:
    def __init__(self):
        self.running = True
        self._shutdown_event = asyncio.Event()
        self._setup_signal_handlers()
        
        # Initialize clients with proper resource management
        self.twitter_client = TwitterClient()
        self.solana_client = SolanaClient()

    def _setup_signal_handlers(self):
        """Handle OS signals for graceful shutdown"""
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)

    def _handle_shutdown(self, signum, frame):
        """Trigger graceful shutdown sequence"""
        logger.info(f"Received shutdown signal {signum}")
        self.running = False
        self._shutdown_event.set()

    async def process_tweets(self, query: str) -> None:
        """Process tweets with enhanced error handling and metrics"""
        try:
            tweets: List[Dict] = self.twitter_client.search_tweets(query)
            logger.info(f"Found {len(tweets)} tweets for processing")
            
            tasks = []
            for tweet in tweets:
                try:
                    if self._should_skip_tweet(tweet):
                        continue

                    sentiment = self.twitter_client.analyze_sentiment(tweet['text'])
                    log_sentiment(sentiment)
                    
                    if self._should_engage(sentiment):
                        tasks.append(self._execute_engagement(tweet))
                    else:
                        self._log_negative_tweet(tweet)

                except Exception as e:
                    self._handle_tweet_error(e)

            if tasks:
                await self._process_engagement_tasks(tasks)

        except Exception as e:
            self._handle_processing_error(e)

    def _should_skip_tweet(self, tweet: Dict) -> bool:
        """Determine if tweet should be skipped"""
        if self.twitter_client.is_spam(tweet):
            logger.info(f"Skipping spam from @{tweet['author']}")
            log_engagement('spam_detected')
            return True
        return False

    def _should_engage(self, sentiment: float) -> bool:
        """Determine engagement based on dynamic threshold"""
        return sentiment >= Config.SENTIMENT_THRESHOLD

    def _log_negative_tweet(self, tweet: Dict):
        """Handle negative sentiment tweets"""
        logger.info(f"Skipping negative tweet from @{tweet['author']}")
        log_engagement('negative_skipped')

    async def _process_engagement_tasks(self, tasks: List):
        """Process engagement tasks with monitoring"""
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            success_count = sum(1 for r in results if not isinstance(r, Exception))
            logger.info(f"Completed engagement on {success_count}/{len(tasks)} tweets")
            metrics.metrics['engagement']['success_rate'] = success_count / len(tasks)
        except Exception as e:
            logger.error(f"Task processing failed: {str(e)}", exc_info=True)

    async def _execute_engagement(self, tweet: Dict) -> Optional[Dict]:
        """Execute engagement with circuit breaker pattern"""
        try:
            if self.twitter_client._action_cooldown(tweet['author']):
                logger.debug(f"Cooldown active for @{tweet['author']}")
                return None

            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, 
                self.twitter_client.engage_with_tweet, 
                tweet
            )
            log_engagement('tweet_engaged')
            log_api_call('twitter', True)
            return result
        except Exception as e:
            self._handle_engagement_error(e, tweet)
            raise

    def _handle_tweet_error(self, error: Exception):
        """Handle individual tweet processing errors"""
        logger.error(f"Error processing tweet: {str(error)}", exc_info=True)
        log_api_call('processing', False)

    def _handle_engagement_error(self, error: Exception, tweet: Dict):
        """Handle engagement-specific errors"""
        logger.error(f"Engagement failed for @{tweet['author']}: {str(error)}", exc_info=True)
        log_api_call('twitter', False)
        metrics.metrics['engagement']['errors'] += 1

    def _handle_processing_error(self, error: Exception):
        """Handle top-level processing errors"""
        logger.error(f"Tweet processing failed: {str(error)}", exc_info=True)
        log_api_call('twitter', False)
        metrics.metrics['engagement']['batch_errors'] += 1

    async def process_airdrops(self) -> None:
        """Process airdrop claims with enhanced monitoring"""
        try:
            logger.info("Processing airdrop claims...")
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                self._process_airdrop_claims_safe
            )
        except SolanaClientError as e:
            self._handle_solana_error(e)
        except Exception as e:
            self._handle_general_airdrop_error(e)

    def _process_airdrop_claims_safe(self):
        """Wrapper with additional error handling"""
        try:
            self.twitter_client.process_airdrop_claims()
        except Exception as e:
            logger.error(f"Airdrop processing failed: {str(e)}", exc_info=True)
            raise

    def _handle_solana_error(self, error: SolanaClientError):
        """Handle Solana-specific errors"""
        logger.error(f"Airdrop processing failed: {str(error)}")
        log_api_call('solana', False)
        metrics.metrics['solana']['errors'] += 1

    def _handle_general_airdrop_error(self, error: Exception):
        """Handle unexpected airdrop errors"""
        logger.error(f"Unexpected airdrop error: {str(error)}", exc_info=True)
        metrics.metrics['alerts'].append({
            'type': 'critical',
            'message': f"Airdrop system failure: {str(error)}"
        })

    async def main_loop(self) -> None:
        """Main application loop with circuit breaker pattern"""
        query = f"${Config.MEME_COIN_SYMBOL} OR #{Config.MEME_COIN_SYMBOL}"
        backoff = Config.MIN_BACKOFF
        
        while self.running:
            try:
                cycle_start = time.monotonic()
                
                await self._execute_main_cycle(query)
                
                sleep_time = self._calculate_sleep_time(cycle_start)
                logger.info(f"Next cycle in {sleep_time/60:.1f} minutes")
                backoff = Config.MIN_BACKOFF  # Reset on success
                
                await asyncio.sleep(sleep_time)
                
            except Exception as e:
                backoff = self._handle_main_loop_error(e, backoff)

    async def _execute_main_cycle(self, query: str):
        """Execute main processing cycle"""
        await self.process_tweets(query)
        await self.process_airdrops()
        self._check_system_health()

    def _calculate_sleep_time(self, start_time: float) -> float:
        """Calculate dynamic sleep time with bounds"""
        elapsed = time.monotonic() - start_time
        return max(
            Config.MAIN_LOOP_INTERVAL - elapsed,
            Config.MIN_SLEEP_INTERVAL
        )

    def _check_system_health(self):
        """Perform system health checks"""
        if metrics.metrics['solana'].get('balance', 0) < Config.MIN_SOL_BALANCE:
            logger.warning("Low Solana balance detected")
            
        if metrics.metrics['api_usage'].get('errors', 0) > Config.ERROR_THRESHOLD:
            logger.error("API error threshold exceeded")

    def _handle_main_loop_error(self, error: Exception, backoff: int) -> int:
        """Handle errors in main loop with exponential backoff"""
        logger.error(f"Main loop error (retry in {backoff}s): {str(error)}")
        new_backoff = min(backoff * 2, Config.MAX_BACKOFF)
        return new_backoff

    async def graceful_shutdown(self):
        """Cleanup resources and flush metrics"""
        logger.info("Initiating graceful shutdown...")
        await self._close_clients()
        self._flush_logs()
        logger.info("Shutdown complete")

    async def _close_clients(self):
        """Close all client connections properly"""
        await asyncio.gather(
            self.twitter_client.close(),
            self.solana_client.close()
        )

    def _flush_logs(self):
        """Ensure all logs are flushed"""
        for handler in logging.getLogger().handlers:
            handler.flush()

def start_dashboard():
    """Start monitoring dashboard with health checks"""
    try:
        from dashboard import app, socketio
        app.add_url_rule('/health', 'health', lambda: ('OK', 200))
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
            logging.FileHandler(Config.LOG_FILE),
            logging.handlers.RotatingFileHandler(
                Config.LOG_FILE,
                maxBytes=10*1024*1024,  # 10MB
                backupCount=5
            )
        ]
    )
    
    # Initialize and run application
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
        asyncio.run(app.graceful_shutdown())
        logger.info("Application shutdown complete")
