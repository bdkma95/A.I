import tweepy
import random
import logging
import re
import time
from time import sleep
from typing import List, Dict, Optional
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from config import Config
from reply_generator import generate_personalized_reply
from solana_client import SolanaClient

logger = logging.getLogger(__name__)

class TwitterClient:
    def __init__(self):
        self._initialize_clients()
        self._setup_analyzers()
        self._load_configurations()
        self._initialize_state()

    def _initialize_clients(self):
        """Initialize Twitter API clients with proper configuration"""
        try:
            # API v2 Client
            self.client = tweepy.Client(
                bearer_token=Config.BEARER_TOKEN,
                consumer_key=Config.API_KEY,
                consumer_secret=Config.API_SECRET_KEY,
                access_token=Config.ACCESS_TOKEN,
                access_token_secret=Config.ACCESS_TOKEN_SECRET,
                wait_on_rate_limit=True
            )
            
            # Legacy API v1.1 Client
            self.legacy_api = tweepy.API(
                tweepy.OAuth1UserHandler(
                    Config.API_KEY,
                    Config.API_SECRET_KEY,
                    Config.ACCESS_TOKEN,
                    Config.ACCESS_TOKEN_SECRET
                ),
                wait_on_rate_limit=True,
                retry_count=3,
                retry_delay=10
            )
            
        except tweepy.TweepyException as e:
            logger.critical(f"Failed to initialize Twitter clients: {str(e)}")
            raise

    def _setup_analyzers(self):
        """Initialize sentiment analysis components"""
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        self.solana_client = SolanaClient()

    def _load_configurations(self):
        """Load configuration-dependent settings"""
        self.spam_keywords = set(Config.SPAM_KEYWORDS)
        self.engagement_messages = self._safe_load_templates(
            Config.ENGAGEMENT_TEMPLATES,
            default=[
                f"🔥 {Config.MEME_COIN_SYMBOL} making waves!",
                f"🚀 {Config.MEME_COIN_SYMBOL} to the moon!"
            ]
        )
        self.airdrop_messages = self._safe_load_templates(Config.AIRDROP_TEMPLATES)

    def _initialize_state(self):
        """Initialize runtime state tracking"""
        self.last_action_time = {}
        self.user_engagement_counts = defaultdict(int)
        self.consecutive_errors = 0

    def _safe_load_templates(self, templates: List[str], default: List[str] = None) -> List[str]:
        """Safely load templates with fallback"""
        if not templates or not isinstance(templates, list):
            logger.warning("Invalid templates, using defaults")
            return default if default else []
        return templates

    def search_tweets(self, query: str, count: int = 100) -> List[Dict]:
        """Search recent tweets with pagination support"""
        try:
            response = self.client.search_recent_tweets(
                query=query,
                max_results=min(count, 100),
                expansions=["author_id", "referenced_tweets.id"],
                tweet_fields=["public_metrics", "created_at", "text", "context_annotations"],
                user_fields=["public_metrics", "verified", "created_at"]
            )
            return self._format_tweets(response)
        except tweepy.TweepyException as e:
            self._handle_api_error(e, "search_tweets")
            return []

    def _format_tweets(self, response) -> List[Dict]:
        """Enrich tweet data with additional metadata"""
        users = {u.id: u for u in response.includes.get("users", [])}
        tweets = []
        
        for tweet in response.data or []:
            user = users.get(tweet.author_id)
            if not user:
                continue
                
            tweets.append({
                "id": tweet.id,
                "text": tweet.text,
                "author": user.username,
                "author_id": user.id,
                "followers": user.public_metrics.get("followers_count", 0),
                "account_age": (datetime.now() - user.created_at).days,
                "verified": user.verified,
                "created_at": tweet.created_at,
                "retweets": tweet.public_metrics["retweet_count"],
                "likes": tweet.public_metrics["like_count"],
                "spam_score": self._calculate_spam_score(tweet.text, user)
            })
        return tweets

    def _calculate_spam_score(self, text: str, user: tweepy.User) -> float:
        """Calculate comprehensive spam score"""
        score = 0.0
        text_lower = text.lower()
        
        # Keyword matching
        score += sum(1 for kw in self.spam_keywords if kw in text_lower) * 0.2
        
        # URL check
        if len(re.findall(r"http[s]?://", text)) > 1:
            score += 0.3
            
        # User profile checks
        if user.followers_count < Config.MIN_FOLLOWERS:
            score += 0.2
        if user.created_at > datetime.now() - timedelta(days=7):
            score += 0.3
            
        return min(score, 1.0)

    def is_spam(self, tweet: Dict) -> bool:
        """Determine if tweet is spam using multiple factors"""
        return tweet.get("spam_score", 0) >= Config.SPAM_THRESHOLD

    def _action_cooldown(self, user_id: str) -> bool:
        """Dynamic cooldown based on engagement frequency"""
        now = time.time()
        last_action = self.last_action_time.get(user_id, 0)
        base_cooldown = Config.ACTION_COOLDOWN
        
        # Increase cooldown for frequent engagements
        engagement_count = self.user_engagement_counts[user_id]
        if engagement_count > Config.MAX_ENGAGEMENTS_PER_USER:
            base_cooldown *= 2
            
        return (now - last_action) < base_cooldown

    def engage_with_tweet(self, tweet: Dict):
        """Execute engagement strategy with enhanced safety checks"""
        try:
            if self._should_skip_engagement(tweet):
                return

            actions = self._prepare_engagement_actions(tweet)
            self._execute_engagement_actions(actions)
            self._handle_airdrop_opportunity(tweet)
            
            self._update_engagement_state(tweet)
            self.consecutive_errors = 0

        except tweepy.TweepyException as e:
            self._handle_api_error(e, "engage_with_tweet")
            self.consecutive_errors += 1
            if self.consecutive_errors > Config.MAX_CONSECUTIVE_ERRORS:
                raise RuntimeError("Too many consecutive errors") from e

    def _should_skip_engagement(self, tweet: Dict) -> bool:
        """Check multiple skip conditions"""
        return (
            self._action_cooldown(tweet["author_id"]) or
            self.is_spam(tweet) or
            tweet["author"] in Config.BLOCKED_USERS
        )

    def _prepare_engagement_actions(self, tweet: Dict) -> List:
        """Prepare engagement actions with randomization"""
        actions = []
        if Config.ENABLE_LIKES:
            actions.append((self._like_tweet, tweet["id"]))
        if Config.ENABLE_RETWEETS:
            actions.append((self._retweet, tweet["id"]))
        if Config.ENABLE_REPLIES:
            actions.append((self._reply_to_tweet, tweet))
            
        random.shuffle(actions)
        return actions

    def _execute_engagement_actions(self, actions: List):
        """Execute actions with proper delays and error handling"""
        for action, param in actions:
            try:
                action(param)
                sleep(random.uniform(1, 3))
            except tweepy.TweepyException as e:
                logger.warning(f"Partial engagement failure: {str(e)}")
                log_api_call('twitter', False)

    def _handle_airdrop_opportunity(self, tweet: Dict):
        """Handle airdrop distribution with probability check"""
        if (
            random.random() < Config.AIRDROP_PROBABILITY and
            self.solana_client.get_balance() > Config.AIRDROP_AMOUNT * 2
        ):
            self._send_airdrop(tweet)

    def _update_engagement_state(self, tweet: Dict):
        """Update engagement tracking state"""
        self.last_action_time[tweet["author_id"]] = time.time()
        self.user_engagement_counts[tweet["author_id"]] += 1

    def _like_tweet(self, tweet_id: str):
        self.client.like(tweet_id)
        log_engagement('like')

    def _retweet(self, tweet_id: str):
        self.client.retweet(tweet_id)
        log_engagement('retweet')

    def _reply_to_tweet(self, tweet: Dict):
        reply = generate_personalized_reply(tweet["text"], tweet["author"])
        self.client.create_tweet(
            text=reply,
            in_reply_to_tweet_id=tweet["id"]
        )
        log_engagement('reply')

    def _send_airdrop(self, tweet: Dict):
        """Send airdrop with validation and logging"""
        try:
            template = random.choice(self.airdrop_messages)
            message = template.format(
                user=tweet["author"],
                coin=Config.MEME_COIN_SYMBOL
            )
            self.client.create_tweet(
                text=message,
                in_reply_to_tweet_id=tweet["id"]
            )
            log_engagement('airdrop_sent')
        except IndexError:
            logger.error("No airdrop templates available")
        except tweepy.TweepyException as e:
            self._handle_api_error(e, "send_airdrop")

    def process_airdrop_claims(self):
        """Process DMs with enhanced validation and error handling"""
        try:
            dms = self.legacy_api.get_direct_messages(count=20)
            for dm in dms:
                self._process_dm_safely(dm)
        except tweepy.TweepyException as e:
            self._handle_api_error(e, "process_airdrop_claims")

    def _process_dm_safely(self, dm):
        """Process single DM with error isolation"""
        try:
            sender_id = dm.message_create["sender_id"]
            message = dm.message_create["message_data"]["text"]
            
            if not self._validate_dm_content(message):
                return
                
            self._handle_valid_dm(sender_id, message)
            
        except Exception as e:
            logger.error(f"Failed to process DM: {str(e)}")
            log_api_call('twitter', False)

    def _validate_dm_content(self, message: str) -> bool:
        """Validate DM content before processing"""
        return (
            self._validate_wallet_address(message) and
            not self._is_duplicate_request(message)
        )

    def _validate_wallet_address(self, address: str) -> bool:
        """Validate Solana address through network check"""
        try:
            return self.solana_client.validate_address(address.strip())
        except SolanaClientError as e:
            logger.warning(f"Address validation failed: {str(e)}")
            return False

    def _is_duplicate_request(self, message: str) -> bool:
        """Check for duplicate airdrop requests"""
        # Implement duplicate checking logic here
        return False

    def _handle_valid_dm(self, sender_id: str, message: str):
        """Process valid airdrop request"""
        from tasks import send_tokens_async
        
        send_tokens_async.delay(
            message.strip(), 
            Config.AIRDROP_AMOUNT,
            sender_id=sender_id
        )
        
        self.legacy_api.send_direct_message(
            sender_id,
            "✅ Airdrop received! Tokens will arrive within 24h."
        )
        log_engagement('airdrop_processed')

    def _handle_api_error(self, error: tweepy.TweepyException, context: str):
        """Handle API errors consistently"""
        logger.error(f"Twitter API error in {context}: {str(error)}")
        log_api_call('twitter', False)
        
        if isinstance(error, tweepy.TooManyRequests):
            sleep_time = error.reset_in + 10 if error.reset_in else 900
            logger.warning(f"Rate limited, sleeping for {sleep_time}s")
            sleep(sleep_time)
