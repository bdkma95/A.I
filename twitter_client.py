import tweepy
import random
import logging
import re
from time import sleep
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from config import Config
from reply_generator import generate_personalized_reply
from typing import List, Dict

logger = logging.getLogger(__name__)

class TwitterClient:
    def __init__(self):
        # Initialize v2 API client with OAuth 2.0
        self.client = tweepy.Client(
            bearer_token=Config.BEARER_TOKEN,
            consumer_key=Config.API_KEY,
            consumer_secret=Config.API_SECRET_KEY,
            access_token=Config.ACCESS_TOKEN,
            access_token_secret=Config.ACCESS_TOKEN_SECRET,
            wait_on_rate_limit=True
        )
        
        # Initialize v1.1 API for endpoints not available in v2
        self.legacy_api = tweepy.API(
            tweepy.OAuth1UserHandler(
                Config.API_KEY,
                Config.API_SECRET_KEY,
                Config.ACCESS_TOKEN,
                Config.ACCESS_TOKEN_SECRET
            ),
            wait_on_rate_limit=True
        )
        
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        self.engagement_messages = self._load_engagement_templates()
        self.airdrop_messages = self._load_airdrop_templates()
        self.spam_keywords = Config.SPAM_KEYWORDS  # Load from config
        self.last_action_time = {}

    def search_tweets(self, query: str, count: int = 10) -> List[Dict]:
        """Search recent tweets using Twitter API v2"""
        try:
            response = self.client.search_recent_tweets(
                query=query,
                max_results=count,
                expansions=["author_id"],
                tweet_fields=["public_metrics", "created_at", "text"]
            )
            return self._format_tweets(response)
        except tweepy.TweepyException as e:
            logger.error(f"Search error: {e}", exc_info=True)
            return []

    def _format_tweets(self, response) -> List[Dict]:
        """Format v2 API response into tweet objects"""
        users = {u.id: u for u in response.includes.get("users", [])}
        return [{
            "id": tweet.id,
            "text": tweet.text,
            "author": users[tweet.author_id].username,
            "created_at": tweet.created_at,
            "retweets": tweet.public_metrics["retweet_count"],
            "likes": tweet.public_metrics["like_count"]
        } for tweet in response.data]

    def analyze_sentiment(self, tweet_text: str) -> float:
        """Analyze sentiment using VADER for social media context"""
        try:
            return self.sentiment_analyzer.polarity_scores(tweet_text)["compound"]
        except Exception as e:
            logger.error(f"Sentiment error: {e}")
            return 0

    def is_spam(self, tweet: Dict) -> bool:
        """Enhanced spam detection"""
        text = tweet["text"].lower()
        user = tweet["author"]
        
        # Check for spam patterns
        if any(kw in text for kw in self.spam_keywords):
            return True
        if re.search(r"http[s]?://", text):  # Multiple links
            return True
        if user.startswith("spam") or user.endswith("bot"):
            return True
            
        return False

    def _action_cooldown(self, user_id: str) -> bool:
        """Prevent rapid consecutive actions on same user"""
        cooldown = Config.ACTION_COOLDOWN  # 300 seconds
        last_action = self.last_action_time.get(user_id, 0)
        return (time.time() - last_action) < cooldown

    def engage_with_tweet(self, tweet: Dict):
        """Smart engagement with cooldown and staggered actions"""
        try:
            if self._action_cooldown(tweet["author"]):
                logger.info(f"Skipping engagement due to cooldown: {tweet['author']}")
                return

            # Randomize action order and delays
            actions = [
                (self._like_tweet, tweet["id"]),
                (self._retweet, tweet["id"]),
                (self._reply_to_tweet, tweet)
            ]
            random.shuffle(actions)
            
            for action, param in actions:
                action(param)
                sleep(random.uniform(1, 3))  # Human-like delays

            # Airdrop with improved template selection
            if random.random() < Config.AIRDROP_PROBABILITY:
                self._send_airdrop(tweet)

            self.last_action_time[tweet["author"]] = time.time()

        except tweepy.TweepyException as e:
            logger.error(f"Engagement error: {e}", exc_info=True)

    def _like_tweet(self, tweet_id: str):
        self.client.like(tweet_id)
        logger.info(f"Liked tweet {tweet_id}")

    def _retweet(self, tweet_id: str):
        self.client.retweet(tweet_id)
        logger.info(f"Retweeted {tweet_id}")

    def _reply_to_tweet(self, tweet: Dict):
        reply = generate_personalized_reply(tweet["text"], tweet["author"])
        self.client.create_tweet(
            text=reply,
            in_reply_to_tweet_id=tweet["id"]
        )
        logger.info(f"Replied to {tweet['id']}")

    def _send_airdrop(self, tweet: Dict):
        template = random.choice(self.airdrop_messages)
        message = template.format(
            user=tweet["author"],
            coin=Config.MEME_COIN_SYMBOL
        )
        self.client.create_tweet(
            text=message,
            in_reply_to_tweet_id=tweet["id"]
        )
        logger.info(f"Sent airdrop to {tweet['author']}")

    def process_airdrop_claims(self):
        """Process DMs with wallet validation"""
        try:
            dms = self.legacy_api.get_direct_messages(count=20)
            for dm in dms:
                self._process_single_dm(dm)
        except tweepy.TweepyException as e:
            logger.error(f"DM processing error: {e}")

    def _process_single_dm(self, dm):
        message = dm.message_create["message_data"]["text"]
        sender_id = dm.message_create["sender_id"]
        
        if not self._validate_wallet_address(message):
            self.legacy_api.send_direct_message(
                sender_id,
                "❌ Invalid Solana address. Please send a valid wallet address."
            )
            return
            
        from tasks import send_tokens_async
        send_tokens_async.delay(message.strip(), Config.AIRDROP_AMOUNT)
        
        self.legacy_api.send_direct_message(
            sender_id,
            "✅ Airdrop received! Tokens will arrive within 24h."
        )
        logger.info(f"Processed airdrop for {sender_id}")

    def _validate_wallet_address(self, address: str) -> bool:
        """Basic Solana address validation"""
        return re.match(r"^[1-9A-HJ-NP-Za-km-z]{32,44}$", address.strip()) is not None

    def _load_engagement_templates(self) -> List[str]:
        """Load dynamic templates from config"""
        return [
            f"🔥 {Config.MEME_COIN_SYMBOL} making waves! #{Config.MEME_COIN_SYMBOL}Mania",
            f"👀 Did someone say {Config.MEME_COIN_SYMBOL}? Let's gooo! 🚀",
            f"💎 Real recognize real! #{Config.MEME_COIN_SYMBOL}Gang"
        ]

    def _load_airdrop_templates(self) -> List[str]:
        return Config.AIRDROP_TEMPLATES
