import tweepy
import random
import logging
from textblob import TextBlob
from config import Config
from reply_generator import generate_personalized_reply

logger = logging.getLogger(__name__)

class TwitterClient:
    def __init__(self):
        auth = tweepy.OAuth1UserHandler(
            Config.API_KEY, Config.API_SECRET_KEY,
            Config.ACCESS_TOKEN, Config.ACCESS_TOKEN_SECRET
        )
        self.api = tweepy.API(auth)
        self.engagement_messages = [
            f"Love the energy around {Config.MEME_COIN_SYMBOL}! Keep it up! 🚀",
            f"{Config.MEME_COIN_SYMBOL} to the moon! 🌕",
            f"Great tweet about {Config.MEME_COIN_SYMBOL}! Let's keep the momentum going! 💪",
        ]

    def search_tweets(self, query, count=10):
        try:
            tweets = self.api.search_tweets(q=query, count=count, tweet_mode="extended")
            return tweets
        except Exception as e:
            logger.error(f"Error searching tweets: {e}")
            return []

    def analyze_sentiment(self, tweet_text):
        try:
            analysis = TextBlob(tweet_text)
            return analysis.sentiment.polarity
        except Exception as e:
            logger.error(f"Sentiment analysis error: {e}")
            return 0

    def is_spam(self, tweet_text):
        # Placeholder spam check – integrate your spam model here.
        return False

    def engage_with_tweet(self, tweet):
        try:
            self.api.create_favorite(tweet.id)
            logger.info(f"Liked tweet: {tweet.full_text}")
            self.api.retweet(tweet.id)
            logger.info(f"Retweeted tweet: {tweet.full_text}")

            reply_message = generate_personalized_reply(tweet.full_text)
            self.api.update_status(
                status=reply_message,
                in_reply_to_status_id=tweet.id,
                auto_populate_reply_metadata=True
            )
            logger.info(f"Replied to tweet: {tweet.full_text}")

            # 10% chance to send an airdrop incentive tweet
            if random.random() < 0.1:
                airdrop_msg = f"🎉 You've been selected for a {Config.MEME_COIN_SYMBOL} airdrop! DM us your Solana wallet address to claim your tokens! 🚀"
                self.api.update_status(
                    status=airdrop_msg,
                    in_reply_to_status_id=tweet.id,
                    auto_populate_reply_metadata=True
                )
                logger.info(f"Sent airdrop message to tweet: {tweet.full_text}")
        except Exception as e:
            logger.error(f"Error engaging with tweet: {e}")

    def process_airdrop_claims(self):
        try:
            dms = self.api.get_direct_messages(count=10)
            for dm in dms:
                message_text = dm.message_create["message_data"]["text"].lower()
                if "airdrop" in message_text:
                    wallet_address = message_text.strip()
                    # Offload token sending via Celery task
                    from tasks import send_tokens_async  # Importing task dynamically
                    send_tokens_async.delay(wallet_address, Config.AIRDROP_AMOUNT)
                    self.api.send_direct_message(
                        dm.message_create["sender_id"],
                        "✅ Airdrop successful! Check your wallet."
                    )
        except Exception as e:
            logger.error(f"Error processing airdrop claims: {e}")
