import tweepy
from textblob import TextBlob
import time
import random
import openai
import os
import logging
from solana.rpc.api import Client
from solana.publickey import PublicKey
from solana.system_program import TransferParams, transfer
from solana.transaction import Transaction
from solana.rpc.commitment import Confirmed
from transformers import pipeline
from flask import Flask, render_template
from flask_socketio import SocketIO
from celery import Celery
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("ai_agent.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Twitter API credentials
API_KEY = os.getenv("API_KEY")
API_SECRET_KEY = os.getenv("API_SECRET_KEY")
ACCESS_TOKEN = os.getenv("ACCESS_TOKEN")
ACCESS_TOKEN_SECRET = os.getenv("ACCESS_TOKEN_SECRET")

# OpenAI API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Solana RPC endpoint
SOLANA_RPC_URL = "https://api.mainnet-beta.solana.com"
client = Client(SOLANA_RPC_URL)

# Meme coin details
MEME_COIN_SYMBOL = "$MEMECOIN"
AIRDROP_MESSAGE = "🎉 You've been selected for a $MEMECOIN airdrop! DM us your Solana wallet address to claim your tokens! 🚀"
AIRDROP_AMOUNT = 1000000  # Amount of tokens to airdrop (in smallest unit, e.g., lamports)

# Authenticate with Twitter API
auth = tweepy.OAuth1UserHandler(API_KEY, API_SECRET_KEY, ACCESS_TOKEN, ACCESS_TOKEN_SECRET)
api = tweepy.API(auth)

# Authenticate with OpenAI
openai.api_key = OPENAI_API_KEY

# List of positive engagement messages
ENGAGEMENT_MESSAGES = [
    f"Love the energy around {MEME_COIN_SYMBOL}! Keep it up! 🚀",
    f"{MEME_COIN_SYMBOL} to the moon! 🌕",
    f"Great tweet about {MEME_COIN_SYMBOL}! Let's keep the momentum going! 💪",
]

# Load pre-trained spam detection model
spam_detector = pipeline("text-classification", model="mrm8488/bert-tiny-finetuned-sms-spam-detection")

# Function to validate Solana wallet addresses
def validate_solana_wallet(address):
    try:
        PublicKey(address)  # This will raise an exception if the address is invalid
        return True
    except Exception as e:
        logger.error(f"Invalid Solana wallet address: {address} - {e}")
        return False

# Function to search for tweets mentioning the meme coin
def search_tweets(query, count=10):
    try:
        tweets = api.search_tweets(q=query, count=count, tweet_mode="extended")
        return tweets
    except Exception as e:
        logger.error(f"Error searching tweets: {e}")
        return []

# Function to analyze tweet sentiment
def analyze_sentiment(tweet_text):
    analysis = TextBlob(tweet_text)
    return analysis.sentiment.polarity  # Returns a value between -1 (negative) and 1 (positive)

# Function to detect spam
def is_spam(tweet_text):
    try:
        result = spam_detector(tweet_text)[0]
        return result["label"] == "spam" and result["score"] > 0.9
    except Exception as e:
        logger.error(f"Error detecting spam: {e}")
        return False

# Function to generate a personalized reply using GPT-3
def generate_personalized_reply(tweet_text):
    try:
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=f"Respond to this tweet in a friendly and engaging way: {tweet_text}",
            max_tokens=50,
            temperature=0.7,
        )
        return response.choices[0].text.strip()
    except Exception as e:
        logger.error(f"Error generating reply with GPT-3: {e}")
        return random.choice(ENGAGEMENT_MESSAGES)

# Function to send tokens via Solana with retry logic
def send_tokens(receiver_wallet, amount, retries=3):
    for attempt in range(retries):
        try:
            sender_wallet = PublicKey(os.getenv("SENDER_WALLET_PUBLIC_KEY"))
            sender_private_key = os.getenv("SENDER_WALLET_PRIVATE_KEY")

            # Check sender balance
            balance = client.get_balance(sender_wallet)
            if balance["result"]["value"] < amount:
                logger.error("Insufficient funds in sender wallet.")
                return False

            txn = Transaction().add(
                transfer(
                    TransferParams(
                        from_pubkey=sender_wallet,
                        to_pubkey=PublicKey(receiver_wallet),
                        lamports=amount,
                    )
                )
            )

            response = client.send_transaction(txn, sender_private_key)
            logger.info(f"Tokens sent to {receiver_wallet}. Transaction ID: {response['result']}")
            return True
        except Exception as e:
            logger.error(f"Attempt {attempt + 1} failed: {e}")
            time.sleep(2)  # Wait before retrying
    return False

# Function to engage with tweets (like, retweet, reply)
def engage_with_tweet(tweet):
    try:
        # Like the tweet
        api.create_favorite(tweet.id)
        logger.info(f"Liked tweet: {tweet.full_text}")

        # Retweet the tweet
        api.retweet(tweet.id)
        logger.info(f"Retweeted tweet: {tweet.full_text}")

        # Generate a personalized reply using GPT-3
        reply_message = generate_personalized_reply(tweet.full_text)
        api.update_status(
            status=reply_message,
            in_reply_to_status_id=tweet.id,
            auto_populate_reply_metadata=True
        )
        logger.info(f"Replied to tweet: {tweet.full_text}")

        # Randomly select a tweet for an airdrop incentive
        if random.random() < 0.1:  # 10% chance of airdrop
            api.update_status(
                status=AIRDROP_MESSAGE,
                in_reply_to_status_id=tweet.id,
                auto_populate_reply_metadata=True
            )
            logger.info(f"Sent airdrop message to tweet: {tweet.full_text}")

    except Exception as e:
        logger.error(f"Error engaging with tweet: {e}")

# Function to handle incoming DMs for airdrop claims
def process_airdrop_claims():
    try:
        dms = api.get_direct_messages(count=10)
        for dm in dms:
            if "airdrop" in dm.message_create["message_data"]["text"].lower():
                wallet_address = dm.message_create["message_data"]["text"].strip()
                if validate_solana_wallet(wallet_address):
                    if send_tokens(wallet_address, AIRDROP_AMOUNT):
                        api.send_direct_message(dm.message_create["sender_id"], "✅ Airdrop successful! Check your wallet.")
                    else:
                        api.send_direct_message(dm.message_create["sender_id"], "❌ Airdrop failed. Please try again.")
    except Exception as e:
        logger.error(f"Error processing airdrop claims: {e}")

# Flask dashboard with real-time updates
app = Flask(__name__)
socketio = SocketIO(app)

@app.route("/")
def dashboard():
    return render_template("dashboard.html")

@socketio.on("connect")
def handle_connect():
    # Emit real-time metrics
    socketio.emit("update_metrics", {"tweets_engaged": 100, "airdrops_sent": 10})

# Celery for asynchronous task handling
celery = Celery("tasks", broker="redis://localhost:6379/0")

@celery.task
def send_tokens_async(receiver_wallet, amount):
    return send_tokens(receiver_wallet, amount)

# Main loop to search and engage with tweets
def main():
    query = MEME_COIN_SYMBOL
    logger.info(f"Starting AI agent for {MEME_COIN_SYMBOL}...")

    while True:
        logger.info("Searching for tweets...")
        tweets = search_tweets(query)

        for tweet in tweets:
            # Skip spam tweets
            if is_spam(tweet.full_text):
                logger.info(f"Skipping spam tweet: {tweet.full_text}")
                continue

            # Analyze sentiment of the tweet
            sentiment = analyze_sentiment(tweet.full_text)
            logger.info(f"Tweet: {tweet.full_text}")
            logger.info(f"Sentiment: {sentiment}")

            # Engage with positive or neutral tweets
            if sentiment >= 0:
                engage_with_tweet(tweet)
            else:
                logger.info("Skipping negative tweet.")

        # Process airdrop claims from DMs
        process_airdrop_claims()

        # Wait for 5 minutes before searching again
        logger.info("Waiting for 5 minutes...")
        time.sleep(300)

if __name__ == "__main__":
    # Start the Flask dashboard in a separate thread
    from threading import Thread
    Thread(target=lambda: socketio.run(app, port=5000)).start()

    # Start the main loop
    main()
