import tweepy
from textblob import TextBlob
import time
import random
import openai
import os
from solana.rpc.api import Client
from solana.publickey import PublicKey
from solana.system_program import TransferParams, transfer
from solana.transaction import Transaction
from solana.rpc.commitment import Confirmed
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pandas as pd
from flask import Flask, render_template
import plotly.express as px
import plotly.graph_objects as go

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

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

# Load spam detection model (simple example using Logistic Regression)
def load_spam_model():
    # Example dataset (replace with your own)
    data = {
        "text": [
            "Buy cheap tokens now!",
            "Earn money fast!",
            "I love $MEMECOIN!",
            "This is a great project!",
        ],
        "label": [1, 1, 0, 0],  # 1 = spam, 0 = not spam
    }
    df = pd.DataFrame(data)
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df["text"])
    model = LogisticRegression()
    model.fit(X, df["label"])
    return vectorizer, model

vectorizer, spam_model = load_spam_model()

# Function to validate Solana wallet addresses
def validate_solana_wallet(address):
    try:
        PublicKey(address)  # This will raise an exception if the address is invalid
        return True
    except:
        return False

# Function to search for tweets mentioning the meme coin
def search_tweets(query, count=10):
    try:
        tweets = api.search_tweets(q=query, count=count, tweet_mode="extended")
        return tweets
    except Exception as e:
        print(f"Error searching tweets: {e}")
        return []

# Function to analyze tweet sentiment
def analyze_sentiment(tweet_text):
    analysis = TextBlob(tweet_text)
    return analysis.sentiment.polarity  # Returns a value between -1 (negative) and 1 (positive)

# Function to detect spam
def is_spam(tweet_text):
    X = vectorizer.transform([tweet_text])
    prediction = spam_model.predict(X)
    return prediction[0] == 1  # Returns True if spam

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
        print(f"Error generating reply with GPT-3: {e}")
        return random.choice(ENGAGEMENT_MESSAGES)

# Function to send tokens via Solana
def send_tokens(receiver_wallet, amount):
    try:
        # Replace with your wallet's private key and public key
        sender_wallet = PublicKey(os.getenv("SENDER_WALLET_PUBLIC_KEY"))
        sender_private_key = os.getenv("SENDER_WALLET_PRIVATE_KEY")

        # Create a transfer transaction
        txn = Transaction().add(
            transfer(
                TransferParams(
                    from_pubkey=sender_wallet,
                    to_pubkey=PublicKey(receiver_wallet),
                    lamports=amount,
                )
            )
        )

        # Send the transaction
        response = client.send_transaction(txn, sender_private_key)
        print(f"Tokens sent to {receiver_wallet}. Transaction ID: {response['result']}")
        return True
    except Exception as e:
        print(f"Error sending tokens: {e}")
        return False

# Function to engage with tweets (like, retweet, reply)
def engage_with_tweet(tweet):
    try:
        # Like the tweet
        api.create_favorite(tweet.id)
        print(f"Liked tweet: {tweet.full_text}")

        # Retweet the tweet
        api.retweet(tweet.id)
        print(f"Retweeted tweet: {tweet.full_text}")

        # Generate a personalized reply using GPT-3
        reply_message = generate_personalized_reply(tweet.full_text)
        api.update_status(
            status=reply_message,
            in_reply_to_status_id=tweet.id,
            auto_populate_reply_metadata=True
        )
        print(f"Replied to tweet: {tweet.full_text}")

        # Randomly select a tweet for an airdrop incentive
        if random.random() < 0.1:  # 10% chance of airdrop
            api.update_status(
                status=AIRDROP_MESSAGE,
                in_reply_to_status_id=tweet.id,
                auto_populate_reply_metadata=True
            )
            print(f"Sent airdrop message to tweet: {tweet.full_text}")

    except Exception as e:
        print(f"Error engaging with tweet: {e}")

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
        print(f"Error processing airdrop claims: {e}")

# Flask dashboard
app = Flask(__name__)

@app.route("/")
def dashboard():
    # Example metrics (replace with actual data)
    metrics = {
        "tweets_engaged": 100,
        "airdrops_sent": 10,
        "positive_sentiment": 80,
        "negative_sentiment": 20,
    }
    return render_template("dashboard.html", metrics=metrics)

# Main loop to search and engage with tweets
def main():
    query = MEME_COIN_SYMBOL
    print(f"Starting AI agent for {MEME_COIN_SYMBOL}...")

    while True:
        print("Searching for tweets...")
        tweets = search_tweets(query)

        for tweet in tweets:
            # Skip spam tweets
            if is_spam(tweet.full_text):
                print(f"Skipping spam tweet: {tweet.full_text}")
                continue

            # Analyze sentiment of the tweet
            sentiment = analyze_sentiment(tweet.full_text)
            print(f"Tweet: {tweet.full_text}")
            print(f"Sentiment: {sentiment}")

            # Engage with positive or neutral tweets
            if sentiment >= 0:
                engage_with_tweet(tweet)
            else:
                print("Skipping negative tweet.")

        # Process airdrop claims from DMs
        process_airdrop_claims()

        # Wait for 5 minutes before searching again
        print("Waiting for 5 minutes...")
        time.sleep(300)

if __name__ == "__main__":
    # Start the Flask dashboard in a separate thread
    from threading import Thread
    Thread(target=lambda: app.run(port=5000)).start()

    # Start the main loop
    main()
