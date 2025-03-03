import asyncio
import threading
import logging
from twitter_client import TwitterClient

logger = logging.getLogger(__name__)

async def process_tweets(twitter_client, query):
    tweets = twitter_client.search_tweets(query)
    tasks = []
    for tweet in tweets:
        # Skip spam tweets
        if twitter_client.is_spam(tweet.full_text):
            logger.info(f"Skipping spam tweet: {tweet.full_text}")
            continue

        sentiment = twitter_client.analyze_sentiment(tweet.full_text)
        logger.info(f"Tweet: {tweet.full_text}, Sentiment: {sentiment}")

        if sentiment >= 0:
            # Use executor to run blocking engagement calls asynchronously
            loop = asyncio.get_event_loop()
            task = loop.run_in_executor(None, twitter_client.engage_with_tweet, tweet)
            tasks.append(task)
        else:
            logger.info("Skipping negative tweet.")
    if tasks:
        await asyncio.gather(*tasks)

async def main_loop():
    twitter_client = TwitterClient()
    query = Config.MEME_COIN_SYMBOL  # or any other query you prefer
    while True:
        logger.info("Searching for tweets...")
        await process_tweets(twitter_client, query)
        logger.info("Processing airdrop claims...")
        twitter_client.process_airdrop_claims()
        logger.info("Waiting for 5 minutes...")
        await asyncio.sleep(300)

def start_dashboard():
    from dashboard import app, socketio
    socketio.run(app, port=5000)

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()]
    )
    
    # Start the Flask dashboard in a separate thread
    dashboard_thread = threading.Thread(target=start_dashboard)
    dashboard_thread.start()

    # Run the main async loop
    asyncio.run(main_loop())
