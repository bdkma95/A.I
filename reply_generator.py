import openai
import random
import logging
from config import Config

logger = logging.getLogger(__name__)

def generate_personalized_reply(tweet_text):
    try:
        openai.api_key = Config.OPENAI_API_KEY
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=f"Respond to this tweet in a friendly and engaging way: {tweet_text}",
            max_tokens=50,
            temperature=0.7,
        )
        return response.choices[0].text.strip()
    except Exception as e:
        logger.error(f"Error generating reply with GPT-3: {e}")
        fallback = random.choice([
            f"Love the energy around {Config.MEME_COIN_SYMBOL}! Keep it up! 🚀",
            f"{Config.MEME_COIN_SYMBOL} to the moon! 🌕",
            f"Great tweet about {Config.MEME_COIN_SYMBOL}! Let's keep the momentum going! 💪",
        ])
        return fallback
