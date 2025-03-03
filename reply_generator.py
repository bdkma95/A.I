import openai
import random
import logging
import hashlib
from typing import Optional, Dict
from textblob import TextBlob
from cachetools import TTLCache
from config import Config

logger = logging.getLogger(__name__)

class ReplyGenerator:
    def __init__(self):
        self.cache = TTLCache(maxsize=Config.REPLY_CACHE_SIZE, ttl=Config.REPLY_CACHE_TTL)
        self.fallback_responses = {
            'positive': [
                f"🔥 {Config.MEME_COIN_SYMBOL} energy is contagious! Keep spreading the word @{{user}}!",
                f"🚀 @{{user}} knows what's up! #{Config.MEME_COIN_SYMBOL}ToTheMoon"
            ],
            'neutral': [
                f"🤔 Interesting take on {Config.MEME_COIN_SYMBOL}! What's your price prediction @{{user}}?",
                f"💡 @{{user}} Let's make {Config.MEME_COIN_SYMBOL} the next big thing!"
            ],
            'negative': [
                f"😢 Sorry you feel that way @{{user}}! We'll work harder to improve {Config.MEME_COIN_SYMBOL}!",
                f"💪 @{{user}} Challenges make us stronger! #{Config.MEME_COIN_SYMBOL}Army"
            ]
        }
        
    def _generate_cache_key(self, tweet_text: str, username: str) -> str:
        """Create unique cache key using tweet content and author"""
        return hashlib.sha256(f"{username}|{tweet_text}".encode()).hexdigest()

    def _get_sentiment_category(self, polarity: float) -> str:
        """Categorize sentiment score"""
        if polarity > 0.2:
            return 'positive'
        elif polarity < -0.2:
            return 'negative'
        return 'neutral'

    def _generate_llm_reply(self, tweet_text: str, username: str, sentiment: str) -> Optional[str]:
        """Generate reply using GPT-3.5-turbo with sentiment awareness"""
        try:
            system_prompt = f"""You're a social media manager for {Config.MEME_COIN_NAME} ({Config.MEME_COIN_SYMBOL}), 
            a Solana-based meme coin. Respond to tweets in a {sentiment} tone while:
            - Mentioning user @{username}
            - Including relevant hashtags
            - Keeping replies under 280 characters
            - Maintaining meme coin culture
            - Never financial advice"""

            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": tweet_text}
                ],
                max_tokens=70,
                temperature=0.8 if sentiment == 'positive' else 0.5,
                presence_penalty=0.5 if sentiment == 'negative' else 0.2
            )
            
            reply = response.choices[0].message.content.strip()
            return self._sanitize_reply(reply, username)
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return None

    def _sanitize_reply(self, reply: str, username: str) -> str:
        """Ensure reply safety and format"""
        # Remove multiple newlines and special characters
        reply = ' '.join(reply.splitlines()).strip()
        reply = reply.replace('  ', ' ').replace('@ ', '@')
        
        # Ensure username mention
        if f"@{username}" not in reply:
            reply = f"@{username} {reply}"
            
        # Truncate to Twitter limits
        return reply[:275] + "..." if len(reply) > 280 else reply

    def _get_fallback_reply(self, username: str, sentiment: str) -> str:
        """Get cached fallback response with username mention"""
        template = random.choice(self.fallback_responses[sentiment])
        return template.format(user=username)

    def generate_reply(self, tweet_text: str, username: str, tweet_id: str) -> str:
        """Generate or retrieve cached reply with duplicate prevention"""
        cache_key = self._generate_cache_key(tweet_text, username)
        
        # Check cache first
        if cache_key in self.cache:
            logger.info(f"Returning cached reply for {username}")
            return self.cache[cache_key]
            
        # Analyze sentiment
        analysis = TextBlob(tweet_text)
        sentiment = self._get_sentiment_category(analysis.sentiment.polarity)
        
        # Generate LLM response
        llm_reply = self._generate_llm_reply(tweet_text, username, sentiment)
        
        # Fallback if LLM fails
        if not llm_reply or llm_reply in self.cache.values():
            llm_reply = self._get_fallback_reply(username, sentiment)
            
        # Update cache
        self.cache[cache_key] = llm_reply
        return llm_reply
