#!/usr/bin/env python3
"""
Telegram Alert System for the Gauravi advisory system.
Sends daily predictions and trade signals via Telegram.

Setup:
    1. Create a Telegram bot via @BotFather
    2. Get your chat ID by messaging @userinfobot
    3. Set environment variables:
       - GAURAVI_TELEGRAM_BOT_TOKEN: Your bot token from BotFather
       - GAURAVI_TELEGRAM_CHAT_ID: Your chat ID

    The legacy KRONOS_* names are still accepted as a fallback.
"""

import os
import json
import requests
from datetime import datetime


def env(suffix, default=""):
    """Read GAURAVI_<suffix>, falling back to the legacy KRONOS_<suffix>."""
    return (os.environ.get(f"GAURAVI_{suffix}")
            or os.environ.get(f"KRONOS_{suffix}")
            or default)


class TelegramAlertSystem:
    def __init__(self):
        self.bot_token = env("TELEGRAM_BOT_TOKEN")
        self.chat_id = env("TELEGRAM_CHAT_ID")
        self.base_url = f"https://api.telegram.org/bot{self.bot_token}"

    def is_configured(self):
        """Check if Telegram is properly configured"""
        return bool(self.bot_token and self.chat_id)

    def send_message(self, text, parse_mode="HTML"):
        """Send message via Telegram"""
        if not self.is_configured():
            print("[TELEGRAM] Not configured. Set GAURAVI_TELEGRAM_BOT_TOKEN "
                  "and GAURAVI_TELEGRAM_CHAT_ID")
            return False
        
        try:
            url = f"{self.base_url}/sendMessage"
            payload = {
                "chat_id": self.chat_id,
                "text": text,
                "parse_mode": parse_mode
            }
            
            response = requests.post(url, json=payload, timeout=10)
            
            if response.status_code == 200:
                print("[TELEGRAM] Message sent successfully")
                return True
            else:
                print(f"[TELEGRAM] Failed: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            print(f"[TELEGRAM] Error: {e}")
            return False
    
    def format_daily_predictions(self, results):
        """Format daily predictions as Telegram message"""
        date_str = datetime.now().strftime("%Y-%m-%d")
        
        tradeable = [r for r in results if r.get("prediction", {}).get("tradeable", False)]
        
        if not tradeable:
            return f"📊 <b>Kronos Daily Report - {date_str}</b>\n\n❌ No tradeable signals today."
        
        message = f"📊 <b>Kronos Daily Report - {date_str}</b>\n\n"
        message += f"✅ <b>Tradeable Signals: {len(tradeable)}</b>\n\n"
        
        for r in tradeable:
            pred = r["prediction"]
            direction = pred["direction"]
            emoji = "🟢" if direction == "UP" else "🔴"
            
            message += f"{emoji} <b>{r['name']}</b> ({r['symbol']})\n"
            message += f"   Direction: {direction}\n"
            message += f"   Confidence: {pred['confidence']}%\n"
            message += f"   Gate: {pred['gate']}\n\n"
        
        message += "⚠️ <i>For educational purposes only. Do your own research.</i>"
        
        return message
    
    def send_daily_predictions(self, results):
        """Send daily predictions via Telegram"""
        message = self.format_daily_predictions(results)
        return self.send_message(message)
    
    def send_trade_signal(self, symbol, name, direction, confidence, entry_price=None):
        """Send individual trade signal"""
        emoji = "🟢" if direction == "UP" else "🔴"
        
        message = f"🚨 <b>TRADE SIGNAL</b> 🚨\n\n"
        message += f"{emoji} <b>{name}</b> ({symbol})\n\n"
        message += f"Direction: <b>{direction}</b>\n"
        message += f"Confidence: <b>{confidence}%</b>\n"
        
        if entry_price:
            message += f"Entry: ₹{entry_price:.2f}\n"
        
        message += f"\n⏰ {datetime.now().strftime('%H:%M:%S')}"
        
        return self.send_message(message)
    
    def send_error_alert(self, error_message):
        """Send error alert"""
        message = f"⚠️ <b>Kronos Error Alert</b>\n\n"
        message += f"Error: {error_message}\n"
        message += f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        return self.send_message(message)


# Singleton instance
_telegram_system = None

def get_telegram_system():
    global _telegram_system
    if _telegram_system is None:
        _telegram_system = TelegramAlertSystem()
    return _telegram_system


if __name__ == "__main__":
    # Test Telegram configuration
    system = TelegramAlertSystem()
    
    if system.is_configured():
        print("Telegram system configured!")
        print(f"  Bot Token: ...{system.bot_token[-8:]}")
        print(f"  Chat ID: {system.chat_id}")
        
        # Send test message
        confirm = input("Send test message? (y/n): ")
        if confirm.lower() == 'y':
            system.send_message("🧪 Test message from Kronos Trading System!")
    else:
        print("Telegram system not configured!")
        print("\nSet these environment variables:")
        print("  GAURAVI_TELEGRAM_BOT_TOKEN=your_bot_token")
        print("  GAURAVI_TELEGRAM_CHAT_ID=your_chat_id")
        print("\nSetup steps:")
        print("1. Message @BotFather on Telegram to create a bot")
        print("2. Message @userinfobot to get your chat ID")
