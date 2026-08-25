#!/usr/bin/env python3
"""
Email Alert System for Kronos Trading System
Sends daily predictions and trade signals via email.

Configuration:
    Set environment variables:
    - KRONOS_EMAIL_ADDRESS: Your email address
    - KRONOS_EMAIL_PASSWORD: App password (not regular password)
    - KRONOS_ALERT_RECIPIENT: Recipient email address

Gmail Setup:
    1. Enable 2FA on your Google account
    2. Go to Security > App passwords
    3. Generate an app password for "Mail"
    4. Use that password in KRONOS_EMAIL_PASSWORD
"""

import os
import json
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from pathlib import Path
from datetime import datetime


class EmailAlertSystem:
    def __init__(self):
        self.sender = os.environ.get("KRONOS_EMAIL_ADDRESS", "")
        self.password = os.environ.get("KRONOS_EMAIL_PASSWORD", "")
        self.recipient = os.environ.get("KRONOS_ALERT_RECIPIENT", self.sender)
        
    def is_configured(self):
        """Check if email is properly configured"""
        return bool(self.sender and self.password and self.recipient)
    
    def send_alert(self, subject, body):
        """Send email alert"""
        if not self.is_configured():
            print("[EMAIL] Not configured. Set KRONOS_EMAIL_ADDRESS, KRONOS_EMAIL_PASSWORD, KRONOS_ALERT_RECIPIENT")
            return False
        
        try:
            msg = MIMEMultipart()
            msg['From'] = self.sender
            msg['To'] = self.recipient
            msg['Subject'] = subject
            
            msg.attach(MIMEText(body, 'html'))
            
            with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
                server.login(self.sender, self.password)
                server.send_message(msg)
            
            print(f"[EMAIL] Alert sent to {self.recipient}")
            return True
            
        except Exception as e:
            print(f"[EMAIL] Failed to send: {e}")
            return False
    
    def format_daily_predictions(self, results):
        """Format daily predictions as HTML email"""
        date_str = datetime.now().strftime("%Y-%m-%d")
        
        # Separate tradeable and non-tradeable
        tradeable = [r for r in results if r.get("prediction", {}).get("tradeable", False)]
        non_tradeable = [r for r in results if not r.get("prediction", {}).get("tradeable", False)]
        
        html = f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
                .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 10px; }}
                .section {{ margin: 20px 0; padding: 15px; border-radius: 8px; }}
                .tradeable {{ background-color: #d4edda; border-left: 4px solid #28a745; }}
                .non-tradeable {{ background-color: #f8f9fa; border-left: 4px solid #6c757d; }}
                .signal {{ font-size: 24px; font-weight: bold; }}
                .up {{ color: #28a745; }}
                .down {{ color: #dc3545; }}
                .neutral {{ color: #6c757d; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #f5f5f5; }}
                .footer {{ margin-top: 30px; padding: 15px; background-color: #f5f5f5; border-radius: 8px; font-size: 12px; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>📊 Kronos Daily Predictions</h1>
                <p>Date: {date_str} | Generated: {datetime.now().strftime("%H:%M:%S")}</p>
            </div>
            
            <div class="section tradeable">
                <h2>✅ Tradeable Signals ({len(tradeable)})</h2>
                <table>
                    <tr>
                        <th>Stock</th>
                        <th>Direction</th>
                        <th>Confidence</th>
                        <th>Gate</th>
                    </tr>
        """
        
        for r in tradeable:
            pred = r["prediction"]
            direction = pred["direction"]
            dir_class = "up" if direction == "UP" else "down" if direction == "DOWN" else "neutral"
            dir_symbol = "▲" if direction == "UP" else "▼" if direction == "DOWN" else "—"
            
            html += f"""
                    <tr>
                        <td><strong>{r['name']}</strong><br><small>{r['symbol']}</small></td>
                        <td class="signal {dir_class}">{dir_symbol} {direction}</td>
                        <td>{pred['confidence']}%</td>
                        <td>{pred['gate']}</td>
                    </tr>
            """
        
        html += """
                </table>
            </div>
            
            <div class="section non-tradeable">
                <h2>⏸️ No Trade ({})</h2>
                <table>
                    <tr>
                        <th>Stock</th>
                        <th>Direction</th>
                        <th>Confidence</th>
                        <th>Reason</th>
                    </tr>
        """.format(len(non_tradeable))
        
        for r in non_tradeable:
            pred = r["prediction"]
            direction = pred["direction"]
            dir_class = "up" if direction == "UP" else "down" if direction == "DOWN" else "neutral"
            
            html += f"""
                    <tr>
                        <td>{r['name']}</td>
                        <td class="{dir_class}">{direction}</td>
                        <td>{pred['confidence']}%</td>
                        <td>{pred['gate']}</td>
                    </tr>
            """
        
        html += """
                </table>
            </div>
            
            <div class="footer">
                <p><strong>⚠️ Disclaimer:</strong> This is for educational purposes only. 
                Always do your own research before trading. Past performance does not guarantee future results.</p>
                <p>Generated by Kronos Trading System v1.0</p>
            </div>
        </body>
        </html>
        """
        
        return html
    
    def send_daily_predictions(self, results):
        """Send daily predictions email"""
        date_str = datetime.now().strftime("%Y-%m-%d")
        subject = f"📊 Kronos Daily Predictions - {date_str}"
        body = self.format_daily_predictions(results)
        return self.send_alert(subject, body)
    
    def send_trade_signal(self, symbol, name, direction, confidence, entry_price=None):
        """Send individual trade signal"""
        date_str = datetime.now().strftime("%Y-%m-%d %H:%M")
        direction_symbol = "▲" if direction == "UP" else "▼"
        
        subject = f"🚨 Trade Signal: {direction_symbol} {name} ({symbol})"
        
        html = f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; }}
                .signal {{ font-size: 32px; font-weight: bold; text-align: center; padding: 20px; }}
                .up {{ color: #28a745; background-color: #d4edda; }}
                .down {{ color: #dc3545; background-color: #f8d7da; }}
                .details {{ padding: 15px; background-color: #f5f5f5; border-radius: 8px; }}
            </style>
        </head>
        <body>
            <div class="signal {'up' if direction == 'UP' else 'down'}">
                {direction_symbol} {direction} SIGNAL
            </div>
            <div class="details">
                <h2>{name} ({symbol})</h2>
                <p><strong>Time:</strong> {date_str}</p>
                <p><strong>Confidence:</strong> {confidence}%</p>
                {"<p><strong>Entry Price:</strong> ₹{:.2f}</p>".format(entry_price) if entry_price else ""}
            </div>
        </body>
        </html>
        """
        
        return self.send_alert(subject, html)


# Singleton instance
_alert_system = None

def get_alert_system():
    global _alert_system
    if _alert_system is None:
        _alert_system = EmailAlertSystem()
    return _alert_system


if __name__ == "__main__":
    # Test email configuration
    system = EmailAlertSystem()
    
    if system.is_configured():
        print("Email system configured!")
        print(f"  Sender: {system.sender}")
        print(f"  Recipient: {system.recipient}")
        
        # Send test email
        test_results = [{
            "symbol": "RELIANCE.NS",
            "name": "Reliance Industries",
            "prediction": {
                "direction": "UP",
                "confidence": 75.5,
                "tradeable": True,
                "gate": "TEST SIGNAL"
            }
        }]
        
        confirm = input("Send test email? (y/n): ")
        if confirm.lower() == 'y':
            system.send_daily_predictions(test_results)
    else:
        print("Email system not configured!")
        print("\nSet these environment variables:")
        print("  KRONOS_EMAIL_ADDRESS=your.email@gmail.com")
        print("  KRONOS_EMAIL_PASSWORD=your_app_password")
        print("  KRONOS_ALERT_RECIPIENT=recipient@email.com")
