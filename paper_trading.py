#!/usr/bin/env python3
"""
Paper Trading Simulator for the Gauravi advisory system.
Simulates real trading without actual money to validate predictions.

Capital model: opening a position reserves `shares * entry_price` (purchase
cost for a long, 100% margin collateral for a short). Closing releases that
reservation and books the P&L, so `capital` always moves in the same direction
as the trade's P&L regardless of side.

Features:
    - Track open positions
    - Calculate P&L based on actual price movements
    - Generate performance reports (realized trades only)
    - Risk management (position sizing, stop-loss, time-based exit)
"""

import json
import pandas as pd
import yfinance as yf
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import List, Optional


@dataclass
class Position:
    symbol: str
    name: str
    direction: str  # "UP" or "DOWN"
    entry_price: float
    entry_date: str
    confidence: float
    shares: int
    stop_loss: float
    target: float
    status: str = "OPEN"  # OPEN, CLOSED_TP, CLOSED_SL, CLOSED_EXPIRY, CLOSED_MANUAL
    
    def to_dict(self):
        return asdict(self)


class PaperTradingSimulator:
    def __init__(self, initial_capital: float = 1000000.0):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.positions: List[Position] = []
        self.closed_positions: List[Position] = []
        self.trade_history = []
        
        # Risk management
        self.max_position_pct = 0.10  # 10% max per position
        self.stop_loss_pct = 0.03     # 3% stop-loss
        self.target_pct = 0.06        # 6% target (2:1 R:R)
        self.max_holding_days = 10    # force-exit after 10 sessions
        
        # Load existing state
        self.state_file = Path("paper_trading_state.json")
        self.load_state()
    
    def load_state(self):
        """Load simulator state from file"""
        if self.state_file.exists():
            with open(self.state_file) as f:
                state = json.load(f)
                self.initial_capital = state.get("initial_capital", self.initial_capital)
                self.capital = state.get("capital", self.initial_capital)
                self.positions = [Position(**p) for p in state.get("positions", [])]
                self.closed_positions = [Position(**p) for p in state.get("closed_positions", [])]
                self.trade_history = state.get("trade_history", [])

    def save_state(self):
        """Save simulator state to file"""
        state = {
            "initial_capital": self.initial_capital,
            "capital": self.capital,
            "positions": [p.to_dict() for p in self.positions],
            "closed_positions": [p.to_dict() for p in self.closed_positions],
            "trade_history": self.trade_history
        }
        with open(self.state_file, "w") as f:
            json.dump(state, f, indent=2)

    def calculate_position_size(self, confidence: float, current_price: float) -> int:
        """Calculate number of shares based on confidence and risk management"""
        # Kelly-like sizing: higher confidence = larger position
        kelly_fraction = (confidence / 100 - 0.5) * 2  # Maps 50-100% to 0-1
        kelly_fraction = max(0, min(1, kelly_fraction))

        # No edge -> no trade. (Previously a 10k floor forced a position even
        # at zero Kelly fraction, so every signal traded regardless of edge.)
        if kelly_fraction <= 0:
            return 0

        # Position value = Kelly fraction * max position size
        max_position_value = self.capital * self.max_position_pct
        position_value = max_position_value * kelly_fraction

        # Floor, but never above what Kelly allows
        position_value = max(position_value, min(10000, max_position_value))

        # Convert to number of shares
        if current_price > 0:
            shares = int(position_value / current_price)
        else:
            shares = 0

        return shares
    
    def open_position(self, symbol: str, name: str, direction: str, 
                     confidence: float, current_price: float) -> Optional[Position]:
        """Open a new paper trading position"""
        # Check if already have position in this symbol
        if any(p.symbol == symbol and p.status == "OPEN" for p in self.positions):
            print(f"[PAPER] Already have open position in {symbol}")
            return None
        
        # Calculate position size
        shares = self.calculate_position_size(confidence, current_price)
        if shares <= 0:
            return None
        
        # Calculate stop-loss and target
        if direction == "UP":
            stop_loss = current_price * (1 - self.stop_loss_pct)
            target = current_price * (1 + self.target_pct)
        else:  # SHORT
            stop_loss = current_price * (1 + self.stop_loss_pct)
            target = current_price * (1 - self.target_pct)
        
        # Reserve capital. For a long this is the purchase cost; for a short it
        # is the margin collateral (modelled at 100% of notional, conservative).
        # Either way the same amount is released on close and the P&L added
        # separately -- see close_position.
        cost = shares * current_price
        if cost > self.capital:
            print(f"[PAPER] Insufficient capital: {self.capital:.0f} < {cost:.0f}")
            return None

        self.capital -= cost
        
        # Create position
        position = Position(
            symbol=symbol,
            name=name,
            direction=direction,
            entry_price=current_price,
            entry_date=datetime.now().strftime("%Y-%m-%d %H:%M"),
            confidence=confidence,
            shares=shares,
            stop_loss=round(stop_loss, 2),
            target=round(target, 2)
        )
        
        self.positions.append(position)
        self.save_state()
        
        print(f"[PAPER] Opened {direction} position: {name} ({symbol})")
        print(f"  Shares: {shares} @ ₹{current_price:.2f}")
        print(f"  Stop-loss: ₹{stop_loss:.2f} | Target: ₹{target:.2f}")
        
        return position
    
    def check_positions(self):
        """Check all open positions for stop-loss / target hits.

        Walks the daily bars since entry and tests each bar's low/high, so a
        level touched intraday -- or on a day this checker did not run -- is
        still caught. Previously only the latest 1-minute close was sampled,
        which silently missed most SL/TP hits.
        """
        # Iterate over a copy: close_position() removes from self.positions.
        for position in list(self.positions):
            if position.status != "OPEN":
                continue

            try:
                entry_day = pd.to_datetime(position.entry_date).normalize()
                bars = yf.Ticker(position.symbol).history(
                    start=entry_day.strftime("%Y-%m-%d"), interval="1d",
                    auto_adjust=False)

                if bars.empty:
                    continue

                closed = False
                for ts, bar in bars.iterrows():
                    low, high = float(bar["Low"]), float(bar["High"])
                    # Stop-loss takes precedence when a bar spans both levels.
                    if position.direction == "UP":
                        if low <= position.stop_loss:
                            self.close_position(position, position.stop_loss, "CLOSED_SL")
                            closed = True
                        elif high >= position.target:
                            self.close_position(position, position.target, "CLOSED_TP")
                            closed = True
                    else:  # SHORT
                        if high >= position.stop_loss:
                            self.close_position(position, position.stop_loss, "CLOSED_SL")
                            closed = True
                        elif low <= position.target:
                            self.close_position(position, position.target, "CLOSED_TP")
                            closed = True
                    if closed:
                        break

                if closed:
                    continue

                # Time-based exit: don't hold a stale signal forever.
                held = len(bars)
                if held >= self.max_holding_days:
                    self.close_position(position, float(bars["Close"].iloc[-1]),
                                        "CLOSED_EXPIRY")

            except Exception as e:
                print(f"[PAPER] Error checking {position.symbol}: {e}")

    def open_mtm(self) -> float:
        """Unrealized P&L across open positions, at last available close."""
        total = 0.0
        for p in self.positions:
            if p.status != "OPEN":
                continue
            try:
                bars = yf.Ticker(p.symbol).history(period="5d", interval="1d")
                if bars.empty:
                    continue
                last = float(bars["Close"].iloc[-1])
                total += ((last - p.entry_price) if p.direction == "UP"
                          else (p.entry_price - last)) * p.shares
            except Exception:
                continue
        return total
    
    def close_position(self, position: Position, exit_price: float, reason: str):
        """Close a position and calculate P&L"""
        # Calculate P&L
        if position.direction == "UP":
            pnl = (exit_price - position.entry_price) * position.shares
        else:  # SHORT
            pnl = (position.entry_price - exit_price) * position.shares

        # Release the reserved collateral and book the P&L. This is direction-
        # agnostic and always moves capital the same way as pnl. (Previously
        # `capital += shares * exit_price` was used for both directions, which
        # made a *profitable* short REDUCE capital.)
        self.capital += position.shares * position.entry_price + pnl
        
        # Update position
        position.status = reason
        self.closed_positions.append(position)
        self.positions.remove(position)
        
        # Record trade
        exit_dt = datetime.now()
        try:
            hold_days = max(1, (exit_dt - pd.to_datetime(position.entry_date)).days)
        except Exception:
            hold_days = 1
        trade = {
            "symbol": position.symbol,
            "name": position.name,
            "direction": position.direction,
            "entry_price": position.entry_price,
            "exit_price": exit_price,
            "shares": position.shares,
            "pnl": round(pnl, 2),
            "pnl_pct": round(pnl / (position.entry_price * position.shares) * 100, 2),
            "entry_date": position.entry_date,
            "exit_date": exit_dt.strftime("%Y-%m-%d %H:%M"),
            "hold_days": hold_days,
            "reason": reason
        }
        self.trade_history.append(trade)
        
        self.save_state()
        
        emoji = "✅" if pnl > 0 else "❌"
        print(f"\n[PAPER] {emoji} Closed {position.symbol}: {reason}")
        print(f"  Entry: ₹{position.entry_price:.2f} → Exit: ₹{exit_price:.2f}")
        print(f"  P&L: ₹{pnl:.2f} ({trade['pnl_pct']:+.2f}%)")
    
    def get_performance_report(self) -> dict:
        """Generate performance report (realized trades only)."""
        if not self.trade_history:
            return {"message": f"No closed trades yet "
                               f"({len(self.positions)} position(s) still open)",
                    "open_positions": len(self.positions),
                    "current_capital": round(self.capital, 2)}
        
        total_trades = len(self.trade_history)
        winning_trades = len([t for t in self.trade_history if t["pnl"] > 0])
        losing_trades = total_trades - winning_trades
        
        total_pnl = sum(t["pnl"] for t in self.trade_history)
        avg_pnl = total_pnl / total_trades if total_trades > 0 else 0
        
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        # Calculate max drawdown
        equity_curve = [self.initial_capital]
        for trade in self.trade_history:
            equity_curve.append(equity_curve[-1] + trade["pnl"])
        
        peak = equity_curve[0]
        max_drawdown = 0
        for equity in equity_curve:
            peak = max(peak, equity)
            drawdown = (peak - equity) / peak * 100
            max_drawdown = max(max_drawdown, drawdown)
        
        # Sharpe: annualize by the actual trade frequency, not by sqrt(252).
        # Each observation is one trade held ~hold_days, so the number of
        # independent periods per year is 252 / mean(hold_days) -- treating a
        # multi-day trade as a daily observation inflated Sharpe ~3x.
        returns = [t["pnl_pct"] / 100 for t in self.trade_history]
        mean_hold = (sum(t.get("hold_days", 1) for t in self.trade_history)
                     / total_trades) or 1
        periods_per_year = 252 / max(1.0, mean_hold)
        avg_return = sum(returns) / len(returns) if returns else 0
        var = (sum((r - avg_return) ** 2 for r in returns) / (len(returns) - 1)
               if len(returns) > 1 else 0.0)
        std_return = var ** 0.5
        sharpe = ((avg_return * periods_per_year)
                  / (std_return * (periods_per_year ** 0.5))) if std_return > 0 else 0

        return {
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "win_rate": round(win_rate, 2),
            "total_pnl": round(total_pnl, 2),
            "avg_pnl": round(avg_pnl, 2),
            "max_drawdown": round(max_drawdown, 2),
            "mean_hold_days": round(mean_hold, 1),
            "sharpe_ratio": round(sharpe, 2),
            "current_capital": round(self.capital, 2),
            "open_positions": len(self.positions),
            "total_return": round((self.capital / self.initial_capital - 1) * 100, 2)
        }
    
    def print_report(self):
        """Print performance report"""
        report = self.get_performance_report()
        
        print(f"\n{'='*60}")
        print("📊 PAPER TRADING PERFORMANCE REPORT")
        print(f"{'='*60}")
        
        if "message" in report:
            print(report["message"])
            print(f"Capital (cash, collateral excluded): ₹{report['current_capital']:.2f}")
        else:
            print(f"Total Trades: {report['total_trades']}")
            print(f"Win/Loss: {report['winning_trades']}/{report['losing_trades']}")
            print(f"Win Rate: {report['win_rate']:.1f}%")
            print(f"Total P&L: ₹{report['total_pnl']:.2f}")
            print(f"Avg P&L: ₹{report['avg_pnl']:.2f}")
            print(f"Mean Hold: {report['mean_hold_days']:.1f} days")
            print(f"Max Drawdown (realized): {report['max_drawdown']:.1f}%")
            print(f"Sharpe Ratio: {report['sharpe_ratio']:.2f}")
            print(f"Current Capital: ₹{report['current_capital']:.2f}")
            print(f"Open Positions: {report['open_positions']}")
            print(f"Total Return: {report['total_return']:+.1f}%")
        
        print(f"{'='*60}\n")


if __name__ == "__main__":
    # Test the simulator
    sim = PaperTradingSimulator(initial_capital=1000000)
    
    print("Paper Trading Simulator initialized!")
    print(f"Capital: ₹{sim.capital:,.2f}")
    print(f"Open positions: {len(sim.positions)}")
    print(f"Trade history: {len(sim.trade_history)}")
    
    sim.print_report()
