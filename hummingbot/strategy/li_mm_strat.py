from hummingbot.strategy.strategy_base import StrategyBase
from hummingbot.core.data_type.order_book import OrderBook
from hummingbot.core.event.events import MarketEvent
from hummingbot.connector.connector_base import ConnectorBase
from decimal import Decimal
import asyncio
import numpy as np
import math

class LiMarketMakingStrategy(StrategyBase):
    def __init__(
        self,
        market: ConnectorBase,
        trading_pair: str,
        tick_size: Decimal = Decimal("1.0"),
        obp_window: int = 15,
        depth_levels: int = 20,
        order_refresh_ticks: int = 5,
        eta: Decimal = Decimal("0.5"),
        mu: Decimal = Decimal("1.0"),
        inventory_target: Decimal = Decimal("0.5"),
    ):
        super().__init__()
        self._market = market
        self._trading_pair = trading_pair
        self._tick_size = tick_size
        self._obp_window = obp_window
        self._depth_levels = depth_levels
        self._order_refresh_ticks = order_refresh_ticks
        self._eta = eta
        self._mu = mu
        self._inventory_target = inventory_target

        self._ticks = 0
        self._order_book_snapshots = []

    @property
    def market(self):
        return self._market

    def on_tick(self):
        self._ticks += 1

        # Record current order book snapshot
        self._record_order_book()

        if self._ticks % self._order_refresh_ticks == 0:
            self.cancel_all_orders()
            self._place_orders()

    def _record_order_book(self):
        ob: OrderBook = self._market.get_order_book(self._trading_pair)
        bids = ob.snapshot["bids"][:self._depth_levels]
        asks = ob.snapshot["asks"][:self._depth_levels]
        self._order_book_snapshots.append((bids, asks))
        if len(self._order_book_snapshots) > self._obp_window:
            self._order_book_snapshots.pop(0)

    def _calculate_obp(self):
        bid_sum, ask_sum = 0, 0
        for bids, asks in self._order_book_snapshots:
            bid_sum += sum([b[1] for b in bids])
            ask_sum += sum([a[1] for a in asks])
        obp = (bid_sum / ask_sum) if ask_sum > 0 else 1
        return obp

    def _get_inventory_ratio(self):
        base, quote = self._market.get_balance(self._trading_pair)
        price = self._market.get_price(self._trading_pair)
        total_value = base * price + quote
        target_base_value = total_value * self._inventory_target
        current_base_value = base * price
        return (current_base_value - target_base_value) / total_value

    def _place_orders(self):
        price = self._market.get_price(self._trading_pair)
        obp = self._calculate_obp()
        sign_obp = np.sign(obp - 1)  # >1 means pressure on buy side
        sign_ns = np.sign(self._get_inventory_ratio())

        skew = (sign_obp * float(self._mu) + sign_ns * float(self._eta)) * float(self._tick_size)

        bid_price = price - skew
        ask_price = price + skew

        bid_price = self._market.quantize_order_price(self._trading_pair, Decimal(bid_price))
        ask_price = self._market.quantize_order_price(self._trading_pair, Decimal(ask_price))

        amount = self._market.get_order_size(self._trading_pair, Decimal("0.001"))  # Replace with desired size logic

        self.place_order(self._market, self._trading_pair, True, bid_price, amount)
        self.place_order(self._market, self._trading_pair, False, ask_price, amount)

    def place_order(self, market, trading_pair, is_buy, price, amount):
        if is_buy:
            market.buy(trading_pair, amount, order_type="limit", price=price)
        else:
            market.sell(trading_pair, amount, order_type="limit", price=price)

    def cancel_all_orders(self):
        self._market.cancel_all_orders(self._trading_pair)

    def format_status(self):
        return f"Tick: {self._ticks} | OBP: {self._calculate_obp():.3f}"
