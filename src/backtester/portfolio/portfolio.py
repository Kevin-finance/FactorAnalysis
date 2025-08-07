class Portfolio:
    def __init__(
        self,
        strategies: Union[Strategy, List[Strategy]],
        price_df: pd.DataFrame,
        init_cash: float,
        strategy_weights: List[float] = None,
        commission_per_trade: float = 0.0,
        slippage_pct: float = 0.0,
    ):
        """
        Parameters:
          strategies: a Strategy or list of Strategy instances
          price_df: price DataFrame for all tickers
          init_cash: starting capital
          strategy_weights: optional list of weights summing to 1 for combining strategies equally if multiple
          commission_per_trade: fixed fee per trade
          slippage_pct: slippage as percentage of trade value
        """
        # normalize strategies list
        self.strategies = strategies if isinstance(strategies, list) else [strategies]
        n = len(self.strategies)
        # default equal weights
        if strategy_weights is None:
            self.strategy_weights = [1.0 / n] * n
        else:
            assert len(strategy_weights) == n, "strategy_weights must match number of strategies"
            self.strategy_weights = strategy_weights

        self.price_df = price_df
        self.returns = price_df.pct_change().fillna(0)
        self.init_cash = init_cash
        self.commission = commission_per_trade
        self.slippage = slippage_pct

    def run_backtest(self) -> pd.Series:
        """
        Execute backtest: generate and combine signals from strategies,
        apply portfolio logic, return daily PnL series
        """
        # 1. Generate and combine signals
        combined = pd.DataFrame(index=self.returns.index)
        for strat, w in zip(self.strategies, self.strategy_weights):
            sig = strat.generate_signals(self.price_df)
            aligned = sig.reindex(self.returns.index).fillna(0)
            combined = combined.add(aligned * w, fill_value=0)
        signals = combined  # weights across tickers sum across strategies

        # 2. Shift signals to avoid lookahead
        shifted = signals.shift(1).fillna(0)

        # 3. Compute daily PnL: weight * returns
        pnl = (shifted * self.returns).sum(axis=1)

        # 4. Subtract commission/slippage whenever signal changes
        trades = shifted.diff().abs().sum(axis=1)
        cost = trades * self.commission + trades * self.slippage
        pnl_net = pnl - cost

        return pnl_net