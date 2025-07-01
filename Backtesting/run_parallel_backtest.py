import sys, os
sys.path.append(os.path.abspath('.'))

import multiprocessing as mp
from tqdm import tqdm
import time
from Backtesting import DSPBacktester

def compute_weights_all_args(args):
    date_idx, tickers, J, df, rebalance_every, decay = args
    bt = DSPBacktester(tickers=tickers, J=J, df=df,
                       rebalance_every=rebalance_every, decay=decay)
    return bt.compute_weights_on_date(date_idx)

def run_parallel_backtest(tickers, J, df, rebalance_every, decay, start_idx, end_idx):
    date_indices = list(range(start_idx, end_idx, rebalance_every))
    args_list = [(date_idx, tickers, J, df, rebalance_every, decay) for date_idx in date_indices]

    num_procs = max(mp.cpu_count() - 2, 1)
    print(f"Launching parallel backtest using {num_procs} processes")
    start_time = time.time()

    results = []
    with mp.Pool(processes=num_procs) as pool:
        with tqdm(total=len(args_list)) as pbar:
            for res in pool.imap(compute_weights_all_args, args_list):
                results.append(res)
                pbar.update()

    elapsed = time.time() - start_time
    print(f"Finished backtest in {elapsed:.2f} seconds")

    bt_final = DSPBacktester(tickers=tickers, J=J, df=df,
                             rebalance_every=rebalance_every, decay=decay)
    bt_final.store_backtest_results(results, start_idx=start_idx, end_idx=end_idx)
    bt_final.performance()
    print(bt_final.get_pnl_df())
    bt_final.summary_stats()
    return bt_final

if __name__ == "__main__":
    tickers = ['spy', 'xlb', 'xle', 'xlf', 'xli', 'xlk', 'xlu', 'xlv', 'xly']
    J = 10000
    df = 6
    rebalance_every = 10
    decay = 0.95

    bt = run_parallel_backtest(
        tickers=tickers, J=J, df=df,
        rebalance_every=rebalance_every, decay=decay,
        start_idx=0, end_idx=50
    )
