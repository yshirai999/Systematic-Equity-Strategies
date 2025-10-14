import sys, os

sys.path.append(os.path.abspath('.'))

import multiprocessing as mp
from tqdm import tqdm
import time
from Backtesting import DSPBacktester, MeanCVaRBacktester, Backtester

# Define save directory relative to current file
SAVE_DIR = os.path.join(os.path.dirname(__file__), 'Results')
os.makedirs(SAVE_DIR, exist_ok=True)
SAVE_PATH = os.path.join(SAVE_DIR, "backtest_results.npy")

def compute_weights_all_args(args):
    date_idx, tickers, lam, J, df, rebalance_every, decay, Optimizer = args
    if Optimizer == "DSP":
        bt = DSPBacktester(tickers=tickers, J=J, df=df,
                           lam=lam, rebalance_every=rebalance_every, decay=decay)
    else:
        bt = MeanCVaRBacktester(tickers=tickers, J=J, df=df,
                                lam=lam, rebalance_every=rebalance_every, decay=decay)
    # bt = Backtester(bt)
    return bt.compute_weights_on_date(date_idx)

def run_parallel_backtest(tickers, lam, J, df, rebalance_every, decay, start_idx, end_idx, Optimizer):
    date_indices = list(range(start_idx, end_idx, rebalance_every))
    args_list = [(date_idx, tickers, lam, J, df, rebalance_every, decay, Optimizer) for date_idx in date_indices]

    num_procs = max(mp.cpu_count() - 2, 1)
    print(f"Launching parallel backtest using {num_procs} processes")
    start_time = time.time()

    results = []

    if SAVE_PATH:
        print(f"Confirm you want to save results to {SAVE_PATH}? (y/n)")
        confirm = input().strip().lower()
        save_path = SAVE_PATH
        if confirm != 'y':
            print("Results will not be saved.")
            save_path = None

    with mp.Pool(processes=num_procs) as pool:
        with tqdm(total=len(args_list)) as pbar:
            for res in pool.imap(compute_weights_all_args, args_list):
                results.append(res)
                pbar.update()

    elapsed = time.time() - start_time
    print(f"Finished backtest in {elapsed:.2f} seconds")

    if Optimizer == "DSP":
        bt = DSPBacktester(tickers=tickers, J=J, df=df,
                                 rebalance_every=rebalance_every, decay=decay)
    else:
        bt = MeanCVaRBacktester(tickers=tickers, J=J, df=df,
                                      lam=lam, rebalance_every=rebalance_every, decay=decay)
    bt_final = Backtester(Optimizer=bt)
    bt_final.store_backtest_results(results, start_idx=start_idx, end_idx=end_idx, save_path=save_path)
    #bt_final.performance()
    print(bt_final.get_pnl_df())
    #bt_final.summary_stats()
    return bt_final

if __name__ == "__main__":
    tickers = ['spy', 'xlb', 'xle', 'xlf', 'xli', 'xlk', 'xlu', 'xlv', 'xly']
    lam = 0.5
    J = 10000
    df = 6
    rebalance_every = 20
    decay = 0.95

    bt = run_parallel_backtest(
        tickers=tickers, lam=lam, J=J, df=df,
        rebalance_every=rebalance_every, decay=decay,
        start_idx=0, end_idx=4330, Optimizer="DSP"
    )
