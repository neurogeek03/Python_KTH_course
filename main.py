import logging
import matplotlib.pyplot as plt
import argparse
import stocks
import pandas as pd


def setup_logger():
    logging.basicConfig(
        filename="main.log",
        format="[%(levelname)s] %(asctime)s : %(message)s",
        encoding="utf-8",
        level=logging.INFO,
    )


def main(args):
    db = stocks.StocksDB("data")

    # Instructions on how to handle start and end date information, if provided
    start_date = args.start if args.start else None
    end_date = args.end if args.end else None

    # Check for start and end dates (logic: end date should be after start date)
    if start_date and end_date:
        if end_date < start_date:
            logging.error("End date must be after start date.")
            return  # Exit if the dates are invalid
    
    # initializing an instance of Plot class and passing db (an instance of the StocksDB class) to it
    plotter = stocks.Plot(db, start_date, end_date)

    #Checking if the plot argument has been provided 
    if args.plot is not None:

        # Plot for all stocks if no specific symbols are provided
        if len(args.plot) > 0:  # Check if the list is not empty
            plotter.plot(args.plot)  # Plot for specific symbols if provided
        else:
            plotter.plot_all()

    elif args.candle:
        plotter = stocks.Plot(db, start_date, end_date)
        # Accessing the first and only item parsed to --candle
        plotter.candlestick(args.candle[0])

    elif args.table:
        table = stocks.Table(db)
        table.print(sort_by=args.table, limit=args.limit)


if __name__ == "__main__":
    setup_logger()

    # -------------------------------------
    # Command line input parser
    # -------------------------------------
    arg_parser = argparse.ArgumentParser()

    # add start/end aguments
    arg_parser.add_argument(
    '--start',
    type=lambda s: pd.to_datetime(s, format="%Y-%m-%d"),
    help="Start date for filtering stock data (format: YYYY-MM-DD)")
    
    arg_parser.add_argument(
    '--end',
    type=lambda s: pd.to_datetime(s, format="%Y-%m-%d"),
    help="End date for filtering stock data (format: YYYY-MM-DD)")

    # add table/limit arguments
    arg_parser.add_argument(
        '--table',
        type=str,
        choices=['symbol', 'max', '5yr', '1yr', '6m', '3m', '2m', '1m', '2w', '1w'],
        help="Sort by stock symbol or price development (max, 5yr, 1yr, 6m, 3m, 2m, 1m, 2w, 1w)")

    arg_parser.add_argument(
        '--limit',
        type=int,
        help="Limit the number of stocks to top n entries")
    
    # add plot/candle arguments
    arg_parser.add_argument(
        '--plot',
        # Allows one or more stock symbols, interpreting them as a list
        nargs='*',
        help="Plot closing prices for one or more stock symbols")

    arg_parser.add_argument(
        '--candle',
        type=str,
        #only 1 argument is allowed here
        nargs=1,
        help="Display candlestick chart for a single stock symbol")

    args = arg_parser.parse_args()
    # -------------------------------------
    logging.info(f"Started running main.py: args: {args}")
    # -------------------------------------

    main(args)