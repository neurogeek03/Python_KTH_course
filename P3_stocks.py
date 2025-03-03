import os
import glob
import copy
import pandas as pd
import numpy as np
import logging
import datetime
import matplotlib.pyplot as plt
import io # for flexible input/output control

plt.rcParams["font.size"] = 14
plt.rcParams["axes.labelsize"] = 17
plt.rcParams["axes.titlesize"] = 17

# Headers for prices
PRICE_HEADERS = ["Close/Last", "Open", "High", "Low"]
DATE = "Date"
CLOSE = "Close/Last"
OPEN = "Open"
VOLUMES = "Volume"
SMA50 = "SM50"  # Simple Moving Average 50 days
SMA200 = "SMA200"  # Simple Moving Average 200 days

# Date constants
# Dataset ends at 06/06/2024, and referred to as 'today'
TODAY = pd.to_datetime("2024-06-06", format="%Y-%m-%d")

# Dictionary for the time intervals
TIMES = {
    "max": TODAY - pd.DateOffset(years=10),
    "5yr": TODAY - pd.DateOffset(years=5),
    "1yr": TODAY - pd.DateOffset(years=1),
    "6m": TODAY - pd.DateOffset(months=6),
    "3m": TODAY - pd.DateOffset(months=3),
    "2m": TODAY - pd.DateOffset(months=2),
    "1m": TODAY - pd.DateOffset(months=1),
    "2w": TODAY - pd.DateOffset(days=14),
    "1w": TODAY - pd.DateOffset(days=7),
}

# Some colors for the context manager below
CLRS = [
    "b",
    "g",
    "r",
    "orange",
    "k",
    "c",
    "m",
    "y",
    "navy",
    "olive",
    "peru",
    "darkblue",
    "lime",
    "grey",
]


class NextColor:
    """Context manager to get the same color
    in a with-statement and automatically
    increment to next available color

    How to use:

    plt.figure()
    with NextColor() as color:
        plt.plot(x,y, color=color)
        plt.plot(x, y**y, color=color)

    plt.show()
    """

    i = 0

    def __init__(self):
        if NextColor.i >= len(CLRS):
            NextColor.i = 0

    def __enter__(self):
        """Return the color next in line"""
        return CLRS[NextColor.i]

    def __exit__(self, type, value, traceback):
        """Increment color number"""
        NextColor.i += 1


class Stock:
    """
    Class for the individual stocks. Holds the raw data, as well as any additionally calculated metrics for the stocks, 
    such as their simple moving averages.
    """
    def __init__(self, filepath: str, symbol: str):
        """Constructor : sets the symbol name
        and reads+parses the given data filepath

        Args:
            filepath (str): File path to read
            symbol (str): Name of the given symbol
        """
        self.__symbol = symbol
        self.__read_stock(filepath)
        self.__add_sma()

    def __read_stock(self, filepath: str):
        """Reads the stock data from the given filepath (relative/absolute)
        and converts it into the proper data types.

        Args:
            filepath (str): path to read
        """
        # Reading the CSV into a pandas dataframe 
        stock_data = pd.read_csv(filepath)

        # First, we define which are the price columns
        for column in PRICE_HEADERS:
            # Remove $ sign from data and convert to floats
            stock_data[column] = stock_data[column].replace({r'\$': ''}, regex=True).astype(float)

        # Convert the date column to dates, and sort all data
        # Ensuring correct format on the date, using "/"
        stock_data['Date'] = pd.to_datetime(stock_data['Date'], format='%m/%d/%Y')
        
        # Sorting the data by date (ascending order)
        stock_data = stock_data.sort_values(by='Date')

        # Storing the processed data 
        self.stock_data = stock_data

    def __add_sma(self):
        """Adds the 50 (SMA50) and 200 (SMA200) days Simple Moving
        Averages for the CLOSE price.
        """
        # Calculate 50/200-day Simple Moving Average (SMA)
        self.stock_data['SMA50'] = self.stock_data[CLOSE].rolling(window=50, min_periods=1).mean()
        self.stock_data['SMA200'] = self.stock_data[CLOSE].rolling(window=200, min_periods=1).mean()

    def change_since(self, since: datetime) -> float:
        """Returns the % change in value since the given date

        Args:
            since (datetime): change since that date

        Returns:
            float: % price change in percent
        """
        # Ensuring stock data is sorted by date
        stock_data = self.stock_data

        # Finding the closing price for the date defined under since:
        start_price = stock_data.loc[stock_data['Date'] == since, CLOSE]

        #Sometimes, the "since" date might be empty (i.e because it was the weekend/holiday)
        while start_price.empty or start_price.iloc[0] == 0:
            since += pd.DateOffset(days=1)
            
            # In case our search goes beyond available data 
            if since > stock_data['Date'].max():
                return 0.0
            
            #Checking for a valid start price again
            start_price = stock_data.loc[stock_data['Date'] == since, CLOSE]

        #Extracting the non-zero start price that has been identified as a pandas series
        start_price = start_price.iloc[0]

        #Establishing an end price with "today's" date
        end_price = stock_data.loc[stock_data['Date'] == TODAY, CLOSE]

        #Ensuring end_price is also not identified as a pandas series
        end_price = end_price.iloc[0]

        # Calculate the % change, avoiding division by zero
        if start_price == 0:
            return 0.0  
        
        change_percent = ((end_price - start_price) / start_price) * 100

        return change_percent


    def get_date_range(self, start: datetime = None, end: datetime = None) -> "Stock":
        """Extracts a subset of this stock that is limited to
        [start, end[ dates.
        If start or end is not given, then no filtering is applied.
        A new Stock() object is returned that is limited to the given date range.

        Args:
            start (datetime, optional): Start date of the data to return. If None, no limit
            end (datetime, optional): End date of the data to return. If None, no limit.

        Returns:
            Stock: Returns a stock with the data limited to the date range given.

        Raises:
            ValueError: If end is same or before start date
        """
        # Check bounds of the start/end date
        # Ensuring that the start date is *before* the end date 
        if start and end and end <= start:
            raise ValueError("End date must be after start date.")
        
        # Stock data will be a copy of the internal stock data...
        stock_data = self.stock_data

        # ...Unless, any start/end filters are applied 
        if start:
            stock_data = stock_data[stock_data['Date'] >= start]
        
        if end:
            stock_data = stock_data[stock_data['Date'] <= end]

        # Creating a CSV buffer to save the filtered data to it 
        buffer = io.StringIO()

        # Write the filtered DataFrame to the buffer with the correct date format
        stock_data['Date'] = stock_data['Date'].dt.strftime('%m/%d/%Y')  
        stock_data.to_csv(buffer, index=False) 

        # Reset the buffer cursor to the start
        buffer.seek(0)

        # Return a new Stock
        # This new stock object will contain less data, based on the start/end date filtering
        new_stock = Stock(buffer, self.__symbol)
        return new_stock
        
    @property
    def symbol(self) -> str:
        return self.__symbol


class StocksDB:
    """
    Class to collect all the individual stocks (stock database).
    """
    def __init__(self, path: str):
        """Reads in all csv files in a path"""
        #A dictionary is initiated here where the filepath and symbol of each stock will be stored
        self.__stocks = {}
        self.read_files(path)

    def read_files(self, path: str):
        """Reads all the csv files in the given path.
        Each will be stored as a {symbol : Stock} in the __stocks attribute.

        Args:
            path (str): relative or absolute path to the stocks directory.
        """
        # Loop over all csv files in the given path
        for filepath in glob.glob(os.path.join(path, "*.csv")):
            # Extract the stock symbol from the filename (e.g., 'aapl.csv' -> 'aapl')
            symbol = os.path.splitext(os.path.basename(filepath))[0]

            # Read the file into a Stock and store
            self.__stocks[symbol] = Stock(filepath, symbol)

    def __getitem__(self, name: str) -> Stock:
        """Overload to access stocks in the
        underlying data structure, e.g. db['aapl']

        Args:
            name (str): symbol name

        Returns:
            Stock: the wanted Stock
        """
        try:
            return self.__stocks[name]
        except KeyError:
            logging.error(f"key [{name}] not found")
            return None

    def __iter__(self):
        """Implements the iterator to loop over all the stocks"""
        for stock in self.__stocks.values():
            yield stock


class Plot:
    """
    Plots stock market data in line form or candlestick form.
    If more than 1 arguments for stock symbols are given, only line form is possible. 
    If exactly 1 argument for a stock symbol is given, both candlestick and line formats are possible. 
    The line format for one argument will also contain two lines displaying SMA. 
    If no arguments are provided, the default is to display a single line per stock, in a single figure (plot_all).
    """
    def __init__(self, db: StocksDB, start: datetime = None, end: datetime = None):
        # Set internal attributes
        self.__db = db
        self.__start = start
        self.__end = end

        # Setup plot - I removed the graph specifications and put them only under plot and plot all, because im using a different method for the candlestick plot
        # This is to avoid multiple figure windows from popping up

    def candlestick(self, symbol: str):
        """Generate a Candlestick plot of the given symbol name

        Args:
            symbol (str): Stock symbol name to plot
        """
        # Fetch stock for the given date interval
        stock = self.__db[symbol]
        stock_data = stock.get_date_range(self.__start, self.__end).stock_data

        if stock_data.empty:
            print(f"No data available for {symbol} in the specified date range.")
            return
        
        # Prepare the data for candlestick plotting
        stock_data['Date'] = pd.to_datetime(stock_data['Date'])
        stock_data.set_index('Date', inplace=True)

        # Calculate percentage change
        start_price = stock_data['Open'].iloc[0] 
        end_price = stock_data[CLOSE].iloc[-1]  
        percent_change = ((end_price - start_price) / start_price) * 100
        percent_change_str = f"{percent_change:.2f}%"

        # Create a figure and axis for the plot
        figure, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={'height_ratios': [4, 1]})

        # Create the candlesticks plot
        for i in range(len(stock_data)):
            # Assigning variables for clarity
            date = stock_data.index[i]
            open_price = stock_data['Open'].iloc[i]
            close_price = stock_data[CLOSE].iloc[i]

            # Define the color based on open/close prices
            color = 'g' if close_price >= open_price else 'r'

            # Plot the high and low prices as a line
            # ax1.plot([date, date], [low_price, high_price], color='black')

            # Plot the open and close prices as a bar
            ax1.bar(date, close_price - open_price, bottom=open_price, color=color, width=0.8)

        # Add the volumes to the plot
        ax2.bar(stock_data.index, stock_data['Volume'], color='blue', alpha=0.6)
        ax2.set_ylabel('Volume')
        ax2.set_xlabel('Date')

        # Set the title and layout for better aesthetics
        ax1.set_title(f'{symbol}')
        # plt.xticks(rotation=45)

        ax1.legend([f"{symbol}: {percent_change_str}"], loc='upper left')
        ax1.grid(True)
        ax2.grid(True)
        plt.show()

    def plot(self, symbols: list[str]):
        """Creates a time series plot of all the symbols provided
        during the dates given before.

        Args:
            symbols (list[str]): List of symbol names
        """
        plt.figure(figsize=(14, 6))
        plt.grid()
        plt.xlabel(DATE)
        plt.ylabel("Closing Price [$]")

        # Getting stock data for a given symbol by iterating over the symbols list
        for symbol in symbols:
            stock = self.__db[symbol]

            if stock:
                # Obtaining a filtered stock, if a date range is specified
                filtered_stock = stock.get_date_range(self.__start, self.__end) if (self.__start or self.__end) else stock.stock_data

                # Extract the stock_data DataFrame from the filtered Stock object
                stock_data = filtered_stock.stock_data

                if not stock_data.empty:
                    # Calculate the change since the beginning of the date range
                    perc_change = stock.change_since(self.__start)
                    #Ensuring consistent coloring across plots for each stock symbol
                    with NextColor() as color:

                        #Adding percentage change to the legend
                        legend_label = f"{symbol} {perc_change:.2f}%"

                        # Plot the data
                        plt.plot(stock_data['Date'], stock_data[CLOSE], label=legend_label, color=color)
                        
                        # Plot SMAs, using a different pattern for each window
                        plt.plot(stock_data['Date'], stock_data['SMA50'], 
                            color=color, linestyle='-.', linewidth=1)
                        plt.plot(stock_data['Date'], stock_data['SMA200'],
                            color=color, linestyle='dotted', linewidth=1)
                    
                    # Set x-axis limits to fill the full frame
                    plt.xlim(stock_data['Date'].min(), stock_data['Date'].max())
            else:
                logging.error(f"Stock {symbol} not found in database.")
        
        #Showing a legend with stocks symbols 
        plt.legend(loc = "upper left")
        plt.show()

    def plot_all(self):
        """Plots all the symbols in the StockDB
        during the dates given in the constructor
        """
        plt.figure(figsize=(14, 6))
        plt.grid()
        plt.xlabel(DATE)
        plt.ylabel("Closing Price [$]")

        #Looping through all the stocks in the database 
        for stock in self.__db: 
            symbol = stock.symbol

            # Obtaining each stock's date range
            stock_data = stock.get_date_range(self.__start, self.__end)
            
            if stock_data:  # Check if stock_data is not None or empty
                plt.plot(stock_data.stock_data['Date'], stock_data.stock_data[CLOSE], label=symbol)
            else:
                logging.error(f"No data available for {symbol} in the specified date range.")
                
        plt.legend(loc="upper left")
        plt.show()



class Table:
    """
    Displays the stock market data according to different date intervals
    """
    def __init__(self, db: StocksDB):
        self.__db = db

    def print(self, sort_by: str = "symbol", limit: int = None):
        """Prints the table

        Args:
            sort_by (str): How to sort the data, default = symbol
            limit (int): How many stocks to print, default = None -> All
        """
        # Create the data structure for the table
        # We set the table data to be a list of values 
        table_data =[]

        for stock in self.__db:
            symbol = stock.symbol
            # Extract price changes based on the TIME constants
            max_change = stock.change_since(TIMES["max"])
            five_year_change = stock.change_since(TIMES["5yr"])
            one_year_change = stock.change_since(TIMES["1yr"])
            six_month_change = stock.change_since(TIMES["6m"])
            three_month_change = stock.change_since(TIMES["3m"])
            two_month_change = stock.change_since(TIMES["2m"])
            one_month_change = stock.change_since(TIMES["1m"])
            two_week_change = stock.change_since(TIMES["2w"])
            one_week_change = stock.change_since(TIMES["1w"])

            # Append to the table data as a tuple
            table_data.append((
                symbol,
                max_change,
                five_year_change,
                one_year_change,
                six_month_change,
                three_month_change,
                two_month_change,
                one_month_change,
                two_week_change,
                one_week_change
            ))

        # Sort the data
        if sort_by == "symbol":
            table_data.sort(key=lambda x: x[0])  # Sort by symbol alphabetically
        elif sort_by in TIMES.keys():  # Sort by price development if valid
            # Get the index of the corresponding time period
            index = list(TIMES.keys()).index(sort_by) + 1  # +1 because index 0 is symbol
            table_data.sort(key=lambda x: x[index], reverse=True)  # Sort descending

        # Limit the number of rows
        if limit is not None:
            table_data = table_data[:limit]

        # Print the results, making sure to align column titles with rows below.
        header = "{:<8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8}".format( "Symbol", "Max", "5yr", "1yr", "6m", "3m", "2m", "1m", "2w", "1w")
        print(header)
        
        for row in table_data:
            print("{:<8} {:>8.2f} {:>8.2f} {:>8.2f} {:>8.2f} {:>8.2f} {:>8.2f} {:>8.2f} {:>8.2f} {:>8.2f}".format(*row))
