import os.path as path
import csv
import argparse
import sys

SCORE_TTL = "Score"
YEAR_TTL = "Year"
TITLE_TTL = "Title"
ACTORS_TTL = "Actors"


def read(filepath: str) -> list[dict]:
    """
    Reads a csv file and makes every row to be of type dictionary
    """
    movies = [] #initalizing the list of movies 

    #using with to elegantly handle this external resource (csv file) and assigning a variable name to the opened file
    with open(filepath, mode='r', encoding='utf-8') as csvfile: 
        #Here, r = read, encoding = utf-8 is very common for csvs

        #To read the file as alist of dictionaries we will use the function below
        #**csv.DictReader** reads a CSV file and maps each row to a dictionary, using the CSV file's headers as the dictionary keys
        interpreter = csv.DictReader(csvfile) #assigning a variable name to the interpreted file 

        #Each row is read individually and appended as a dictionary to the movies variable 
        for row in interpreter: 
            movies.append(dict(row)) #this line populates the movies list we initiated at the start
            #the loop stops iterating once all csv rows have been appended to the list
    return movies 


def enrich_data(movies: list[dict]) -> list[dict]:
    """
    Adds the year and actor columns to the movie dictionaries
    """
    #Calling the add_year function 
    movies_with_year = add_year(movies)
    
    #Calling the add_actor function 
    enriched_movies = add_actor(movies_with_year)
    
    # Return the enriched list of movies
    return enriched_movies


def print_movies(data: list[dict]):
    """
    Prints the movies that we have filtered for, in a specific format, limiting the number of characters per column. 
    The column names will appear in this order: Score, Year, Title, Actors.
    """
    #First, lets add a header and a separator line: 
    header = f"{'Score':<7} {'Year':<8} {'Title':<33} {'Actors':<50}"
    print(header)
    print("=" * 100)

    #We will loop through each movie and format it
    for movie in data:
        score = f"{movie.get('score', ''):<7}"  #Score, limited to 7 characters
        year = f"{movie.get('year', ''):<8}"    #Year, limited to 8 characters
        
        #Some titles might be longer than 33 characters, so we will add "..."
        title = movie.get('names')
        if len(title) > 33:
            title = title[:30] + '...' #indexing until character 30 and adding ellipsis for the rest 3 characters
        title = f"{title:<33}"
    #Explantation of formatted string: 
    #title wil be replaced by the actual movie title
    #< means that the text will be left-aligned 
    #33 specifies the width of the output

    #Limiting actors to 50 characters 
        actors = ", ".join(movie.get('actors', []))  #Joining actor names by commas

        if len(actors) > 50:
            actors = actors[:47] + '...'  #same as for title
            actors = f"{actors:<50}"
    
    #Printing the formatted movie details in one row
        print(f"{score} {year} {title} {actors}")

    #Optional return statement 
    return "Movies formatted successfully!"




def add_year(movies: list[dict]) -> list[dict]:
    """
    Adds a 'year' key to each movie dictionary by extracting the year from the 'date_x' key.

    Format of date_x is expected as dd/mm/yyyy

    The function extracts the year (yyyy) and stores it as an integer in a new 'year' key 
    for each movie dictionary
    """
    # We have to extract the year from the component "date" of each movie dictionary
    # This is the format of the date: 03/02/2023
    # We have to split the string based on the "/"

    #we will repeat certain steps for each movie until we do this for all the movies in our csv file
    for movie in movies:

    #assigning a variable name to the component "date_x" of each movie dictionary
        date = movie.get('date_x') #we use quotation marks to access the dictionary key of choice

        #Splitting the date entry for each movie by the "/"
        date_list = date.split("/")

        #Extracting the year, which is always going to be the third element
        #in the string that we split during the previous step 
        year = int(date_list[2])

        #Add the "year" as a new key in the movie dictionary
        movie["year"] = year
    
    return movies 


def add_actor(movies: list[dict]) -> list[dict]:
    '''
    Adds ta list of actors to the movie dictionary, after extracting them from the "crew" key
    '''
    # We will iterate over all movies 
    for movie in movies:
        #assigning a variable name to the component "crew" of each movie dictionary 
        crew = movie.get('crew')

        #separating the crew string into smaller strings based on the commas
        crew_list = [item.strip() for item in crew.split(',')]

        #The crew column also contains the name of the movie characters played by the actors
        #But we only want actors names so lets extract them
        #The format of "crew" is: actor, character, actor, character etc.
        #To get the actors we get all the even indexes along the string
        actors = [crew_list[i] for i in range(0, len(crew_list), 2)]

        #Storing actors as a list inside the movie dictionary 
        movie['actors'] = actors
    
    return movies 


def get_filtered_movies(
    movies: list[dict],
    actors: list[str] = None,
    genres: list[str] = None,
    years: list[int] = None,
    top: int = 0,
    sort_by: str = None,
    ascending: bool = False,
) -> list[dict]:
    """Takes in the data list and filters the movies based on the
    given parameters

    Args:
        movies (list[dict]): List of all the movies
        actors (list[str]): List of Actors that needs to be in the movies
        genres (list[str]): List of Genres to include in the filter
        years (list[int]): List of years movies were released in
        top (int): How many movies to return
        sort_by (str): What key to sort by, default 'score'
        asc (bool): Sort ascending (e.g. the lowest score is returned first)

    Returns:
        list[dict]: The filtered list of movies
    """
    #Starting a full list that will evetually contain the movies we filter (after all filters are applied)
    movies_filtered = movies.copy()

    #Filter by actors if an actors list is provided as input (not None)
    if actors:
        movies_filtered_actors = []  #list for actor filtering

        for movie in movies_filtered:
            #get the actors key for each movie, or an empty list if no actors are there 
            movie_actors = movie.get('actors', [])

            #Initialize a boolean. To test if all input actors are present in the movie list 
            all_actors_present = True #we assume they are all present for now

            #Now, in this list of actors that was given input there are two options. 
            #Each actor can be: present or not present 
            #So, we check this for each actor name in the input and include the movie if all 
            #actor names are present.

            for actor in actors: #we loop over the input of actors
                if actor not in movie_actors:
                    all_actors_present = False
                    break
            
            if all_actors_present:
                movies_filtered_actors.append(movie)
            
        movies_filtered = movies_filtered_actors #update the filtered movies list
    
    #Filter by genres such that at least one of the genres in the input matches the movie's genres
    if genres: 
        movies_filtered_genres = [] #initialize a new list for filtering

        for movie in movies_filtered: #iterate through list of movies

            movie_genres = movie.get('genre', []) #get the genre component of each movie dictionary 

            #Looping over all the genres provided as input
            for genre in genres:
                 #Checking if any genre in the input matches the movie genres 
                 if genre in movie_genres:
                     movies_filtered_genres.append(movie) #add movie to the list if it matches 
                     break
                 
        movies_filtered = movies_filtered_genres
    
    #Filter by year released such that the movie is released in one of the years provided by the list
    if years: 
        movies_filtered_years = [] #initialize a new list for filtering

        for movie in movies_filtered: #iterate through list of movies
            movie_year = movie.get('year') #get the genre component of each movie dictionary 

            #Checking if any genre in the input matches the movie genres 
            if movie_year in years:
                movies_filtered_years.append(movie) #add movie to the list if it matches
                 
        movies_filtered = movies_filtered_years
    
    if sort_by:
        #If any other input is given other than score or year, an error will occur
        if sort_by not in ['score', 'year']:
            raise ValueError("sort_by must be 'score' or 'year'.")
        
        #otherwise, sorting will happen based on the input given 
        #the sort() function changes the order of elements in an existing list based on the key indicated
        #To get the sorting directive key, a lambda function is defined inside the sort() function
        #Finally, the order (ascending or descending) will depend on the boolean input for the variable "ascending"   
        movies_filtered.sort(key = lambda x: x.get(sort_by), reverse= not ascending)

    #Return the top n movies
    if top > 0:
        movies_filtered = movies_filtered[:top]

    return movies_filtered
        


def main():
    # path.join() is agnostic to operating system (Windows Vs Linux)
    data = read(path.join("data", "imdb_movies.csv"))

    # -------------------------------------
    # Command line input parser
    # -------------------------------------
    arg_parser = argparse.ArgumentParser()

    #Adding argument for actors 
    arg_parser.add_argument(
        "--actors", 
        nargs="+", #ability to add multiple actors separated by spaces 
        help = "Space-separated list of desired actors: eg. --actors 'Tom Cruise' 'Morgan Freeman'"
    )
    
    arg_parser.add_argument(
        "--genres", 
        nargs="+",
        help = "Space-separated list of desired genres: eg. --genres 'Action' 'Drama'"
    )

    arg_parser.add_argument(
        "--years", 
        nargs="+",
        type = int, #specifying that years are integer
        help = "Space-separated list of desired years: eg. --years 2023 2024"
    )

    arg_parser.add_argument(
        "--top", 
        type = int,  
        help = "Returns the top n movies: eg. --top 20"
    )

    arg_parser.add_argument(
        "--sort",
        choices=["score", "year"], #limiting the input
        help = "Sort movies by either 'score' or 'year': eg. --sort score"
    )

    #The ascending variable is not an argument, but a flag 
    #If it is provided, it takes the value "True"

    arg_parser.add_argument(
        "--ascending", 
        action="store_true",  
        help = "Sort in ascending order: eg. --ascending"
    )

    # -------------------------------------
    args = arg_parser.parse_args()
    # -------------------------------------

    # Add the additional fields to the raw data
    movies = enrich_data(data)

    # Run the filter function
    movies = get_filtered_movies(
        movies,
        args.actors,
        args.genres,
        args.years,
        args.top,
        args.sort,
        args.ascending,
    )
    # And finally print a nice table
    print_movies(movies)

if __name__ == "__main__":
    main()
