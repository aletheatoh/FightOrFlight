# from flight_data import data
from data import all_data
import numpy as np
import math
import operator
import datetime
import time
import sys
import heapq
import csv

# Dictionary of strings and ints
airlineIndexDict = {
    "Alaska Airlines": 0,
    "Allegiant Air" : 1 ,
    "Delta Air Lines" : 2,
    "Frontier Airlines" : 3,
    "Hawaiian Airlines" : 4,
    "jetBlue" : 5,
    "Southwest Airlines" : 6,
    "Spirit Airlines" : 7,
    "United" : 8,
    "Linear Air" : 9,
    "Boutique Air": 10,
    "Sun Country Airlines": 11
    }

citiesIndexDict = {}
cityIndex = 0

'''
Obtains all possible flight paths/combinations for a given set of destinations
'''
def user_input(destinations):
    flight_paths = [(x,y) for x in destinations for y in destinations if x != y]
    return flight_paths

'''
Creates dictionary where each key is an airline, where the correspoding value is
in the format of (average duration, average cost, frequency) across all flight paths that
include the airline. We are only considering flight paths that a user travels on.
'''
def create_dict(flight_data, user_history):
    global citiesIndexDict
    global cityIndex
    dict = {}

    for flight_path in flight_data:
        flights = flight_path['segments']
        from_place = flights[0]['from_place']['Code'] # from location
        to_place = flights[-1]['to_place']['Code'] # end location

        # Add each city and their corresponding index to the citiesIndexDict
        if from_place not in citiesIndexDict:
            citiesIndexDict[from_place] = cityIndex
            cityIndex += 1

        if to_place not in citiesIndexDict:
            citiesIndexDict[to_place] = cityIndex
            cityIndex += 1

        airlineWithMaxDuration = 0
        max_duration = 0
        # for each flight in a given flight path
        for flight in flights:
            carrier = flight['carrier']['Name']

            # find the airline with max duration in a given flight path
            if flight['duration'] >= max_duration:
                airlineWithMaxDuration = carrier
                max_duration = flight['duration']

        # add flight_path to dictionary with airlineWithMaxDuration as the key.
        if airlineWithMaxDuration not in dict:
            dict[airlineWithMaxDuration] = [] # include duplicate flight paths

        flight_item = (flight_path['duration'], flight_path['price'])
        dict[airlineWithMaxDuration].append(flight_item) # add a flight path (NOT flight) to an airline in dictionary

    return dict

'''Computes the average cost, duration, and number of offered flights for each airline (key)'''
def computeAverages(airlineDict):
    # compute averages of duration, cost for all flight paths of a given airline
    for airline in airlineDict.keys():
        airline_flight_paths = airlineDict[airline]
        freq = len(airline_flight_paths) # frequency of flight paths offered by the airline
        airlineDict[airline] = [sum(x)/freq for x in zip(*airline_flight_paths)]
        airlineDict[airline].append(freq)
        airlineDict[airline].append(airline)
    return airlineDict

''' Sort the airlines in order of their flight duration '''
def rankAirlinesByDuration(airlineDict):
    return sorted([(airline, duration) for (duration, cost, freq, airline) in airlineDict.values()], key=operator.itemgetter(1))

''' Sort the airlines in order of their flight cost '''
def rankAirlinesByCost(airlineDict):
    return sorted([(airline, cost) for (duration, cost, freq, airline) in airlineDict.values()], key=operator.itemgetter(1))

''' Sort the airlines in order of their number of flights offered '''
def rankAirlinesByFrequency(airlineDict):
    return sorted([(airline, freq) for (duration, cost, freq, airline) in airlineDict.values()], key=operator.itemgetter(1), reverse=True)

''' Uses the ranking list for each category (duration, cost, frequency) to compute an average score'''
def scoreAirlines(durationRank, costRank, frequencyRank, airlineIndexDict):
    airlines = [0 for airline in airlineIndexDict.keys()]
    airlines_list = list(airlineIndexDict.keys())

    for rank in range(len(durationRank)):
        airline = durationRank[rank][0]
        airlineIndex = airlineIndexDict[airline]
        airlines[airlineIndex] += rank

        airline = costRank[rank][0]
        airlineIndex = airlineIndexDict[airline]
        airlines[airlineIndex] += rank

        airline = frequencyRank[rank][0]
        airlineIndex = airlineIndexDict[airline]
        airlines[airlineIndex] += rank

    averageScores = [score/3 for score in airlines]

    return sorted([(airlines_list[i], score) for (i, score) in enumerate(averageScores) if score > 0], key=operator.itemgetter(1))

def filterFlightDataByAirline(flight_data, winning_airline):
    filtered = []
    for flight_path in flight_data:
        include = False
        for flight in flight_path['segments']:
            # we only want to look at flight paths offered by the winning airline
            if winning_airline == flight['carrier']['Name']:
                include = True
                break

        if include:
            filtered.append(flight_path)

    return filtered

'''Create a single vector to represent a flight path
    In the format of flightPathVector = [Price, Duration, from_placeIndex, to_placeIndex, DepartureTime, ArrivalTime, # of stops]'''
def createVector(flight_path):
    global citiesIndexDict

    flights = flight_path['segments']
    from_place = flights[0]['from_place']['Code'] # from location
    to_place = flights[-1]['to_place']['Code'] # end location
    duration = flight_path['duration']
    price = flight_path['price']
    departureTimeObj = datetime.datetime.strptime(flight_path['departure_time'],"%Y-%m-%dT%H:%M:%S")
    departureTime = departureTimeObj.hour*60*60 + departureTimeObj.minute*60 + departureTimeObj.second
    arrivalTimeObj = datetime.datetime.strptime(flight_path['arrival_time'],"%Y-%m-%dT%H:%M:%S").time()
    arrivalTime = arrivalTimeObj.hour*60*60 + arrivalTimeObj.minute*60 + arrivalTimeObj.second
    numOfStops = len(flights)-1
    from_placeIndex = citiesIndexDict[from_place]
    to_placeIndex = citiesIndexDict[to_place]

    return [price, duration, from_placeIndex, to_placeIndex, departureTime, arrivalTime, numOfStops]

'''Create vectors for each flight path in a user's past flight data and return a np.2dArray'''
def createUserMatrix(fileName):
    global citiesIndexDict
    global cityIndex

    user_history = []
    with open(fileName) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        next(csv_reader) # skip header row
        for row in csv_reader:
            flightPath = []

            flightPath.append(float(row[0])) # price
            flightPath.append(float(row[1])*60) # duration in minutes

            from_place = row[2]
            if from_place not in citiesIndexDict:
                citiesIndexDict[from_place] = cityIndex
                cityIndex += 1
            to_place = row[3]
            if to_place not in citiesIndexDict:
                citiesIndexDict[to_place] = cityIndex
                cityIndex += 1

            flightPath.append(citiesIndexDict[from_place]) # from place
            flightPath.append(citiesIndexDict[to_place]) # to place

            departureTimeObj = datetime.datetime.strptime(row[4],"%H%M")
            flightPath.append(departureTimeObj.hour*60*60 + departureTimeObj.minute*60 + departureTimeObj.second)
            arrivalTimeObj = datetime.datetime.strptime(row[5],"%H%M")
            flightPath.append(arrivalTimeObj.hour*60*60 + arrivalTimeObj.minute*60 + arrivalTimeObj.second)
            flightPath.append(float(row[6])) # number of stops

            user_history.append(flightPath)

    return user_history

'''Create vectors for each flight path in the flight data and return a np.2dArray'''
def createFlightMatrix(flight_data):
    flightMatrix = []
    idx = 0
    for flight_path in flight_data:
        flightMatrix.append(createVector(flight_path))
    return flightMatrix

'''
Computes the cosine similarity between two flight paths
'''
def cosine_sim(x, y):
    if np.count_nonzero(x) == 0 or np.count_nonzero(y) == 0:
        return 0 # runtime warning: if either vector is 0
    return np.dot(x, y) / (np.linalg.norm(x)*np.linalg.norm(y))

'''
Creates cache for cosine similarities
'''
def create_cosine_sim_cache(user_matrix, flight_matrix):
    cosine_sim_cache = np.zeros((len(user_matrix),len(flight_matrix)))
    for i in range(len(user_matrix)):
        for j in range(len(flight_matrix)):
            val = cosine_sim(user_matrix[i],flight_matrix[j])
            cosine_sim_cache[i][j] = val
    return cosine_sim_cache

# gets sign vector of vector, returns bit value of sign vector (0 =)
def sign_bit_val(dp):
    # sign vector
    sign_vector = np.sign(dp)
    bit_val = 0
    pow = len(sign_vector) - 1
    for i in range(len(sign_vector)):
        sign = sign_vector[i]
        if sign > 0:
            bit_val += math.pow(2, pow)
        pow -= 1
    return bit_val

'''
Locality Sensitive Hashing
'''
def lsh(k, flight_matrix, user_matrix, y, cosine_sim_cache):
    user_vector = user_matrix[y]
    max_cos_sim_in_bucket = -sys.maxsize-1
    max_flight_path_in_bucket = 0
    comparison_count = 0

    # normalize each vector
    M = np.random.standard_normal(size=(k,7)) # lth hash table
    for i in range(len(M)):
        temp = M[i] / (M[i]**2).sum()**0.5
        M[i] = temp

    for i in range(len(flight_matrix)):
        x = flight_matrix[i]

        # compute dot product and hash value for user_vector
        dp_y = np.dot(M,user_vector)
        hash_val_flight_path = sign_bit_val(dp_y) # My

        # compute dot product and hash value for x
        dp_x = np.dot(M,x) # Mx
        hash_val_x = sign_bit_val(dp_x)

        if hash_val_x == hash_val_flight_path:
            comparison_count += 1
            cosine_sim_val = cosine_sim_cache[y][i]

            if cosine_sim_val >= max_cos_sim_in_bucket:
                max_cos_sim_in_bucket = cosine_sim_val
                max_flight_path_in_bucket = i
    # print("Number of comparisons: " + str(comparison_count))
    return (max_cos_sim_in_bucket, max_flight_path_in_bucket, comparison_count)

def lsh_test(l, k, user_matrix, flight_matrix, cache):
    max_heap = []
    recommendations = []
    total_comparisons = 0
    for i in range(len(user_matrix)): # query - user flight path
        user_vector = user_matrix[i]
        max_cos_sim = -sys.maxsize-1
        max_flight_path = 0

        # CREATE HASH TABLES HERE
        for ml in range(l): # go through all l hash tables
            lsh_val = lsh(k, flight_matrix, user_matrix, i, cache)
            lsh_val_max_cosim = lsh_val[0] # max in bucket
            lsh_val_max_flight_path = lsh_val[1] # max in bucket

            if lsh_val_max_cosim >= max_cos_sim:
                max_cos_sim = lsh_val_max_cosim
                max_flight_path = lsh_val_max_flight_path

            total_comparisons += lsh_val[2]

        recommendations.append(max_flight_path)

    # print("total # of comparisons: " + str(total_comparisons))
    return recommendations

'''
Compares the average cost of a user's flight history and that of flight recommendations
Evaluates if our recommendations indeed perform better than the user's current travel patterns
'''
def compareCost(user_matrix, flight_matrix, recommendations):
    avg_user_costs = np.mean([path[0] for path in user_matrix])
    avg_rec_costs = np.mean([flight_matrix[idx][0] for idx in recommendations])
    return (avg_rec_costs < avg_user_costs, abs(round(avg_user_costs-avg_rec_costs,2)))

'''
Compares the average duration of a user's flight history and that of flight recommendations
Evaluates if our recommendations indeed perform better than the user's current travel patterns
'''
def compareDuration(user_matrix, flight_matrix, recommendations):
    avg_user_durations = np.mean([path[1] for path in user_matrix])
    avg_rec_durations = np.mean([flight_matrix[idx][1] for idx in recommendations])
    return (avg_rec_durations < avg_user_durations, abs(round(avg_user_durations-avg_rec_durations,2)))

'''---------------------MAIN METHOD-----------------------'''

'''
PHASE 1
'''
flight_data = []
for line in all_data:
    flight_data += line['result']
user_history = user_input(['RDU','ATL','BOS','DCA','SJU','SFO','DEN','CLT','JFK','IND','DEN','MSP'])
dict = create_dict(flight_data, user_history)

# compute averages and score each airline
averagesDict = computeAverages(dict)
costRank = rankAirlinesByCost(averagesDict)
durationRank = rankAirlinesByDuration(averagesDict)
frequencyRank = rankAirlinesByFrequency(averagesDict)
print("Airlines ranked by duration:")
print(durationRank)
print()
print("Airlines ranked by cost:")
print(costRank)
print()
print("Airlines ranked by frequency:")
print(frequencyRank)
print()
print("------------BEST AIRLINE ACROSS ALL FACTORS (COST, DURATION, FREQUENCY):------------")
airlinesScored = scoreAirlines(durationRank, costRank, frequencyRank, airlineIndexDict)
print("Top airlines: " + str(airlinesScored))
winning_airline = airlinesScored[0][0]
print("Winning airline: " + winning_airline)
print()
# print("Scoring:")
# print(scoreAirlines(durationRank, costRank, frequencyRank, airlineIndexDict))
# print()
'''
PHASE 2
'''
filteredFlightData = filterFlightDataByAirline(flight_data, winning_airline)
flight_matrix = createFlightMatrix(filteredFlightData)
user_matrix = createUserMatrix('test.csv')
cache = create_cosine_sim_cache(user_matrix, flight_matrix)

print("------------USER SHOULD PURCHASE THE FOLLOWING FLIGHTS OFFERED BY " + winning_airline + ":------------")
recommendations = set(lsh_test(16, 16, user_matrix, flight_matrix, cache))
for rec in recommendations:
    print(filteredFlightData[rec])
    print()

print("------------WILL USER SAVE ON COST WITH OUR RECOMMENDATIONS?------------")
costBenchmark = compareCost(user_matrix, flight_matrix, recommendations)
if costBenchmark[0]:
    print("Yes, User saves on average $" + str(costBenchmark[1]) + " per flight path")
else:
    if costBenchmark[1] == 0:
        print("No, User pays on average the same cost per flight path")
    else: print("No, User pays on average $" + str(costBenchmark[1]) + " more per flight path")
print()
print("------------WILL USER SAVE ON DURATION WITH OUR RECOMMENDATIONS?------------")
durationBenchmark = compareDuration(user_matrix, flight_matrix, recommendations)
if durationBenchmark[0]:
    print("Yes, User saves on average " + str(durationBenchmark[1]) + " min per flight path")
else:
    if durationBenchmark[1] == 0:
        print("No, User travels on average the same duration per flight path")
    else: print("No, User travels on average " + str(durationBenchmark[1]) + " more min per flight path")
