import requests 
import json 

def create_session(originPlace, destinationPlace, date):
    url = "https://skyscanner-skyscanner-flight-search-v1.p.rapidapi.com/apiservices/pricing/v1.0"
    headers = {
    'x-rapidapi-host': "skyscanner-skyscanner-flight-search-v1.p.rapidapi.com",
    'x-rapidapi-key': "0dc1882529mshd78b1b907f784dap1b1d60jsn3530abf3f164",
    'content-type': "application/x-www-form-urlencoded"
    }
    payload = "country=US&currency=USD&locale=en-US&originPlace={}&destinationPlace={}&outboundDate={}&adults=1".format(originPlace, destinationPlace, date)
    response = requests.request("POST", url, data=payload, headers=headers)
    if 'Location' in response.headers:
        return response.headers["Location"].split("/")[-1]
    else: 
        return None
    
def poll_results(originPlace, destinationPlace, date):
    headers = {
    'x-rapidapi-host': "skyscanner-skyscanner-flight-search-v1.p.rapidapi.com",
    'x-rapidapi-key': "0dc1882529mshd78b1b907f784dap1b1d60jsn3530abf3f164"
    }
    querystring = {"pageIndex":"0","pageSize":"10"}
    session_key = create_session(originPlace, destinationPlace, date)
    if session_key is not None: 
        url = "https://skyscanner-skyscanner-flight-search-v1.p.rapidapi.com/apiservices/pricing/uk2/v1.0/{}".format(session_key)
        response = requests.request("GET", url, headers=headers, params=querystring)
        return response
    else: 
        return None
    
def generate_web_links(originPlace, destinationPlace, date):
    res = json.loads(poll_results(originPlace, destinationPlace, date).text)
    try: 
        it = res["Itineraries"][0]
        links = []
        for opt in it["PricingOptions"]: 
            links.append(opt["DeeplinkUrl"])
        return links
    except:
        print("No web link available")

print(generate_web_links("SFO-sky", "EWR-sky", "2019-12-20"))
    