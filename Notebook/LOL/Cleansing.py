import numpy as np
import pandas as pd

import getpass
import random
import string

import requests
import datetime
import time

region = 'https://na1.api.riotgames.com'
active_date = datetime.datetime.today() - datetime.timedelta(days = 7)
number_of_data = 10000
sleep_time = 0.05
username_length = 10

api_key = getpass.getpass('Please input your API key: ')

total_match_ids = {}
account_names = {}
data = []

print('Start Crawling API')

while len(data) < number_of_data:

    username = ''.join(random.choices(string.ascii_letters + string.digits, k=np.random.randint(1, username_length + 1, 1)[0]))
    while account_names.get(username) is not None:
        initial_username = ''.join(random.choices(string.ascii_letters + string.digits, k=3))
    account_names[username] = True

    url = region + '/lol/summoner/v4/summoners/by-name/' + username
    params = {'api_key': api_key}
    response = requests.get(url, params=params)
    code = response.status_code
    result = response.json()
    if code != 200:
        time.sleep(sleep_time)
        continue
    else:
        active_time = result['revisionDate']

    if active_time <= active_date.timestamp() * 1e3:
        time.sleep(sleep_time)
        continue
    else:
        url = region + '/lol/match/v4/matchlists/by-account/' + result['accountId']
        params = {'api_key': api_key,
                  'queue': 430,
                  'season': 13}
        response = requests.get(url, params)
        code = response.status_code
        result = response.json()
        if code != 200:
            time.sleep(sleep_time)
            continue
        else:
            for match in result['matches']:
                match_id = match['gameId']
                if total_match_ids.get(match_id) is None:
                    url = region + '/lol/match/v4/matches/' + str(match_id)
                    params = {'api_key': api_key}
                    response = requests.get(url, params)
                    code = response.status_code
                    result = response.json()
                    if code != 200:
                        time.sleep(sleep_time)
                        continue
                    else:
                        for team in result['teams']:
                            datum = [team['teamId'], team['win'], team['firstBlood'], team['firstTower'],
                                     team['firstInhibitor'],
                                     team['firstBaron'], team['firstDragon'], team['firstRiftHerald'],
                                     team['towerKills'], team['inhibitorKills'], team['baronKills'],
                                     team['dragonKills'], team['vilemawKills'], team['riftHeraldKills']]
                            data.append(datum)
                            if len(data) % 1000 == 0 and len(data) > 0:
                                print('There are {} data in total'.format(len(data)))
                                print('')
                        time.sleep(0.05)
                else:
                    total_match_ids[match_id] = True

print('Finish crawling')

data = pd.DataFrame(data, columns = ['Side', 'Win', 'First_Blood', 'First Tower', 'First_Inhibitor',
                                     'First_Baron', 'First_Dragon', 'First_Rift_Herald', 'Tower_Destroyed',
                                     'Inhibitor_Destroyed', 'Baron_Killed', 'Dragon_Killed', 'Vilemaw_Killed',
                                     'Rift_Herald_Killed'])
data['Side'] = ['Blue' if x == 100 else 'Red' for x in data['Side']]
data['Win'] = [1 if x == 'Win' else 0 for x in data['Win']]
data.iloc[:, 2:8] = data.iloc[:, 2:8]*1

