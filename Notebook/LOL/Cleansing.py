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
number_of_data = 10
sleep_time = 1
username_length = 10

api_key = getpass.getpass('Please input your API key: ')

total_match_ids = {}
account_names = {}
data = []

print('Start Crawling API')

while len(data) < number_of_data:

    username = ''.join(random.choices(string.ascii_letters + string.digits, k=np.random.randint(3, username_length + 1, 1)[0]))
    while account_names.get(username) is not None:
        initial_username = ''.join(random.choices(string.ascii_letters + string.digits, k=3))
    account_names[username] = True

    url = region + '/lol/summoner/v4/summoners/by-name/' + username
    params = {'api_key': api_key}
    response = requests.get(url, params=params)
    code = response.status_code
    result = response.json()
    if code != 200:
        # print(result)
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
            # print(result)
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
                        # print(result)
                        time.sleep(sleep_time)
                        continue
                    else:
                        for team in result['teams']:
                            total_kills = 0
                            total_deaths = 0
                            total_assists = 0
                            total_magic_damage_dealt = 0
                            total_physical_damage_dealt = 0
                            total_true_damage_dealt = 0
                            total_magic_damage_to_champions = 0
                            total_physical_damage_to_champions = 0
                            total_true_damage_to_champions = 0
                            total_heal = 0
                            total_damage_self_mitigated = 0
                            total_damage_dealt_to_objectives = 0
                            total_damage_dealt_to_turrets = 0
                            total_vision_score = 0
                            total_time_ccing_others = 0
                            total_magical_damage_taken = 0
                            total_physical_damage_taken = 0
                            total_true_damage_taken = 0
                            total_gold_earned = 0
                            total_minions_killed = 0
                            total_neutral_minions_killed = 0
                            total_neutral_minions_killed_team_jungle = 0
                            total_neutral_minions_killed_enemy_jungle = 0
                            total_time_crowd_control_dealt = 0
                            total_champion_level = 0

                            for participant in result['participants']:
                                if participant['teamId'] == team['teamId']:
                                    participant = participant['stats']

                                    total_kills += participant['kills']
                                    total_deaths += participant['deaths']
                                    total_assists += participant['assists']
                                    total_magic_damage_dealt += participant['magicDamageDealt']
                                    total_physical_damage_dealt += participant['physicalDamageDealt']
                                    total_true_damage_dealt += participant['trueDamageDealt']
                                    total_magic_damage_to_champions += participant['magicDamageDealtToChampions']
                                    total_physical_damage_to_champions += participant['physicalDamageDealtToChampions']
                                    total_true_damage_to_champions += participant['trueDamageDealtToChampions']
                                    total_heal += participant['totalHeal']
                                    total_damage_self_mitigated += participant['damageSelfMitigated']
                                    total_damage_dealt_to_objectives += participant['damageDealtToObjectives']
                                    total_damage_dealt_to_turrets += participant['damageDealtToTurrets']
                                    total_vision_score += participant['visionScore']
                                    total_time_ccing_others += participant['timeCCingOthers']
                                    total_magical_damage_taken += participant['magicalDamageTaken']
                                    total_physical_damage_taken += participant['physicalDamageTaken']
                                    total_true_damage_taken += participant['trueDamageTaken']
                                    total_gold_earned += participant['goldEarned']
                                    total_minions_killed += participant['totalMinionsKilled']
                                    total_neutral_minions_killed += participant['neutralMinionsKilled']
                                    total_neutral_minions_killed_team_jungle += participant['neutralMinionsKilledTeamJungle']
                                    total_neutral_minions_killed_enemy_jungle += participant['neutralMinionsKilledEnemyJungle']
                                    total_time_crowd_control_dealt += participant['totalTimeCrowdControlDealt']
                                    total_champion_level += participant['champLevel']

                            datum = [team['teamId'], team['win'], team['firstBlood'], team['firstTower'],
                                     team['firstInhibitor'], team['firstBaron'], team['firstDragon'],
                                     team['firstRiftHerald'], team['towerKills'], team['inhibitorKills'],
                                     team['baronKills'], team['dragonKills'], team['vilemawKills'],
                                     team['riftHeraldKills'], result['gameDuration'], total_kills, total_deaths,
                                     total_assists, total_magic_damage_dealt, total_physical_damage_dealt,
                                     total_true_damage_dealt, total_magic_damage_to_champions, total_physical_damage_to_champions,
                                     total_true_damage_to_champions, total_heal, total_damage_self_mitigated,
                                     total_damage_dealt_to_objectives, total_damage_dealt_to_turrets, total_vision_score,
                                     total_time_ccing_others, total_magical_damage_taken, total_physical_damage_taken,
                                     total_true_damage_taken, total_gold_earned, total_minions_killed,
                                     total_neutral_minions_killed, total_neutral_minions_killed_team_jungle,
                                     total_neutral_minions_killed_enemy_jungle, total_time_crowd_control_dealt,
                                     total_champion_level]

                            data.append(datum)
                            if len(data) % 100 == 0 and len(data) > 0:
                                print('There are {} data in total'.format(len(data)))
                                print('')
                        time.sleep(sleep_time)
                else:
                    total_match_ids[match_id] = True

print('Finish crawling')

data = pd.DataFrame(data, columns = ['Side', 'Win', 'First_Blood', 'First Tower', 'First_Inhibitor',
                                     'First_Baron', 'First_Dragon', 'First_Rift_Herald', 'Tower_Destroyed',
                                     'Inhibitor_Destroyed', 'Baron_Killed', 'Dragon_Killed', 'Vilemaw_Killed',
                                     'Rift_Herald_Killed', 'Game_Duration', 'Total_Kills', 'Total_Deaths',
                                     'Total_Assists', 'Total_Magic_Damage_Dealt', 'Total_Physical_Damage_Dealt',
                                     'Total_True_Damage_Dealt', 'Total_Magic_Damage_To_Champions',
                                     'Total_Physical_Damage_To_Champions', 'Total_True_Damage_To_Champions',
                                     'Total_Heal', 'Total_Damage_Self_Mitigated', 'Total_Damage_Dealt_To_Objectives',
                                     'Total_Damage_Dealt_To_Turrets', 'Total_Vision_Score', 'Total_Time_CCing_Others',
                                     'Total_Magical_Damage_Taken', 'Total_Physical_Damage_Taken',
                                     'Total_True_Damage_Taken', 'Total_Gold_Earned', 'Total_Minions_Killed',
                                     'Total_Neutral_Minions_Killed', 'Total_Neutral_Minions_Killed_Team_Jungle',
                                     'Total_Neutral_Minions_Killed_Enemy_Jungle', 'Total_Time_Crowd_Control_Dealt',
                                     'Total_Champion_Level'])
data['Side'] = ['Blue' if x == 100 else 'Red' for x in data['Side']]
data['Win'] = [1 if x == 'Win' else 0 for x in data['Win']]
data.iloc[:, 2:8] = data.iloc[:, 2:8]*1

data.to_csv('../../Data/LOL/lol_matches.csv')

