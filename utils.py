import requests
import json
import pandas as pd
from bs4 import BeautifulSoup
import time
from datetime import datetime
import pickle
from unidecode import unidecode
from datetime import date
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor


class Player:
    def __init__(self, name, url, team, pos, points, away):
        self.name = name
        self.url = url
        self.team = team
        self.pos = pos
        self.points = points
        self.away = away

class Player_Bet:
    def __init__(self,id, name, diff,underdog_points, model_points, actual_points, sign ):
        self.diff = diff
        self.name = name
        self.underdog_points = underdog_points
        self.model_points = model_points
        self.actual_point = actual_points
        self.id = id
        self.sign = sign


def grab_players_info():
    response = requests.get("https://basketball.realgm.com/nba/players")
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, "html.parser")
        #print(soup.get_text())
        table = soup.find("table", attrs={"class": "tablesaw"})
        #print(table)
        thead = table.find("thead")
        ##print(thead)
        ths = thead.find_all("th")
        col_names = [th.get_text() for th in ths]
        # # print(col_names)
        # # task: try to parse the tbody
        tbody = table.find("tbody")
        trs = tbody.find_all("tr")
        rows = []
        for tr in trs:
             row = []
             tds = tr.find_all("td")
             for td in tds:
                 row.append(td.get_text())
             rows.append(row)
        df = pd.DataFrame(rows, columns=col_names)
        df.to_csv("player_names.csv")
        return df
    return None # TODO: should do better error handling

def grab_players_game_logs(url):
    response = requests.get(url)
    if response.status_code == 200:
        print("hey")
        soup = BeautifulSoup(response.text, "html.parser")
        #print(soup.get_text())
        table = soup.find("table", attrs={"id": "pgl_basic"})
        #print(table)
        thead = table.find("thead")
        ##print(thead)
        ths = thead.find_all("th")
        col_names = [th.get_text() for th in ths]
        col_names = col_names[1:]
        # # print(col_names)
        # # task: try to parse the tbody
        tbody = table.find("tbody")
        trs = tbody.find_all("tr")
        rows = []
        for tr in trs:
             row = []
             tds = tr.find_all("td")
             for td in tds:
                 row.append(td.get_text())
             ##print(len(row))
             rows.append(row)
             #print()
        df = pd.DataFrame(rows, columns=col_names)
        df = df.drop(['Date', 'Age'], axis=1)
        print(df.columns)
        return df
    return None # TODO: should do better error handling



def grab_players():
    players = []
    final_players = []
    url_counts = {}
    response = requests.get("https://www.basketball-reference.com/leagues/NBA_2023_per_game.html")
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, "html.parser")
        #print(soup.get_text())
        table = soup.find("table", attrs={"id": "per_game_stats"})
        thead = table.find("thead")
        ##print(thead)
        ths = thead.find_all("th")
        col_names = [th.get_text() for th in ths]
        col_names = col_names[1:]
        # # print(col_names)
        # # task: try to parse the tbody
        tbody = table.find("tbody")
        trs = tbody.find_all("tr")
        count = 0
        for tr in trs:
            count += 1
            a = tr.find('a')
            if a:
                url = a['href'][0:-5]
                url_counts[url] = 0
                name = unidecode(a.get_text())
                team = tr.find("td", attrs={"data-stat": "team_id"}).get_text()
                if team != "TOT":
                    if team == "BRK":
                        team = "BKN"
                    elif team == "CHO":
                        team = "CHA"
                    elif team == "PHO":
                        team = "PHX"

                    if name == "Cameron Johnson":
                        name = "Cam Johnson"

                    if name == "Luguentz Dort":
                        name = "Lu Dort"
                    pos = tr.find("td", attrs={"data-stat": "pos"}).get_text()
                    p = Player(url=url, name=name, pos=pos, team=team, points=-1, away=-1)
                    players.append(p)
                    print(p.name)
                    print(p.team)

        with open('players_and_teams.pkl', 'wb') as file:
            pickle.dump(players, file)
        print(count)
        for p in players:
            url = p.url
            if url_counts[url] == 0:
                url_counts[url] += 1
                final_players.append(p)

        with open('players.pkl', 'wb') as file:
            pickle.dump(final_players, file)

    else:
        print("fail")
    return final_players

def get_one_players_stats(players):
    p = players[1]
    player_url = "https://www.basketball-reference.com" + '/players/c/curryst01' + "/gamelog/2023"
    print(player_url)

    df = grab_players_game_logs(player_url)
    print(df)
    # #if not df.empty():
    # player_name = [p.name] * len(df)
    # pos = [p.pos] * len(df)
    # df["name"] = player_name
    # df["pos"] = pos
    # df.to_csv("Steven_Adams.csv", index=False)

def get_all_players_stats():
    fileObj = open('players.pkl', 'rb')
    players = pickle.load(fileObj)
    print(len(players))
    players_df = pd.DataFrame()

    start = time.time()
    
    for i,p in enumerate(players):
        if i % 50 == 0:
            print(i)
        time.sleep(2)
        player_name = []
        player_url = "https://www.basketball-reference.com" + p.url + "/gamelog/2023"
        print(player_url)

        df = grab_players_game_logs(player_url)
        if df is not None:
            player_name = [p.name] * len(df)
            df["player_name"] = player_name

            player_url = [p.url] * len(df)
            df["player_url"] = player_url

            player_pos = [p.pos] * len(df)
            df["pos"] = player_pos

            players_df = pd.concat([players_df, df])
    
    end = time.time()
    print(end - start)
    players_df.to_csv("player_data.csv", index=False)
    return players_df

def clean_data():
    df = pd.read_csv("player_data.csv")

    #removing non-games
    df = df[df['Date'].notna()]

    #removing unneeded columns
    df.drop(columns=['Age','FG%', '3P%','FT', 'FTA', 'PF', 'ORB', 'FT%','DRB', 'TRB','AST', 'STL', 'BLK','TOV', '\xa0.1'], axis=1, inplace=True)
    df.rename(columns={'\xa0': 'away'}, inplace=True)

    df['+/-'].fillna(0, inplace=True)

    # changing home/away values to 0/1
    df['away'] = df['away'].notnull().astype("int")

    df.to_csv("player_data_clean.csv", index=False)
    time.sleep(4)

def get_ms(time_str):
    """Get seconds from time."""
    m, s = time_str.split(':')
    return (int(m) * 60 + int(s)) * 1000

def transform_data():
    df1 = pd.read_csv("player_data_clean.csv")
    print(df1.columns)
    grouped = df1.groupby(['player_url'])
    final_df = pd.DataFrame()
    for name, df in grouped:
        # add day of week
        df['Date'] = pd.to_datetime(df['Date'])
        df['day'] = df['Date'].dt.day_name()


        # inactive games
        df.loc[(df['GS'] == 'Inactive') | (df['GS'] == 'Did Not Play') |  (df['GS'] == 'Did Not Dress') | (df['GS'] == 'Player Suspended') | (df['GS'] == 'Not With Team'), 'MP'] = '00:00' 
        df.loc[(df['GS'] == 'Inactive') | (df['GS'] == 'Did Not Play') |  (df['GS'] == 'Did Not Dress') | (df['GS'] == 'Player Suspended') | (df['GS'] == 'Not With Team'), 'FG'] = '0' 
        df.loc[(df['GS'] == 'Inactive') | (df['GS'] == 'Did Not Play') |  (df['GS'] == 'Did Not Dress') | (df['GS'] == 'Player Suspended') | (df['GS'] == 'Not With Team'), 'FGA'] = '0'
        df.loc[(df['GS'] == 'Inactive') | (df['GS'] == 'Did Not Play') |  (df['GS'] == 'Did Not Dress') | (df['GS'] == 'Player Suspended') | (df['GS'] == 'Not With Team'), '3P'] = '0'
        df.loc[(df['GS'] == 'Inactive') | (df['GS'] == 'Did Not Play') |  (df['GS'] == 'Did Not Dress') | (df['GS'] == 'Player Suspended') | (df['GS'] == 'Not With Team'), '3PA'] = '0'
        df.loc[(df['GS'] == 'Inactive') | (df['GS'] == 'Did Not Play') |  (df['GS'] == 'Did Not Dress') | (df['GS'] == 'Player Suspended') | (df['GS'] == 'Not With Team'), 'FT'] = '0'
        df.loc[(df['GS'] == 'Inactive') | (df['GS'] == 'Did Not Play') |  (df['GS'] == 'Did Not Dress') | (df['GS'] == 'Player Suspended') | (df['GS'] == 'Not With Team'), 'FTA'] = '0'
        df.loc[(df['GS'] == 'Inactive') | (df['GS'] == 'Did Not Play') |  (df['GS'] == 'Did Not Dress') | (df['GS'] == 'Player Suspended') | (df['GS'] == 'Not With Team'), 'ORB'] = '0'
        df.loc[(df['GS'] == 'Inactive') | (df['GS'] == 'Did Not Play') |  (df['GS'] == 'Did Not Dress') | (df['GS'] == 'Player Suspended') | (df['GS'] == 'Not With Team'), 'PF'] = '0'
        df.loc[(df['GS'] == 'Inactive') | (df['GS'] == 'Did Not Play') |  (df['GS'] == 'Did Not Dress') | (df['GS'] == 'Player Suspended') | (df['GS'] == 'Not With Team'), 'PTS'] = '0'
        df.loc[(df['GS'] == 'Inactive') | (df['GS'] == 'Did Not Play') |  (df['GS'] == 'Did Not Dress') | (df['GS'] == 'Player Suspended') | (df['GS'] == 'Not With Team'), 'GmSc'] = '0'
        df.loc[(df['GS'] == 'Inactive') | (df['GS'] == 'Did Not Play') |  (df['GS'] == 'Did Not Dress') | (df['GS'] == 'Player Suspended') | (df['GS'] == 'Not With Team'), '+/-'] = '0'
        df.loc[(df['GS'] == 'Inactive') | (df['GS'] == 'Did Not Play') |  (df['GS'] == 'Did Not Dress') | (df['GS'] == 'Player Suspended') | (df['GS'] == 'Not With Team'), 'GS'] = '0'  

        df['TA'] = df['FGA'] + df['3PA']
        # changing to milliseconds
        df['MP'] = df['MP'].apply(get_ms)

        #inactive or active
        df['played'] = df['MP'].apply(lambda x: 1 if x > 0 else 0)
        
        
        # minutes #
        df['last_MP'] = df['MP'].shift()
        df['last_3_game_MP_avg'] = df['MP'].rolling(3).mean().shift()
        # df['last_2_game_MP_diff'] = df['MP'].rolling(window=2).apply(lambda x: x.iloc[1] - x.iloc[0]).shift()
        # df['last_4_game_MP_diff'] = df['MP'].rolling(window=4).apply(lambda x: (x.iloc[3] - x.iloc[2]) +  (x.iloc[2] - x.iloc[1]) + (x.iloc[1] - x.iloc[0])).shift()

        # points #
        df['last_PTS'] = df['PTS'].shift()
        df['last_3_game_PTS_avg'] = df['PTS'].rolling(3).mean().shift()
        # df['last_2_game_PTS_diff'] = df['PTS'].rolling(window=2).apply(lambda x: x.iloc[1] - x.iloc[0]).shift()
        # df['last_4_game_PTS_diff'] = df['PTS'].rolling(window=4).apply(lambda x: (x.iloc[3] - x.iloc[2]) +  (x.iloc[2] - x.iloc[1]) + (x.iloc[1] - x.iloc[0])).shift()

        # total attempts #
        df['last_TA'] = df['TA'].shift()
        df['last_3_game_TA_avg'] = df['TA'].rolling(3).mean().shift()
        # df['last_2_game_TA_diff'] = df['TA'].rolling(window=2).apply(lambda x: x.iloc[1] - x.iloc[0]).shift()
        # df['last_4_game_TA_diff'] = df['TA'].rolling(window=4).apply(lambda x: (x.iloc[3] - x.iloc[2]) +  (x.iloc[2] - x.iloc[1]) + (x.iloc[1] - x.iloc[0])).shift()

        # Game Score #
        df['last_GmSc'] = df['GmSc'].shift()
        df['last_3_game_GmSc_avg'] = df['GmSc'].rolling(3).mean().shift()
        # df['last_2_game_GmSc_diff'] = df['GmSc'].rolling(window=2).apply(lambda x: x.iloc[1] - x.iloc[0]).shift()
        # df['last_4_game_GmSc_diff'] = df['GmSc'].rolling(window=4).apply(lambda x: (x.iloc[3] - x.iloc[2]) +  (x.iloc[2] - x.iloc[1]) + (x.iloc[1] - x.iloc[0])).shift()
        
        # +/- #
        df['last_+/-'] = df['+/-'].shift()
        df['last_3_game_+/-_avg'] = df['+/-'].rolling(3).mean().shift()
        # df['last_2_game_+/-_diff'] = df['+/-'].rolling(window=2).apply(lambda x: x.iloc[1] - x.iloc[0]).shift()
        # df['last_4_game_+/-_diff'] = df['+/-'].rolling(window=4).apply(lambda x: (x.iloc[3] - x.iloc[2]) +  (x.iloc[2] - x.iloc[1]) + (x.iloc[1] - x.iloc[0])).shift()

        # GS #
        df['last_GS'] = df['GS'].shift()
        df['last_3_game_GS_avg'] = df['GS'].rolling(3).mean().shift()

        # games_played # 
        df['games_played'] = df.played.cumsum().shift()

        df['points_avg'] = df['PTS'].expanding().mean().shift()
        df['minutes_avg'] = df['MP'].expanding().mean().shift()
        

        df = df.iloc[3:]

        df = df[df['played'] == 1]

        final_df = pd.concat([final_df, df])

    final_df.drop(columns=['G', 'Date', 'GS', 'MP', 'FG','FGA','3P','3PA','GmSc','+/-','player_name', 'FT', 'FTA', 'ORB', 'PF','played', 'TA'], axis=1, inplace=True)

    final_df.to_csv("player_data_final.csv", index=False)


def extract_bets_raw():
    lines = []
    players = []
    id_to_points = {}
    players_final = []
    with open('bets_raw.txt', 'r') as f:
        for line in f:
            stripped_line = line.strip()
            if stripped_line:
                lines.append(stripped_line)
    
    i = 0
    size = len(lines)
    while i < size:
        curr_line = lines[i]
        if '@' in curr_line or 'vs' in curr_line:
            i+= 1
            if i < size:
                curr_line = lines[i]
                if '-' in curr_line:
                    player = []
                    i -= 2
                    # player name
                    curr_line = lines[i]
                    player.append(curr_line)
                    i+= 1

                    # player team opponent and away
                    curr_line = lines[i]
                    words = curr_line.split()

                    #team
                    player.append(words[0])

                    #away
                    player.append(words[1])

                    #opp
                    player.append(words[2])

                    while 'Points' not in curr_line or 'Fantasy' in curr_line or '+' in curr_line:
                        i+= 1
                        curr_line = lines[i]

                    words = curr_line.split()
                    player.append(words[0])
                    players.append(player)
                    
        i+= 1
    
    # for p in players:
    #     for item in p:
    #         print(item)
    #     print()
    # print(len(players))
    
    
    with open('players_and_teams.pkl', 'rb') as file:
        with open('somefile.txt', 'a') as the_file:
            # Call load method to deserialze
            all_players = pickle.load(file)
            for p in all_players:
                # if name == "Kyrie Irving":
                #     print(name)
                #     print(team)
                name = p.name
                id = p.url
                team = p.team
                #print(team)
                # the_file.write(name)
                # the_file.write(team)
                for p_bet in players:
                    if (name in p_bet[0] or p_bet[0] in name) and p_bet[1] == team:
                        #print(p.name)
                        p.points = p_bet[4]
                        if p_bet[2] == "vs":
                            p.away = 0
                        else:
                            p.away = 1
                        p.opp = p_bet[3]
                        players_final.append(p)


    return players_final

def pred_bets(players):
    x_vals = []
    import warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)
    df_final = pd.read_csv("player_data_final.csv")
    df1 = pd.read_csv("player_data_clean.csv")
    grouped = df1.groupby(['player_url'])
    print(len(players))
    for p in players:

        for i, df in grouped:  
            df_col = df["player_url"]
            df_id = df_col.iloc[0]
            if df_id == p.url:
                if len(df) >= 4:

                    # print(id)
                    # print(points)

                    s = pd.Series(index=df_final.columns)
                    s["player_url"] = df_id

                    dt = datetime.now()
                    s["day"] = dt.strftime('%A')
                    s["away"] = p.away
                    s["Opp"] = p.opp
                    s["Tm"] = p.team
                    s["PTS"] = p.points


                    # inactive games
                    df.loc[(df['GS'] == 'Inactive') | (df['GS'] == 'Did Not Play') |  (df['GS'] == 'Did Not Dress') | (df['GS'] == 'Player Suspended') | (df['GS'] == 'Not With Team'), 'MP'] = '00:00' 
                    df.loc[(df['GS'] == 'Inactive') | (df['GS'] == 'Did Not Play') |  (df['GS'] == 'Did Not Dress') | (df['GS'] == 'Player Suspended') | (df['GS'] == 'Not With Team'), 'FG'] = '0' 
                    df.loc[(df['GS'] == 'Inactive') | (df['GS'] == 'Did Not Play') |  (df['GS'] == 'Did Not Dress') | (df['GS'] == 'Player Suspended') | (df['GS'] == 'Not With Team'), 'FGA'] = '0'
                    df.loc[(df['GS'] == 'Inactive') | (df['GS'] == 'Did Not Play') |  (df['GS'] == 'Did Not Dress') | (df['GS'] == 'Player Suspended') | (df['GS'] == 'Not With Team'), '3P'] = '0'
                    df.loc[(df['GS'] == 'Inactive') | (df['GS'] == 'Did Not Play') |  (df['GS'] == 'Did Not Dress') | (df['GS'] == 'Player Suspended') | (df['GS'] == 'Not With Team'), '3PA'] = '0'
                    df.loc[(df['GS'] == 'Inactive') | (df['GS'] == 'Did Not Play') |  (df['GS'] == 'Did Not Dress') | (df['GS'] == 'Player Suspended') | (df['GS'] == 'Not With Team'), 'FT'] = '0'
                    df.loc[(df['GS'] == 'Inactive') | (df['GS'] == 'Did Not Play') |  (df['GS'] == 'Did Not Dress') | (df['GS'] == 'Player Suspended') | (df['GS'] == 'Not With Team'), 'FTA'] = '0'
                    df.loc[(df['GS'] == 'Inactive') | (df['GS'] == 'Did Not Play') |  (df['GS'] == 'Did Not Dress') | (df['GS'] == 'Player Suspended') | (df['GS'] == 'Not With Team'), 'ORB'] = '0'
                    df.loc[(df['GS'] == 'Inactive') | (df['GS'] == 'Did Not Play') |  (df['GS'] == 'Did Not Dress') | (df['GS'] == 'Player Suspended') | (df['GS'] == 'Not With Team'), 'PF'] = '0'
                    df.loc[(df['GS'] == 'Inactive') | (df['GS'] == 'Did Not Play') |  (df['GS'] == 'Did Not Dress') | (df['GS'] == 'Player Suspended') | (df['GS'] == 'Not With Team'), 'PTS'] = '0'
                    df.loc[(df['GS'] == 'Inactive') | (df['GS'] == 'Did Not Play') |  (df['GS'] == 'Did Not Dress') | (df['GS'] == 'Player Suspended') | (df['GS'] == 'Not With Team'), 'GmSc'] = '0'
                    df.loc[(df['GS'] == 'Inactive') | (df['GS'] == 'Did Not Play') |  (df['GS'] == 'Did Not Dress') | (df['GS'] == 'Player Suspended') | (df['GS'] == 'Not With Team'), '+/-'] = '0'
                    df.loc[(df['GS'] == 'Inactive') | (df['GS'] == 'Did Not Play') |  (df['GS'] == 'Did Not Dress') | (df['GS'] == 'Player Suspended') | (df['GS'] == 'Not With Team'), 'GS'] = '0'  

                    df['TA'] = df['FGA'] + df['3PA']
                    # changing to milliseconds
                    df['MP'] = df['MP'].apply(get_ms)

                    #inactive or active
                    df['played'] = df['MP'].apply(lambda x: 1 if x > 0 else 0)
                    
                    df = df.append(s, ignore_index=True)


                    # minutes #
                    df['last_MP'] = df['MP'].shift()
                    df['last_3_game_MP_avg'] = df['MP'].rolling(3).mean().shift()
                    # df['last_2_game_MP_diff'] = df['MP'].rolling(window=2).apply(lambda x: x.iloc[1] - x.iloc[0]).shift()
                    # df['last_4_game_MP_diff'] = df['MP'].rolling(window=4).apply(lambda x: (x.iloc[3] - x.iloc[2]) +  (x.iloc[2] - x.iloc[1]) + (x.iloc[1] - x.iloc[0])).shift()

                    # points #
                    df['last_PTS'] = df['PTS'].shift()
                    df['last_3_game_PTS_avg'] = df['PTS'].rolling(3).mean().shift()
                    # df['last_2_game_PTS_diff'] = df['PTS'].rolling(window=2).apply(lambda x: x.iloc[1] - x.iloc[0]).shift()
                    # df['last_4_game_PTS_diff'] = df['PTS'].rolling(window=4).apply(lambda x: (x.iloc[3] - x.iloc[2]) +  (x.iloc[2] - x.iloc[1]) + (x.iloc[1] - x.iloc[0])).shift()

                    # total attempts #
                    df['last_TA'] = df['TA'].shift()
                    df['last_3_game_TA_avg'] = df['TA'].rolling(3).mean().shift()
                    # df['last_2_game_TA_diff'] = df['TA'].rolling(window=2).apply(lambda x: x.iloc[1] - x.iloc[0]).shift()
                    # df['last_4_game_TA_diff'] = df['TA'].rolling(window=4).apply(lambda x: (x.iloc[3] - x.iloc[2]) +  (x.iloc[2] - x.iloc[1]) + (x.iloc[1] - x.iloc[0])).shift()

                    # Game Score #
                    df['last_GmSc'] = df['GmSc'].shift()
                    df['last_3_game_GmSc_avg'] = df['GmSc'].rolling(3).mean().shift()
                    # df['last_2_game_GmSc_diff'] = df['GmSc'].rolling(window=2).apply(lambda x: x.iloc[1] - x.iloc[0]).shift()
                    # df['last_4_game_GmSc_diff'] = df['GmSc'].rolling(window=4).apply(lambda x: (x.iloc[3] - x.iloc[2]) +  (x.iloc[2] - x.iloc[1]) + (x.iloc[1] - x.iloc[0])).shift()
                    
                    # +/- #
                    df['last_+/-'] = df['+/-'].shift()
                    df['last_3_game_+/-_avg'] = df['+/-'].rolling(3).mean().shift()
                    # df['last_2_game_+/-_diff'] = df['+/-'].rolling(window=2).apply(lambda x: x.iloc[1] - x.iloc[0]).shift()
                    # df['last_4_game_+/-_diff'] = df['+/-'].rolling(window=4).apply(lambda x: (x.iloc[3] - x.iloc[2]) +  (x.iloc[2] - x.iloc[1]) + (x.iloc[1] - x.iloc[0])).shift()

                    # GS #
                    df['last_GS'] = df['GS'].shift()
                    df['last_3_game_GS_avg'] = df['GS'].rolling(3).mean().shift()

                    # games_played # 
                    df['games_played'] = df.played.cumsum().shift()

                    df['points_avg'] = df['PTS'].expanding().mean()
                    df['minutes_avg'] = df['MP'].expanding().mean()

                    df["pos"] = df["pos"].shift()

                    s = df.iloc[-1]
                    s = s.dropna()
                    df_final = pd.DataFrame()
                    df_final = pd.read_csv("player_data_final.csv")
                    df_final = df_final.append(s, ignore_index=True)

                    le = LabelEncoder()
                    df_final['Tm'] = le.fit_transform(df_final['Tm'])
                    df_final['Opp'] = le.fit_transform(df_final['Opp'])
                    df_final['player_url'] = le.fit_transform(df_final['player_url'])
                    df_final['pos'] = le.fit_transform(df_final['pos'])
                    df_final['day'] = le.fit_transform(df_final['day'])
                    s = df_final.iloc[-1]
                    s = s.dropna()
                    s["name"] = p.name
                    s["id"] = df_id
                    if df_id == "/players/b/branhma01":
                        print(s)
                    #print(s)
                    x_vals.append(s)

                        
                    #df.iloc[-1].to_csv("test_data.csv", index=False)
    print(len(x_vals))
    fileObj = open('x_vals.pkl', 'wb')
    pickle.dump(x_vals,fileObj)
    fileObj.close()
    return x_vals

def evaluate(model,X_train,X_test,y_train,y_test,X,y):    

    # fitting data and saving model    
    model.fit(X_train, y_train)
    fileObj = open('model.pkl', 'wb')
    pickle.dump(model,fileObj)
    fileObj.close()

    pred = model.predict(X_test)

    score = r2_score(y_test, pred) 
    print("R^2 score:", score)
    return model
    # # generating scatter plot for residuals
    # df = pd.DataFrame(columns=['points', 'predictions', 'residuals'])
    # df.attendance = y
    # df.predictions = model.predict(X)
    # df.residuals = df.attendance - df.predictions
    # df.plot(x='points', y='points', kind='scatter')
    # plt.show()

def train():    
    df =  pd.read_csv("player_data_final.csv")
    df = df.dropna()

    le = LabelEncoder()
    df['Tm'] = le.fit_transform(df['Tm'])
    df['Opp'] = le.fit_transform(df['Opp'])
    df['player_url'] = le.fit_transform(df['player_url'])
    df['pos'] = le.fit_transform(df['pos'])
    df['day'] = le.fit_transform(df['day'])
    
    all_cols = df.columns
    pred_col = ['PTS']
    remove_cols = ["PTS"]
    cols = list(set(all_cols) - set(pred_col))

    X = df[cols]
    print(X.columns)
    y = df['PTS']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    print(X_test.iloc[0])

    model = RandomForestRegressor(n_estimators=200, max_features=3)
    model = evaluate(model,X_train,X_test,y_train,y_test,X,y)
    return model

def pred(model, x_vals):
    props = []
    names = []
    ids = []
    x_tests = []
  
    for x in x_vals:
        props.append(x['PTS'])
        names.append(x['name'])
        ids.append(x['id'])
        new_x = x.drop(labels=['PTS','name','id'])
        print(x['name'])
        x_tests.append((new_x))

    # x_test = [x_tests[2]]
    # print(x_test)
    # print(names[2])
    # fileObj = open('model.pkl', 'rb')
    # model = pickle.load(fileObj)
    
    preds = model.predict(x_tests)

    l = []
    for i,v in enumerate(preds):
        diff = abs(float(props[i]) -  v)
        act = 0
        act = v - float(props[i]) 
        sign = ''
        if act > 0:
            sign = '+'
        else:
            sign = '-'
        #print(x_tests[i])
        # print(names[i])
        name = names[i]
        id = ids[i]
        # print("Underdog Points: ", props[i])
        underdog_points = float(props[i])
        model_points = v
        p = Player_Bet(id=id,diff=diff, name=name, underdog_points=underdog_points, model_points=model_points, actual_points=-1, sign=sign)
        l.append(p)

    df = pd.DataFrame(columns=['Name', 'Underdog Points', 'Model Points', 'Differential', 'Pos/Neg'])
    l = sorted(l, key=lambda x: x.diff, reverse=True)
    for x in l:
        data = { 'Name':x.name, 'Underdog Points':x.underdog_points, 'Model Points':x.model_points, 'Differential':x.diff, 'Pos/Neg':x.sign}
        df = df.append(data, ignore_index=True)
        #print(x.name," Diff:", x.diff, " Underdog Points", x.underdog_points, " Model Points:", x.model_points, x.id)
        #print()
    print(df)
    df.to_csv("todays_bets.csv", index=False)
    fileObj = open('bet_stats.pkl', 'wb')
    pickle.dump(l,fileObj)
    fileObj.close()

def playoff_stuff():
    # Make a GET request to the URL
    url = "https://www.basketball-reference.com/players/j/jamesle01/gamelog-playoffs/"
    response = requests.get(url)

    # Parse the HTML content using Beautiful Soup
    soup = BeautifulSoup(response.content, 'html.parser')

    table = soup.find("table", attrs={"id": "pgl_basic_playoffs"})
    thead = table.find("thead")
    ##print(thead)
    ths = thead.find_all("th")
    col_names = [th.get_text() for th in ths]
    col_names = col_names[1:]

    # Find all the td elements containing '2023' in the data-stat="date_game" attribute
    trs = table.find_all('tr')
    # Loop through the td elements and find the ones containing '2023'
    good_trs = []
    for tr in trs :
        tds = tr.findAll('td')
        for td in tds:
            if '2023' in td.text:
                good_trs.append(tr)

    rows = []
    rows = []
    for tr in good_trs:
        row = []
        tds = tr.find_all("td")
        for td in tds:
            row.append(td.get_text())
        #print(len(row))
        rows.append(row)
        #print()
    df = pd.DataFrame(rows, columns=col_names)
    df = df.drop(1, axis=1)
    print(df)
    print(df.columns)


#def compare_bets():

#get_one_players_stats("players")



#print(len(x_vals))
# for bet in x_vals:
#     print(bet)


#grab_players()



#get_all_players_stats()

#clean_data()

#transform_data()

ps = extract_bets_raw()
x_vals = pred_bets(ps)
model = train()
pred(model, x_vals)



