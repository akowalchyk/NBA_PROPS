import requests
import json
import pandas as pd
from bs4 import BeautifulSoup
import time
from datetime import datetime
import pickle

class Player:
    def __init__(self, name, url, team, pos):
        self.name = name
        self.url = url
        self.team = team
        self.pos = pos

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
             rows.append(row)
             #print()
        df = pd.DataFrame(rows, columns=col_names)
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
        for tr in trs:
            time.sleep(0.25)
            a = tr.find('a')
            if a:
                url = a['href'][0:-5]
                url_counts[url] = 0
                name = a.get_text()
                team = tr.find("td", attrs={"data-stat": "team_id"}).get_text()
                pos = tr.find("td", attrs={"data-stat": "pos"}).get_text()
                p = Player(url=url, name=name, pos=pos, team=team)
                players.append(p)
        
        for p in players:
            url = p.url
            if url_counts[url] == 0:
                url_counts[url] += 1
                final_players.append(p)

        with open('players.pkl', 'wb') as file:
            pickle.dump(final_players, file)

    
    return final_players

def get_one_players_stats(players):
    p = players[1]
    player_url = "https://www.basketball-reference.com" + '/players/t/thybuma01' + "/gamelog/2023"
    print(player_url)

    df = grab_players_game_logs(player_url)
    if not df.empty():
        player_name = [p.name] * len(df)
        pos = [p.pos] * len(df)
        df["name"] = player_name
        df["pos"] = pos
        df.to_csv("Steven_Adams.csv", index=False)

def get_all_players_stats(players):
    players_df = pd.DataFrame()
    for p in players:
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
    players_df.to_csv("player_data.csv", index=False)
    return players_df

def clean_data():
    df = pd.read_csv("player_data.csv")

    #removing non-games
    df = df[df['Date'].notna()]

    #removing unneeded columns
    df.drop(columns=['Age','FG%', '3P%','FT', 'FTA', 'PF', 'ORB', 'FT%','DRB', 'TRB','AST', 'STL', 'BLK','TOV', '\xa0.1'], axis=1, inplace=True)
    df.rename(columns={'\xa0': 'away'}, inplace=True)

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
        df['last_2_game_MP_diff'] = df['MP'].rolling(window=2).apply(lambda x: x.iloc[1] - x.iloc[0]).shift()
        df['last_4_game_MP_diff'] = df['MP'].rolling(window=4).apply(lambda x: (x.iloc[3] - x.iloc[2]) +  (x.iloc[2] - x.iloc[1]) + (x.iloc[1] - x.iloc[0])).shift()

        # points #
        df['last_PTS'] = df['PTS'].shift()
        df['last_3_game_PTS_avg'] = df['PTS'].rolling(3).mean().shift()
        df['last_2_game_PTS_diff'] = df['PTS'].rolling(window=2).apply(lambda x: x.iloc[1] - x.iloc[0]).shift()
        df['last_4_game_PTS_diff'] = df['PTS'].rolling(window=4).apply(lambda x: (x.iloc[3] - x.iloc[2]) +  (x.iloc[2] - x.iloc[1]) + (x.iloc[1] - x.iloc[0])).shift()

        # total attempts #
        df['last_TA'] = df['TA'].shift()
        df['last_3_game_TA_avg'] = df['TA'].rolling(3).mean().shift()
        df['last_2_game_TA_diff'] = df['TA'].rolling(window=2).apply(lambda x: x.iloc[1] - x.iloc[0]).shift()
        df['last_4_game_TA_diff'] = df['TA'].rolling(window=4).apply(lambda x: (x.iloc[3] - x.iloc[2]) +  (x.iloc[2] - x.iloc[1]) + (x.iloc[1] - x.iloc[0])).shift()

        # Game Score #
        df['last_GmSc'] = df['GmSc'].shift()
        df['last_3_game_GmSc_avg'] = df['GmSc'].rolling(3).mean().shift()
        df['last_2_game_GmSc_diff'] = df['GmSc'].rolling(window=2).apply(lambda x: x.iloc[1] - x.iloc[0]).shift()
        df['last_4_game_GmSc_diff'] = df['GmSc'].rolling(window=4).apply(lambda x: (x.iloc[3] - x.iloc[2]) +  (x.iloc[2] - x.iloc[1]) + (x.iloc[1] - x.iloc[0])).shift()
        
        # +/- #
        df['last_+/-'] = df['+/-'].shift()
        df['last_3_game_+/-_avg'] = df['+/-'].rolling(3).mean().shift()
        df['last_2_game_+/-_diff'] = df['+/-'].rolling(window=2).apply(lambda x: x.iloc[1] - x.iloc[0]).shift()
        df['last_4_game_+/-_diff'] = df['+/-'].rolling(window=4).apply(lambda x: (x.iloc[3] - x.iloc[2]) +  (x.iloc[2] - x.iloc[1]) + (x.iloc[1] - x.iloc[0])).shift()

        # GS #
        df['last_GS'] = df['GS'].shift()
        df['last_3_game_GS_avg'] = df['GS'].rolling(3).mean().shift()
        df['last_6_game_GS_avg'] = df['GS'].rolling(4).mean().shift()

        # games_played # 
        df['games_played'] = df.played.cumsum().shift()
        
        df = df.iloc[3:]

        df = df[df['played'] == 1]

        final_df = pd.concat([final_df, df])

    final_df.drop(columns=['G', 'Date', 'GS', 'MP', 'FG','FGA','3P','3PA','GmSc','+/-','player_name', 'FT', 'FTA', 'ORB', 'PF','played', 'TA'], axis=1, inplace=True)

    final_df.to_csv("player_data_final.csv", index=False)


def extract_bets_raw():
    lines = []
    players = []
    with open('bets_raw_2.txt', 'r') as f:
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
                    curr_line = lines[i]
                    player.append(curr_line)
                    i+= 1
                    curr_line = lines[i]
                    player.append(curr_line[0:3])
                    while 'Points' not in curr_line or 'Fantasy' in curr_line or '+' in curr_line:
                        i+= 1
                        curr_line = lines[i]
                    player.append(curr_line)
                    players.append(player)
                        
        i+= 1
    
    
    df1 = pd.read_csv("factorized.csv")
    grouped = df1.groupby(['player_id'])
    player_dfs = []
    for name, df in grouped:  
        player_dfs.append(df)
    
    with open('players.pkl', 'rb') as file:
            # Call load method to deserialze
            all_players = pickle.load(file)
            for p in all_players:
                name = p.name
                id = p.url
                pos = p.pos
                team = p.team
                for p_bet in players:
                    if p_bet[0] == name and p_bet[1] == team:
                        print(p_bet[0]) 
                        print(id)
    

    return players

players = extract_bets_raw()


#players = grab_players()
#get_all_players_stats(players)
#get_one_players_stats(players)
#clean_data()


#transform_data()

# df = pd.read_csv("player_data.csv")
# df.drop(columns=df.columns[0], axis=1, inplace=True)
# df.to_csv("player_data.csv", index=False)

#df = pd.read_csv("player_data.csv")
#print(df["GS"].unique())


#extract_by_player()



