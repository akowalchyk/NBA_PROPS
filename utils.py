import requests
import json
import pandas as pd
from bs4 import BeautifulSoup
import time

class Player:
    def __init__(self, name, url, pos):
        self.name = name
        self.url = url
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
            a = tr.find('a')
            if a:
                url = a['href'][0:-5]
                name = a.get_text()
                pos = tr.find("td", attrs={"data-stat": "pos"}).get_text()
                p = Player(url=url, name=name, pos=pos)
                players.append(p)
    return players

def get_one_players_stats(players):
    p = players[1]
    player_url = "https://www.basketball-reference.com" + '/players/t/thybuma01' + "/gamelog/2023"
    print(player_url)

    df = grab_players_game_logs(player_url)

    player_name = [p.name] * len(df)
    pos = [p.pos] * len(df)
    df["name"] = player_name
    df["pos"] = pos
    df.to_csv("Steven_Adams.csv", index=False)

def get_all_players_stats(player_urls):
    players_df = pd.DataFrame()
    for name, url in player_urls.items():
        time.sleep(2)
        player_name = []
        player_url = "https://www.basketball-reference.com" + url + "/gamelog/2023"
        print(player_url)

        df = grab_players_game_logs(player_url)

        player_name = [name] * len(df)
        df["player name"] = player_name
        players_df = pd.concat([players_df, df])
        players_df.to_csv("player_data.csv")
    return players_df
def clean_data():
    df = pd.read_csv("Steven_Adams.csv")

    #removing non-games
    df = df[df['Date'].notna()]

    #removing unneeded columns
    df.drop(columns=['Age','FG%', '3P%', 'FT%','DRB', 'TRB','AST', 'STL', 'BLK','TOV', '\xa0.1'], axis=1, inplace=True)
    df.rename(columns={'\xa0': 'away'}, inplace=True)

    # changing home/away values to 0/1
    df['away'] = df['away'].notnull().astype("int")

    df.to_csv("adams_results.csv", index=False)

def transform_data():
    df = pd.read_csv("adams_results.csv")

    # add day of week
    df['Date'] = pd.to_datetime(df['Date'])
    df['day'] = df['Date'].dt.day_name()

    # inactive games
    df.loc[(df['GS'] == 'Inactive') | (df['GS'] == 'Did Not Play') | (df['GS'] == 'Did Not Dress'), 'MP'] = '00:00' 
    # df.loc[df['GS'] == 'Inactive', 'FG'] = '0' 
    # df.loc[df['GS'] == 'Inactive', 'FGA'] = '0'
    # df.loc[df['GS'] == 'Inactive', '3P'] = '0'
    # df.loc[df['GS'] == 'Inactive', '3PA'] = '0'
    # df.loc[df['GS'] == 'Inactive', 'FT'] = '0'
    # df.loc[df['GS'] == 'Inactive', 'FTA'] = '0'
    # df.loc[df['GS'] == 'Inactive', 'ORB'] = '0'
    # df.loc[df['GS'] == 'Inactive', 'PF'] = '0'
    # df.loc[df['GS'] == 'Inactive', 'PTS'] = '0'
    # df.loc[df['GS'] == 'Inactive', 'GmSc'] = '0'
    # df.loc[df['GS'] == 'Inactive', '+/-'] = '0'
    df.loc[(df['GS'] == 'Inactive') | (df['GS'] == 'Did Not Play') | (df['GS'] == 'Did Not Dress'), 'GS'] = '0'  



    #minutes played to minutes 
    df['MP'] = df['MP'].str.replace(':','.')
    df['MP'] = df['MP'].astype(float)
    
    #inactive or active
    df['played'] = df['MP'].apply(lambda x: '1' if x > 0 else '0')

    # past minutes avg
    df['3_game_MP_avg'] = df['MP'].rolling(3).mean().shift()
    df['last_MP'] = df['MP'].rolling(1).mean().shift()

    df['2_game_MP_diff'] = df['MP'].rolling(window=2).apply(lambda x: x.iloc[1] - x.iloc[0])
    df['4_game_MP_diff'] = df['MP'].rolling(window=4).apply(lambda x: (x.iloc[3] - x.iloc[2]) +  (x.iloc[2] - x.iloc[1]) + (x.iloc[1] - x.iloc[0]))

    

    df.to_csv("adams_trans.csv", index=False)


players = grab_players()
get_one_players_stats(players)
clean_data()

transform_data()

#df = pd.read_csv("player_data.csv")
#print(df["GS"].unique())



