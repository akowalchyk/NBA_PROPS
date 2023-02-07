import requests
import json
import pandas as pd
from bs4 import BeautifulSoup
import time

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

def grab_players_urls():
    player_urls = {}
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
        for a in tbody.find_all('a'):
            link = a.get('href')
            if "players" in link:
                name = a.get_text()
                player_urls[name] = link[0:-5]
        
    return player_urls

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
player_urls = grab_players_urls()
players_df = get_all_players_stats(player_urls)



