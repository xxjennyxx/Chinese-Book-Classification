from selenium.common.exceptions import NoSuchElementException, WebDriverException
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium import webdriver
from bs4 import BeautifulSoup
from webdriver_manager.chrome import ChromeDriverManager
from urllib.parse import urlparse, parse_qs

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

from ArticutAPI import Articut
import pandas as pd
import requests
import time
import json
import re

with open("/Users/jennyyang/Documents/CCU/111-2/PythonAndNaturalLanguageProcessing/ArticutAccount/account.info", 'r', encoding='utf8') as f:
    account = json.load(f)
username = account['username']
apikey = account['api_key']
articut = Articut(username, apikey)

# 找博客來連結
def search_book_in_google(title):
    search_query = f"博客來 {title}"
    url = f"https://www.google.com/search?q={search_query}"
    response = requests.get(url)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, 'html.parser')

    link = soup.find('a', href=lambda href: href and 'books.com.tw' in href)
    if link:
        parsed_url = urlparse(link['href'])
        params = parse_qs(parsed_url.query)
        for key in params.keys():
            if 'q' in key:
                for value in params[key]:
                    if "https://www.books.com.tw/products/" in value:
                        return value
    return None

import unicodedata 
def is_chinese(text):
    for char in text:
        if unicodedata.category(char) != 'Lo':
            return False
    return True

exit = 0
while (exit == 0):
    print("您想搜尋哪本中文書？")
    print("若您要提供書名，輸入1")
    print("若您要提供書籍介紹，輸入2")
    print("若您要提供博客來連結，輸入3")
    print("若您要離開，輸入4")

    information = None 
    title = None
    url = None

    while (True):
        option = input()

        if option == "4":
            exit = 1
            break

        if (option == "1" or option == "3"):
            if (option == "1"):
                while True:
                    title = input("請輸入書名（中文）：")
                    if is_chinese(title):
                        break
                    else:
                        print("不好意思，您輸入不是中文，請重新輸入。")
                        continue
                    
                # 找博客來連結
                url = search_book_in_google(title)
                if not url:
                    print("抱歉，我找不到此本書籍介紹，請您換一本。\n")
                    print("您想搜尋哪本中文書？")
                    print("若您要提供書名，輸入1")
                    print("若您要提供書籍介紹，輸入2")
                    print("若您要提供博客來連結，輸入3")
                    print("若您要離開，輸入4")
                    continue

            elif (option == "3"):
                while True:
                    url = input("請輸入博客來連結：")
                    if "https://www.books.com.tw/products/" in url:
                        break
                    else:
                        print("不好意思，您輸入的好像不是博客來連結，請重新輸入。")
                        continue
            
            print("正在為您搜尋書籍中，請稍候...")
            # 找書籍介紹

            options = Options()
            options.add_argument("--disable-notifications")
            options.add_argument("--headless") 

            service = webdriver.chrome.service.Service(ChromeDriverManager().install())
            chrome = webdriver.Chrome(service=service, options=options)

            # if url:
            chrome.get(url)
            time.sleep(3)

            try:
                content_div = chrome.find_element(By.CLASS_NAME, "content")
                information = content_div.text.strip()
                chrome.quit()
                break
            except (NoSuchElementException, WebDriverException):
                chrome.quit()
                print("抱歉，我找不到此本書籍介紹，請您換一本")
                continue

        elif option == "2":
            print("請輸入書籍介紹：")
            print("輸入完成後請按enter，接著按 Ctrl-D (mac) 或 Ctrl-Z (windows)")
            contents = []
            while True:
                try:
                    line = input()
                except EOFError:
                    break
                contents.append(line)
            information = str(contents)
            break

        else:
            print("\n請輸入1、2或3: (1: 提供書名, 2: 提供書籍介紹, 3: 提供博客來連結)")

    if option == "2":
        print("\n\n")

    if option == "4":
        break

    print("正在為您分析中，請稍候...")
    
    # 新書
    newParse = articut.parse(information)
    newSeg = newParse['result_segmentation']
    newSeg = re.sub('/',' ', newSeg)

    # 把新的斷詞加到原來訓練的語料的pd.DataFrame中
    df = pd.read_csv('book_segmented_information_Chinese_books.csv', encoding='utf-8')
    df.dropna(subset=['書籍資訊斷詞'], inplace=True) 
    newBookDF = pd.DataFrame({"書籍資訊":[information], "博客來連結":url, "書籍資訊斷詞":newSeg, "適讀對象":["成人"], "articut.parse": str(newParse), "書名": title}) # 先把新的兩則和舊的df合併再一起，這樣bow才會有每個字，隨意設定label
    evaDataDF = pd.concat([df, newBookDF], axis=0) 

    # 把上面的concatData轉成BOW
    vecArticut = CountVectorizer(token_pattern=r'(?u)\b\w+\b')
    newX = vecArticut.fit_transform(evaDataDF['書籍資訊斷詞']).toarray()
    newY = evaDataDF['適讀對象']
    X_train, X_test, y_train, y_test = train_test_split(newX[:-1], newY[:-1], test_size=0.2, random_state=10) 

    # 訓練
    nb = MultinomialNB()
    nb.fit(X_train, y_train)

    # 預測
    print("\n")
    if option == "1": 
        print(f"{title}的試讀對象是{nb.predict([newX[-1]])[0]}\n")

    elif option == "3" or option == "2":
        print(f"此本書的試讀對象是{nb.predict([newX[-1]])[0]}\n")
    
    print("-----------------------------------")
    
    if option == "1" or option == "3":
        print("若您想查看此書的介紹，輸入1")
        print("若您要搜尋其他書籍，輸入2")
        print("若您要離開，請隨意輸入")
        choice = input()
        if (choice == "1"):
            print(f"以下是書籍介紹:\n{information}")
        elif (choice == "2"):
            continue
        else:
            exit = 1
    elif option == "2":
        print("若您要搜尋其他書籍，輸入1")
        print("若您要離開，請隨意輸入")
        choice = input()
        if (choice == "1"):
            continue
        else:
            exit = 1
    if exit != 1:
        print("\n\n-----------------------------------")

