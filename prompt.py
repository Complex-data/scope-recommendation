import time

import openai
import requests

ChatGPT_Selecting_item_Prompt = [
    {
        'role': 'system',
        'content': "A good movie recommender is always able to select the most suitable movie for users. The movie selected by recommender is always comlying with user's preference and intention. Below we define three evaluation axes for the selected movie: Consistency of preference, Consistency of intention and Overall quality.\n"
                   "1. Consistency of preference: This axis answers the question about the selected movie \"how consistent is the selected movie comparing with user's preference?\". The selected movie should be as consistent as possible with the user's preference. User's preference will be presented in \"#User's Preference#\" or can be inferred from his historical watching list presented in \"#User's Historical Watching list#\". If \"#User's Preference#\" and \"#User's Historical Watching list#\" is not exist, you can ignore this axis."
                   "2. Consistency of intention: This axis answers the question about the selected movie \"how consistent is the selected movie comparing with user's intention?\". The selected movie should be as consistent as possible with the user's intention. User's intention will be presented in #User's Intention#. If \"#User's Intention#\" is not exist, you can ignore this axis.\n"
                   "3. Overall quality: This axis answers the overall quality of the selected movie. Ideally, the selected movie should keep as consistent as possible with the user's preference and intention simultaneously if this information exists. But, there are some situations where the user's intention is conflict with his preference, what you consider firstly is user's intention. You can also think that the user's intention is more important than his preference.\n"
                   "You are an expert movie recommender. Given user's information(in \"#User's Historical Watching list#\", \"#User's Preference#\" or \"#User's Intention#\") and candidate movies(in \"#Candidate Movies#\"), output the movie title in \"#Selected Movie#\" to indicate which movie best adheres to Consistency of preference, Consistency of intention and Overall quality as defined above.\n"
    },
    {
        'role': 'user',
        'content': "#User's Intention#: \n"
                   "\"I'm craving a heartwarming romantic story with a touch of humor. Can you suggest any feel-good romantic comedies?\"\n"
                   "#Candidate Movies#: \n"
                   "['Another Harvest Moon', 'Demon City Shinjuku VHS', 'Da: Burn Fat Fast Cardio Blast', 'American Gigolo', 'Red Scorpion VHS']\n"
                   "#Selected Movie#: \n"
    },
    {
        'role': 'assistant',
        'content': "'American Gigolo'"
    },
    {
        'role': 'user',
        'content': "{information}\n"
                   "#Selected Movie#: \n"
    },
]

ChatGPT_Ranking_item_Prompt = [
    {
        'role': 'system',
        'content': "A good movie recommender is always able to select the most suitable movie for users. The movie selected by recommender is always comlying with user's preference and intention. Below we define three evaluation axes for the selected movie: Consistency of preference, Consistency of intention and Overall quality.\n"
                   "1. Consistency of preference: This axis answers the question about the selected movie \"how consistent is the selected movie comparing with user's preference?\". The selected movie should be as consistent as possible with the user's preference. User's preference will be presented in \"#User's Preference#\" or can be inferred from his historical watching list presented in \"#User's Historical Watching list#\". If \"#User's Preference#\" and \"#User's Historical Watching list#\" is not exist, you can ignore this axis."
                   "2. Consistency of intention: This axis answers the question about the selected movie \"how consistent is the selected movie comparing with user's intention?\". The selected movie should be as consistent as possible with the user's intention. User's intention will be presented in #User's Intention#. If \"#User's Intention#\" is not exist, you can ignore this axis.\n"
                   "3. Overall quality: This axis answers the overall quality of the selected movie. Ideally, the selected movie should keep as consistent as possible with the user's preference and intention simultaneously if this information exists. But, there are some situations where the user's intention is conflict with his preference, what you consider firstly is user's intention. You can also think that the user's intention is more important than his preference.\n"
                   "You are an expert movie recommender. Given user's information(in \"#User's Historical Watching list#\", \"#User's Preference#\" or \"#User's Intention#\") and candidate movies(in \"#Candidate Movies#\"), output the movie title in \"#Selected Movie#\" to indicate which movie best adheres to Consistency of preference, Consistency of intention and Overall quality as defined above.\n"
    },
    {
        'role': 'user',
        'content': "#User's Intention#: \n"
                   "\"I'm craving a heartwarming romantic story with a touch of humor. Can you suggest any feel-good romantic comedies?\"\n"
                   "#Candidate Movies#: \n"
                   "['Another Harvest Moon', 'Demon City Shinjuku VHS', 'Da: Burn Fat Fast Cardio Blast', 'American Gigolo', 'Red Scorpion VHS']\n"
                   "#Selected Movie#: \n"
    },
    {
        'role': 'assistant',
        'content': "'American Gigolo'"
    },
    {
        'role': 'user',
        'content': "{information}\n"
                   "#Selected Movie#: \n"
    },
]


ChatGPT_Ranking_item_COT_Prompt = [
    {
        'role': 'system',
        'content': "A good movie recommender is always able to select the most suitable movie for users. The movie selected by recommender is always comlying with user's preference and intention. Below we define three evaluation axes for the selected movie: Consistency of preference, Consistency of intention and Overall quality.\n\n"
                   "1. Consistency of preference: This axis answers the question about the selected movie \"how consistent is the selected movie comparing with user's preference?\". The selected movie should be as consistent as possible with the user's preference. User's preference can be inferred from his historical watching list presented in \"#User's Historical Watching list#\". If \"#User's Historical Watching list#\" is not exist, you can ignore this axis.\n"
                   "2. Consistency of intention: This axis answers the question about the selected movie \"how consistent is the selected movie comparing with user's intention?\". The selected movie should be as consistent as possible with the user's intention. User's intention will be presented in \"#User's Intention#\". If \"#User's Intention#\" is not exist, you can ignore this axis.\n"
                   "3. Overall quality: This axis answers the overall quality of the selected movie. Ideally, the selected movie should keep as consistent as possible with the user's preference and intention simultaneously if this information exists. But, there are some situations where the user's intention is conflict with his preference, what you consider firstly is user's intention. You can also think that the user's intention is more important than his preference.\n\n"
                   "You are an expert movie recommender. Given user's information(in \"#User's Preference#\" or \"#User's Intention#\") and candidate movie list(in \"#Candidate Movie List#\"), you need to sort the movie title into \"#Sorted Movie List#\" to indicate the rank of movies from the best one adhering to Consistency of preference, Consistency of intention and Overall quality as defined above to the worst one. If there is no movie complying user's preference or intention, you need to select the movie closer to these axes.\n\n"
                   "You need to do the following step by step: \n"
                   "First step, you need to summarize the content of each movie or TV.\n"
                   "Second step, you need to consider the consistency of preference, consistency of intention and overall quality of each movie.\n"
                   "Third step, explain which movie is better.\n"
                   "Fourth step, you need to output the sorted movie list in this template \"<Sorted Movie List>[selected movie1, ... selected movie5]</Sorted Movie List>\".\n"
    },
    # {
    #     'role': 'user',
    #     'content': "#User's Intention#: \n"
    #                "\"I'm craving a heartwarming romantic story with a touch of humor. Can you suggest any feel-good romantic comedies?\"\n"
    #                "#Candidate Movies#: \n"
    #                "['Another Harvest Moon', 'Demon City Shinjuku VHS', 'Da: Burn Fat Fast Cardio Blast', 'American Gigolo', 'Red Scorpion VHS']\n\n"
    # },
    # {
    #     'role': 'assistant',
    #     'content': "Certainly! Let's go step by step through the process of evaluating and sorting the movies based on the user's preference and intention.\n\n"
    #                "Step 1: Summarize the content of each movie\n"
    #                "1. Anoher Harvest Moon: A drama that follows the story of a group of seniors in a nursing home as they face challenges and reminisce about their lives.\n"
    #                "2. Demon City Shinjuku VHS: An anime movie set in a post-apocalyptic Tokyo, where a young man must battle demonic forces.\n"
    #                "3. Da: Burn Fat Fast Cardio Blast: A workout video focusing on cardio exercises to aid in weight loss and fitness.\n"
    #                "4. American Gigolo: A crime drama featuring a male escort in Los Angeles who becomes the prime suspect in a murder investigation.\n"
    #                "5. Red Scorpion VHS: An action-packed movie following a Soviet Special Forces agent on a mission in Africa.\n\n"
    #                "Step 2: Evaluate the movies based on preference, intention, and overall quality\n"
    #                "Now, let's assess each movie based on the three evaluation axes: Consistency of preference, Consistency of intention, and Overall quality.\n"
    #                "Movie Evaluation:\n"
    #                "Another Harvest Moon:\n"
    #                "Consistency of preference: Low (Not a romantic comedy)\n"
    #                "Consistency of intention: Low (Not a heartwarming romantic story with humor)\n"
    #                "Overall quality: Moderate\n"
    #                "Demon City Shinjuku VHS:\n"
    #                "Consistency of preference: Low (Not a romantic comedy)\n"
    #                "Consistency of intention: Low (Not heartwarming or humorous)\n"
    #                "Overall quality: Low\n"
    #                "Da: Burn Fat Fast Cardio Blast:\n"
    #                "Consistency of preference: Low (Not related to the preference)\n"
    #                "Consistency of intention: Low (Not related to the intention)\n"
    #                "Overall quality: Low\n"
    #                "American Gigolo:\n"
    #                "Consistency of preference: Low (Not a romantic comedy)\n"
    #                "Consistency of intention: Low (Not heartwarming or humorous)\n"
    #                "Overall quality: Moderate\n"
    #                "Red Scorpion VHS:\n"
    #                "Consistency of preference: Low (Not a romantic comedy)\n"
    #                "Consistency of intention: Low (Not heartwarming or humorous)\n"
    #                "Overall quality: Low\n\n"
    #                "Step 3: Determine which movie is better\n"
    #                "Given the user's intention for a heartwarming romantic story with a touch of humor, we would prioritize movies that align with this intention.\n"
    #                "Another Harvest Moon ranks the highest in terms of overall quality and is the closest to the user's preference for a heartwarming story, although it lacks the comedic element.\n\n"
    #                "Step 4: Output the sorted movie list\n"
    #                "<Sorted Movie List>['Another Harvest Moon', 'American Gigolo, Red Scorpion VHS', 'Demon City Shinjuku VHS', 'Da: Burn Fat Fast Cardio Blast']</Sorted Movie List>"
    #                # "#Sorted Movie List#: ['Another Harvest Moon', 'American Gigolo', 'Demon City Shinjuku VHS', 'Red Scorpion VHS', 'Da: Burn Fat Fast Cardio Blast']"
    # },
    {
        'role': 'user',
        'content': "{information}\n"
    },
]


ChatGPT_Select_list_COT_Prompt = [
    {
        'role': 'system',
        'content': "A good movie recommender is always able to evaluate a movie list appropriately. The movie list with a higher rank is always closer to the user's intention than the lower one. \n"
                   "You are an expert movie recommender. You will be provided with {list_count} movie lists (delimited with XML tags <Movie List i>) and 1 user's intention (delimited with XML tag <User's Intention>). \n"
                   "Then, you need to compare these movie lists, and select one movie list which is closer to user's intention.\n"

    },
    {
        'role': 'user',
        'content': "<User's Intention> \"I'm craving a heartwarming romantic story with a touch of humor. Can you suggest any feel-good romantic comedies? But the proportion of them don't be more than 40% on the recommendation list\" </User's Intention>\n"
                   "<Movie List 1> [\"Another Harvest Moon\", \"Demon City Shinjuku VHS\", \"Da: Burn Fat Fast Cardio Blast\", \"American Gigolo\", \"Red Scorpion VHS\"] </Movie List 1>\n"
                   "<Movie List 2> [\"The History of WWE: 50 Years of Sports Entertainment\", \"Knight's Tale\", \"In Paris\", \"Morlang\", \"Wonder Boys\"] </Movie List 2>\n"
                   "\n"
                   "You need to do the following step by step: \n"
                   "Step 1 - Summarize each movie of each list with few words.\n"
                   "Step 2 - Output the movie list you think is closer to the user's intention in this template \"<Selected List> List 1 or 2 </Selected List>\"."
    },
    {
        'role': 'assistant',
        'content': "Step 1 - Summarize each movie of each list with a few words:\n"
                   "<Movie List 1>:\n"
                   "Another Harvest Moon - A heartfelt drama about aging and friendship.\n"
                   "Demon City Shinjuku VHS - An animated fantasy film with dark themes.\n"
                   "Da: Burn Fat Fast Cardio Blast - A fitness video.\n"
                   "American Gigolo - A classic romantic drama.\n"
                   "Red Scorpion VHS - An action movie with VHS format.\n"
                   "<Movie List 2>:\n"
                   "The History of WWE: 50 Years of Sports Entertainment - A documentary about WWE wrestling.\n"
                   "Knight's Tale - A medieval romantic comedy with a mix of humor.\n"
                   "In Paris - A romantic drama set in the city of love.\n"
                   "Morlang - A psychological drama with mystery elements.\n"
                   "Wonder Boys - A comedy-drama about an English professor.\n"
                   "\n"
                   "Step 2 - Output the movie list closer to the user's intention:\n"
                   "Based on the user's intention for a heartwarming romantic story with a touch of humor and a maximum of 40% romantic comedies, List 2 seems closer to the user's request. It contains \"Knight's Tale,\" which is a romantic comedy with humor, and other diverse movie options, while List 1 does not have any movies that align well with the user's criteria.\n"
                   "<Selected List> List 2 </Selected List>\n"
    },
    {
        'role': 'user',
        'content': "<User's Intention> {user_intention} </User's Intention>\n"
                   "<Movie List 1> {movie_list_1} </Movie List 1>\n"
                   "<Movie List 2> {movie_list_2} </Movie List 2>\n"
                   "\n"
                   "You need to do the following step by step: \n"
                   "Step 1 - Summarize each movie of each list with few words.\n"
                   "Step 2 - Output the movie list you think is closer to the user's intention in this template \"<Selected List> List 1 or 2 </Selected List>\"."
    },
]


fake_apis = [
    'fk-FenB34h670Zez0vIzPevo8Jt_vXpALkfbbjkHmgZZLw',
    'fk-SnLOCp9ImlwggLUe19i1i-oMqBtPNxupgfJHmPcwpPE',
    'fk-T6q_T_Qt6n3y1HUj785F1kgv5E5fa-Xpk1CEDLC_5Xs',
]


def generate_openai(messages, top_p, temperature):
    openai.api_key = "sk-E9oyiDL777ZaNZdRrzRSPzsbvbqvhebRl2xiTheKjh6bE4Jx"
    openai.proxy = "http://172.31.225.67:12621"
    openai.api_base = "https://openkey.cloud/v1"
    while True:
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                top_p=top_p,
                temperature=temperature,
                messages=messages,
            )
            return response['choices'][0]['message']['content']
        except:
            print(messages)


def generate_fake(message, top_p, temperature):
    global fake_apis
    error_count = 0
    while True:
        try:
            response = requests.post('https://ai.fakeopen.com/v1/chat/completions',
                                     json={
                                         "model": "gpt-3.5-turbo",
                                         "temperature": temperature,
                                         "top_p": top_p,
                                         "messages": message
                                     },
                                     headers={
                                         'Authorization': f'Bearer {fake_apis[0]}',
                                         'Content-Type': 'application/json'
                                     })
            return response.json()['choices'][0]['message']['content']
        except:
            fake_apis = fake_apis[1:] + fake_apis[0:1]
            error_count += 1
            if error_count % len(fake_apis) == 0:
                print('sleep 60 min')
                time.sleep(60 * 60)
