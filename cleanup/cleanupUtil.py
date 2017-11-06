#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  1 12:50:31 2017

@author: tante, simon
"""



Fields_ORIGIN = ["_id","appId","lastAppInfoCrawlTimestamp","name","linkName","price",
          "starRating","category","badge","author","totalNrOfReviews","reviewsPerStarRating",
          "description","whatsNew","lastUpdated","size","installs","currentVersion","requiredAndroidVersion",
          "contentRating","permissions","inAppProducts","appUrl","similarApps", "lastAppBrainCrawlTimestamp",
          "ranking", "binarySize", "libraries", "age", "commentsTag", "resourcePermissions"]

# non numeric features in dataframe which have to be kicked out of train data
NON_NUMERIC_FEATURES = ['appId', 'similarApps']

DUMMY_LIST = ['category','contentRating','requiredAndroidVersion_d1','requiredAndroidVersion_d2',
              'requiredAndroidVersion_d3','ranking', 'offersInAppPurchases', 'installs']

NUMERIC_TO_BE_CATEGORIZED = ['libraries', 'totalNrOfReviews', 'binarySize']

# if price be treated as a range category for classification
# (0, 0.99]:1   (0.99, 2.99]:2  ... 
PRICE_RANGE = {0.99: 1, 1.99: 2, 3.99: 3, "above": 4}

# category indexes, app without category will be given an index of 0
CATEGORY_INDEX = {"Art & Design": 1, "Auto & Vehicles": 2, "Beauty": 3, "Android Wear": 4, "Books & Reference": 5, "Business": 6,
                  "Comics": 7, "Communication": 8, "Dating": 9, "Education": 10, "Entertainment": 11, "Events": 12, "Finance": 13,
                  "Food & Drink": 14, "Health & Fitness": 15, "House & Home":16, "Libraries & Demo": 17, "Lifestyle": 18,
                  "Maps & Navigation": 19, "Medical": 20, "Music & Audio": 21, "News & Magazines": 22, "Parenting": 23, 
                  "Personalization": 24, "Photography": 25, "Productivity": 26, "Shopping": 27, "Social": 28, "Sports":29,
                  "Tools": 30, "Travel & Local": 31, "Video Players & Editors": 32, "Weather": 33, "Game": 34}

# Game subcategories
GAME_SUBCATEGORY = ["Action", "Adventure", "Arcade", "Board", "Card", "Casual", "Educational",
                  "Puzzle", "Racing", "Role Playing", "Simulation", "Sports", "Strategy"]

# subcategory indexes
SUBCATEGORY_INDEX = {"Game_Action": 35, "Game_Adventure": 36, "Game_Arcade": 37, "Game_Board": 38, "Game_Card": 39, "Game_Casino": 40,
                     "Game_Casual": 41, "Game_Educational": 42, "Game_Music": 43, "Game_Puzzle": 44, "Game_Racing": 45,
                     "Game_Role Playing": 46, "Game_Simulation": 47, "Game_Sports": 48, "Game_Strategy": 49, "Game_Trivia": 50, "Game_Word": 51}

# text feature
TEXT_FEATURES_FOR_NEIGHBOR = ['name', 'author']
TEXT_FEATURES = ['name', 'author', 'description', 'whatsNew', 'permissions', 'commentsTag', 'resourcePermissions', 'userComments']

# similar apps price statistics
SIMILAR_APPS_STATS = ['similarApps_count', 'similarApps_mean', 'similarApps_std', 'similarApps_min',
                             'similarApps_25%', 'similarApps_50%', 'similarApps_75%', 'similarApps_max']

