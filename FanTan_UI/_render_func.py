from setup import SHORT_PATH
from PIL import Image, ImageFont, ImageDraw
IMG_PATH = SHORT_PATH + "Base/FanTan_UI/Cards_Image/"
BG_SIZE = (2100, 900)
CARD_SIZE = (100, 130)

import numpy as np

from Base.FanTan_UI import _env
from PIL import Image, ImageEnhance

def get_description(action):
    card_values = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"]
    card_suits = ["Heart", "Diamond", "Club", "Spade"]
    if action == 52:
        return "Skip"
    else:
        value = action % 13
        suit = action // 13
        return f'{card_values[value]}-{card_suits[suit]}'
def get_state_image(state):
    background = Image.open(IMG_PATH + 'bg.png')
    draw = ImageDraw.Draw(background)
    font = ImageFont.FreeTypeFont("ImageFonts/arial.ttf", 40)
    if state is None:
        return background
    else:
        # generate my cards
        my_cards = state[0:52]
        my_cards = np.where(my_cards == 1)[0]
        my_cards_value = my_cards % 13 + 1
        my_cards_type = my_cards // 13
        for i in range(len(my_cards)):
            card_img = Image.open(IMG_PATH + f'{my_cards_value[i] }-{my_cards_type[i]}.png').resize(CARD_SIZE)
            background.paste(card_img, (round(BG_SIZE[0] * 0.44 + i * 18 - len(my_cards) * 3),round(BG_SIZE[1]* 0.76 + 25)))
        # generate my chip 
        my_chip = int(state[104])
        draw.text((round(BG_SIZE[0] * 0.53),round(BG_SIZE[1]* 0.69 )),str(my_chip),(255,255,255),font)
        # generate another player cards
        another_player_cards_len = state[105:108]
        # generate player 1 card:
        cards_len_1 = int(another_player_cards_len[0])
        for i in range(cards_len_1):
            card_img = Image.open(IMG_PATH + 'Card_back.png').resize(CARD_SIZE)
            background.paste(card_img, (round(BG_SIZE[0] * 0.18+ i * 18 ),round(BG_SIZE[1]* 0.2 + 20 )))
        # generate player 1 chip
        player1_chip = int(state[109])
        draw.text((round(BG_SIZE[0] * 0.23 ),round(BG_SIZE[1]* 0.38 )),str(player1_chip),(255,255,255),font)
        # generate player 2 card:
        card_len_2 = int(another_player_cards_len[1])
        for i in range(card_len_2 ):
            card_img = Image.open(IMG_PATH + 'Card_back.png').resize(CARD_SIZE)
            background.paste(card_img, (round(BG_SIZE[0] * 0.75 + i * 18 - card_len_2 * 11),round(BG_SIZE[1]* 0.2 + 20 )))
        # generate player 2 chip
        player2_chip = int(state[110])
        draw.text((round(BG_SIZE[0] * 0.52 ),round(BG_SIZE[1]* 0.25 )),str(player2_chip),(255,255,255),font)
        # generate player 3 cards:
        card_len_3= int(another_player_cards_len[2])
        for i in range(card_len_3):
            card_img = Image.open(IMG_PATH + 'Card_back.png').resize(CARD_SIZE)
            background.paste(card_img, (round(BG_SIZE[0] * 0.44 - i * 18 + card_len_3 * 10),round(BG_SIZE[1]* 0.06 + 25)))
        # generate player 3 chip
        player3_chip = int(state[110])
        draw.text((round(BG_SIZE[0] * 0.74 ),round(BG_SIZE[1]* 0.38 )),str(player3_chip),(255,255,255),font)
        # generate card on board
        cards_can_play = state[52:104]
        cards_can_play = np.where(cards_can_play == 1)[0]
        value_cards_can_play = cards_can_play % 13 + 1
        type_cards_can_play = cards_can_play // 13
        for i in range(len(cards_can_play) ):
            card_img = Image.open(IMG_PATH + f'{value_cards_can_play[i] }-{type_cards_can_play[i]}.png').resize(CARD_SIZE)
            background.paste(card_img, (round(BG_SIZE[0] * 0.5+ i * CARD_SIZE[0] - len(cards_can_play) * 40),round(BG_SIZE[1]* 0.45)))
    return background

class Env_components:
    def __init__(self) -> None:
        self.allGame = True
        self.saveStoreChip = np.array([50,50,50,50])
        self.idxPlayerChip =  np.array([21,35,49,63])

        self.env = _env.initEnv()
        self.env[self.idxPlayerChip] = self.saveStoreChip
        self.oneGame = True
        self.i = 0
        self.list_other = np.array([-1, 1, 2, 3])
        np.random.shuffle(self.list_other)

def get_env_components():
    return Env_components()

def step_env(com: Env_components, action, list_agent, list_data):
    stepEnvReturn = _env.stepEnv(action, com.env)
    if stepEnvReturn == -1:
        com.oneGame = False
        com.env[8+com.i*14+13] += com.env[65]
        com.saveStoreChip = com.env[com.idxPlayerChip]
        com.env[65] = 0
    elif stepEnvReturn == -2:
        com.env[66] = 1
        player_chip = com.env[com.idxPlayerChip]
        player_id_not_0_chip = np.where(player_chip > 0)[0]
        arr_player_cards = np.zeros(13*3)
        for i in range(len(player_id_not_0_chip)):
            player_cards = com.env[8+ player_id_not_0_chip[i] * 13:8+player_id_not_0_chip[i]*13+13]
            arr_player_cards[i*13:i*13+13] = player_cards.astype(np.float64)

        arr_player_cards = np.reshape(arr_player_cards,(3,13))
        player_card_len = np.array([len(np.where(player_cards > -1)) for player_cards in arr_player_cards])
        player_lowest_card = np.argmax(player_card_len)
        player_lowest_card_id = player_id_not_0_chip[player_lowest_card]
        com.env[com.idxPlayerChip[player_lowest_card_id]] += com.env[65]
        com.env[65] = 0

        env = com.env.copy()
        for pIdx in range(4):
            env[64] = pIdx
            state = _env.getAgentState(env)
            if com.list_other[pIdx] == -1:
                _state_ = state.copy()
                if _env.getReward(state) == 1:
                    win = 1
                else:
                    win = 0
            else:
                agent = list_agent[com.list_other[pIdx]-1]
                data = list_data[com.list_other[pIdx]-1]
                action, data = agent(state, data)

        com.allGame = False
        return win, _state_

    return -1, None

def get_main_player_state(env_components: Env_components, list_agent, list_data, action=None):
    if not action is None:
        win, state = step_env(env_components, action, list_agent, list_data)
        if win != -1:
            return win, state, env_components

        env_components.i += 1

    while True:
        if env_components.i == 4:
            if not env_components.oneGame:
                env_components.env = _env.initEnv()
                env_components.env[env_components.idxPlayerChip] = env_components.saveStoreChip
                env_components.oneGame = True

            env_components.i = 0

        env_components.env[64] = env_components.i
        state = _env.getAgentState(env_components.env)
        if env_components.list_other[env_components.i] == -1:
            return -1, state, env_components

        agent = list_agent[env_components.list_other[env_components.i]-1]
        data = list_data[env_components.list_other[env_components.i]-1]
        action, data = agent(state, data)
        win, state = step_env(env_components, action, list_agent, list_data)
        if win != -1:
            return win, state, env_components

        env_components.i += 1