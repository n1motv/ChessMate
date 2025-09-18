import pyautogui
from vision import square_to_xy

def play_move(src:str, dst:str):
    x1,y1 = square_to_xy(src); x2,y2 = square_to_xy(dst)
    pyautogui.moveTo(x1,y1, duration=0.05); pyautogui.click()
    pyautogui.moveTo(x2,y2, duration=0.05); pyautogui.click()

def highlight_move(src:str, dst:str):
    x1,y1 = square_to_xy(src); x2,y2 = square_to_xy(dst)
    pyautogui.moveTo(x1,y1, duration=0.05)
    pyautogui.moveTo(x2,y2, duration=0.05)
