import cv2 
from Snake import SnakeGame 

delay = 0

game = SnakeGame()
game.reset()

while True:
    frame = game.get_frame()
    cv2.imshow("frame", frame)
    key = cv2.waitKey(delay)
    action = 0
    if key == ord('a'):
        action = 1
    if key == ord('d'):
        action = 2
    next_state, reward, done, score = game.step(action)
    print(next_state)
    print(reward)
    print(score)
    if done:
        exit()