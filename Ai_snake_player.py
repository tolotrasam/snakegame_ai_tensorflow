from Snake_game import Game as game
import pygame
from pygame.locals import *

env = game()
env.reset()
action = -1
while True:

    for event in pygame.event.get():

        if event.type == KEYDOWN:
            if event.key == K_UP:
                action = 0

            elif event.key == K_DOWN:
                action = 1

            elif event.key == K_LEFT:
                action = 2

            elif event.key == K_RIGHT:
                action = 3

    # do it! render the previous view
    env.render()
    done = env.step(action)
    if done:
        break;