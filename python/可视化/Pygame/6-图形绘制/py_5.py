import pygame
import sys
import math
from pygame.locals import *

pygame.init()

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)

size = width, height = 640, 300
screen = pygame.display.set_mode(size)
pygame.display.set_caption("FishC Demo")

clock = pygame.time.Clock()

while True:
    for event in pygame.event.get():
        if event.type == QUIT:
            sys.exit()

    screen.fill(WHITE)
    
    pygame.draw.arc(screen, BLACK, (100, 100, 440, 100), 0, math.pi, 1)
    pygame.draw.arc(screen, BLACK, (220, 50, 200, 200), math.pi, math.pi * 2, 1)

    pygame.display.flip()

    clock.tick(120)
