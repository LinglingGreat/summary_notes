import pygame
import sys
from pygame.locals import *

pygame.init()

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)

points = [(200, 75), (300, 25), (400, 75), (450, 25), (450, 125), (400, 75), (300, 125)]

size = width, height = 640, 480
screen = pygame.display.set_mode(size)
pygame.display.set_caption("FishC Demo")

clock = pygame.time.Clock()

while True:
    for event in pygame.event.get():
        if event.type == QUIT:
            sys.exit()

    screen.fill(WHITE)
    
    pygame.draw.lines(screen, GREEN, 1, points, 1)
    pygame.draw.line(screen, BLACK, (100, 200), (540, 250), 1)
    pygame.draw.aaline(screen, BLACK, (100, 250), (540, 300), 1)
    pygame.draw.aaline(screen, BLACK, (100, 300), (540, 350), 0)

    pygame.display.flip()

    clock.tick(10)
