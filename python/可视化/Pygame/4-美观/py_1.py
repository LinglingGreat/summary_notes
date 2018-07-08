import pygame
import sys
from pygame.locals import *

pygame.init()

size = width, height = 640, 480
bg = (255, 255, 255)

clock = pygame.time.Clock()
screen = pygame.display.set_mode(size)
pygame.display.set_caption("FishC Demo")

oturtle = pygame.image.load("turtle.png")
turtle = pygame.transform.chop(oturtle, (207, 200, 50, 50))
# turtle = oturtle
position = turtle.get_rect()
position.center = width // 2, height // 2

while True:
    for event in pygame.event.get():
        if event.type == QUIT:
            sys.exit()

    screen.fill(bg)
    screen.blit(turtle, position)
    pygame.draw.rect(screen, (0, 0, 0), position, 1)
    
    pygame.display.flip()
    
    clock.tick(30)
