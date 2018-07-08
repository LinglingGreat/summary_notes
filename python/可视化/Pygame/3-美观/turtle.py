import pygame
import sys
from pygame.locals import *

pygame.init()

pygame.mixer.init()
pygame.mixer.music.load("music.ogg")
pygame.mixer.music.set_volume(0.1)
pygame.mixer.music.play(-1)

# 获得当前显示器支持的分辨率
size = pygame.display.list_modes()
size_i = 3
width, height = size[size_i]

# speed_x 和 speed_y 用于控制速度
speed_x, speed_y = 1, 1

# 控制小乌龟方向
dir_x, dir_y = -1, 0

# 设置透明度
transparent = 255

clock = pygame.time.Clock()

screen = pygame.display.set_mode(size[size_i])
pygame.display.set_caption("FishC Demo")
fullscreen = False

# 设置放大缩小比率
ratio = 1.0

# 为了避免多次缩放导致图片模糊
# 设置 obg 为原始背景图，background 为实际使用背景图
obg = pygame.image.load("background.jpg")
background = obg
# 同上
oturtle = pygame.image.load("turtle.png")
turtle = oturtle
oturtle_rect = oturtle.get_rect()
position = turtle_rect = oturtle_rect

l_head = turtle
r_head = pygame.transform.flip(turtle, True, False)

def blit_alpha(target, source, location, opacity):
    x = location[0]
    y = location[1]
    temp = pygame.Surface((source.get_width(), source.get_height())).convert()
    temp.blit(target, (-x, -y ))
    temp.blit(source, (0, 0))
    temp.set_alpha(opacity)        
    target.blit(temp, location)

while True:
    for event in pygame.event.get():
        if event.type == QUIT:
            sys.exit()

        if event.type == MOUSEBUTTONDOWN:
            
            # 鼠标按键分别表示增加、恢复、减少游戏速度
            if event.button == 1:
                if speed_x < 30 and speed_y < 30:
                    speed_x += 1
                    speed_y += 1
            if event.button == 2:
                speed_x, speed_y = 1, 1
            if event.button == 3:
                if speed_x > 0 and speed_y > 0:
                    speed_x -= 1
                    speed_y -= 1
                    
            # 鼠标滚轮往上、下滚修改小乌龟透明度
            if event.button == 4:
                if transparent < 255:
                    transparent += 5
            if event.button == 5:
                if transparent > 0:
                    transparent -= 5
                
        if event.type == KEYDOWN:
                
            # 控制小乌龟移动方向（方向键）
            if event.key == K_LEFT:
                dir_x = -1
                dir_y = 0
                turtle = l_head
            if event.key == K_RIGHT:
                dir_x = 1
                dir_y = 0
                turtle = r_head
            if event.key == K_UP:
                dir_y = -1
                dir_x = 0
            if event.key == K_DOWN:
                dir_y = 1
                dir_x = 0
                
            # 放大、缩小屏幕（CTRL + B、S）
            if event.key == K_b and event.mod & KMOD_CTRL:
                if size[size_i] != size[0]:
                    size_i -= 1
                width, height = size[size_i]
                screen = pygame.display.set_mode(size[size_i])
                background = pygame.transform.scale(obg, size[size_i])
            if event.key == K_s and event.mod & KMOD_CTRL:
                if size[size_i] != size[-1]:
                    size_i += 1
                width, height = size[size_i]
                screen = pygame.display.set_mode(size[size_i])
                background = pygame.transform.scale(obg, size[size_i])
                
            # 全屏（F11）
            if event.key == K_F11:
                fullscreen = not fullscreen
                if fullscreen:
                    width, height = size[0]
                    screen = pygame.display.set_mode(size[0], FULLSCREEN | HWSURFACE)
                    background = pygame.transform.scale(obg, size[0])
                else:
                    width, height = size[size_i]
                    screen = pygame.display.set_mode(size[size_i])
                    background = pygame.transform.scale(obg, size[size_i])
                    
            # 放大、缩小小乌龟（=、-），空格恢复原始尺寸
            if event.key == K_EQUALS or event.key == K_MINUS or event.key == K_SPACE:
                # 最大只能放大一倍，缩小50%
                if event.key == K_EQUALS and ratio < 2:
                    ratio += 0.1
                if event.key == K_MINUS and ratio > 0.5:
                    ratio -= 0.1
                if event.key == K_SPACE:
                    ratio = 1
                turtle = l_head = pygame.transform.smoothscale(oturtle, (int(oturtle_rect.width * ratio), int(oturtle_rect.height * ratio)))  
                r_head = pygame.transform.flip(turtle, True, False)
                if dir_x == 1:
                        turtle = r_head     
                
                turtle_rect = turtle.get_rect()
                position.width, position.height = turtle_rect.width, turtle_rect.height

    # 移动小乌龟            
    position = position.move(dir_x * speed_x, dir_y * speed_y)

    if position.left < 0 or position.right > width:
        # 翻转图像
        turtle = pygame.transform.flip(turtle, True, False)
        # 反方向移动
        dir_x = -dir_x
        # 防止小乌龟卡在边缘
        if position.right > width:
            position[0] = width - turtle_rect.width
        else:
            position[0] = 0
                
    if position.top < 0 or position.bottom > height:
        dir_y = -dir_y
        if position.bottom > height:
            position[1] = height - turtle_rect.height
        else:
            position[1] = 0

    screen.blit(background, (0, 0))
    blit_alpha(screen, turtle, position, transparent)
    pygame.display.flip()

    clock.tick(30)
