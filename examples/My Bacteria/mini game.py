import neat
import pygame
import numpy as np


#load images
IMG_BACTERIA = pygame.image.load('bacteria.png')
IMG_TARGET = pygame.image.load('target.png')
IMG_BG = pygame.image.load('bg.jpg')

#scale images
bt_factor = 0.02
bg_factor = 2
IMG_BACTERIA = pygame.transform.scale(IMG_BACTERIA, (IMG_BACTERIA.get_width()*bt_factor, IMG_BACTERIA.get_height()*bt_factor))
IMG_TARGET = pygame.transform.scale(IMG_TARGET, (IMG_TARGET.get_width()*bt_factor, IMG_TARGET.get_height()*bt_factor))
IMG_BG = pygame.transform.scale(IMG_BG, (IMG_BG.get_width()*bg_factor, IMG_BG.get_height()*bg_factor))

#initialize window's width and height
WIN_WIDTH, WIN_HEIGHT = IMG_BG.get_width(), IMG_BG.get_height()

class Bacteria:
    IMG = IMG_BACTERIA
    score = 0
    MAX_VEL = 5
    STOP_FACTOR = 0.9
    
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.velx = 0
        self.vely = 0
    
    def move(self):
        keys = pygame.key.get_pressed()
        x, y = 1,1
        if(keys[pygame.K_RIGHT]):
            self.velx += x
        if(keys[pygame.K_LEFT]):
            self.velx -= x
        if(keys[pygame.K_UP]):
            self.vely -= y
        if(keys[pygame.K_DOWN]):
            self.vely += y
        
        if abs(self.velx) >= self.MAX_VEL:
            self.velx = self.MAX_VEL*abs(self.velx)/self.velx
        if abs(self.vely) >= self.MAX_VEL:
            self.vely = self.MAX_VEL*abs(self.vely)/self.vely
        self.velx*= self.STOP_FACTOR
        self.vely*= self.STOP_FACTOR
        if(self.x>=WIN_WIDTH-self.IMG.get_width() and self.velx>0) or (self.x<=0 and self.velx<0):
            self.velx = 0
        if(self.y>=WIN_HEIGHT-self.IMG.get_height() and self.vely>0) or (self.y<=0 and self.vely<0):
            self.vely = 0
        self.x += self.velx
        self.y += self.vely

    def check_collision(self, target):
        if(self.IMG.get_rect(topleft=(self.x, self.y)).colliderect(target.IMG.get_rect(topleft=(target.x, target.y)))):
            self.score+=1
            target.gen()
            print(self.score)
            return True

    def draw(self, win):
        win.blit(self.IMG, (self.x, self.y))
    
    def get_mask(self):
        return pygame.mask.from_surface(self.IMG)

class Target:
    IMG = IMG_TARGET
    
    def __init__(self):
        self.gen()

    def gen(self):
        self.x = np.random.randint(0, WIN_WIDTH - IMG_TARGET.get_width()/2)
        self.y = np.random.randint(0, WIN_HEIGHT - IMG_TARGET.get_height()/2)

    def draw(self, win):
        win.blit(self.IMG, (self.x, self.y))
    
    def get_mask(self):
        return pygame.mask.from_surface(self.IMG)

def draw(win, bacterias, target):
    win.blit(IMG_BG, (0,0))
    target.draw(win)
    for bacteria in bacterias:
        bacteria.draw(win)
    
    pygame.display.update()

def main():
    bacterias = [Bacteria(np.random.randint(0, WIN_WIDTH - IMG_BACTERIA.get_width()/2),np.random.randint(0, WIN_HEIGHT - IMG_BACTERIA.get_height()/2))]
    target = Target()
    win = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
    clock = pygame.time.Clock()
    run = True
    while run:
        clock.tick(30)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
        for bacteria in bacterias:
            bacteria.move()
            bacteria.check_collision(target)
        draw(win, bacterias, target)

print('Starting Game')
main()