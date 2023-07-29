import pygame
import numpy as np
import neatiheb


#load images
IMG_BACTERIA = pygame.image.load('bacteria.png')
IMG_TARGET = pygame.image.load('target.png')
IMG_BG = pygame.image.load('bg.jpg')

#scale images
bt_factor = 0.02
bg_factor = 3
IMG_BACTERIA = pygame.transform.scale(IMG_BACTERIA, (IMG_BACTERIA.get_width()*bt_factor, IMG_BACTERIA.get_height()*bt_factor))
IMG_TARGET = pygame.transform.scale(IMG_TARGET, (IMG_TARGET.get_width()*bt_factor, IMG_TARGET.get_height()*bt_factor))
IMG_BG = pygame.transform.scale(IMG_BG, (IMG_BG.get_width()*bg_factor, IMG_BG.get_height()*bg_factor))

#initialize window's width and height
WIN_WIDTH, WIN_HEIGHT = IMG_BG.get_width(), IMG_BG.get_height()

win = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
gen = 0
MAX_FITNESS = 50

class Bacteria:
    IMG = IMG_BACTERIA
    score = 0
    MAX_VEL = 5
    STOP_FACTOR = 0.9
    
    def __init__(self, x, y, genome:neatiheb.genome.Genome):
        self.x = x
        self.y = y
        self.velx = 0
        self.vely = 0
        self.genome = genome
    
    def move(self, target):
        input = (target.x-self.x, target.y-self.y)
        x, y = self.genome.forward(input)
        self.velx+=x
        self.vely+=y
        
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
            self.genome.fitness+=5
            target.gen()
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

def main(genomes):
    global win, gen

    gen+= 1

    bacterias = []
    for genome in genomes:
        bacterias.append(Bacteria(np.random.randint(0, WIN_WIDTH - IMG_BACTERIA.get_width()/2),np.random.randint(0, WIN_HEIGHT - IMG_BACTERIA.get_height()/2), genome))
    
    target = Target()
    clock = pygame.time.Clock()
    running = True
    #print(f'running gen{running}')
    while running and max([b.genome.fitness for b in bacterias])<MAX_FITNESS:
        #print(f'running gen{gen}')
        clock.tick(60)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        for bacteria in bacterias:
            bacteria.move(target)
            bacteria.check_collision(target)
        draw(win, bacterias, target)

def run():
    p = neatiheb.population.Population(2, 2, 50)
    winner = p.evaluate(main)
    print("we've got a winner!!")


run()