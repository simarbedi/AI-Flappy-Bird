import pygame
import numpy 
import pandas
import os
import random
import neat
import pickle
from pygame import mixer
pygame.font.init()
pygame.mixer.init()

WIN_WIDTH=500
WIN_HEIGHT=700
STAT_FONT=pygame.font.SysFont("comicstans",50)
GEN=0


BIRD_IMGS=[pygame.transform.scale2x(pygame.image.load(os.path.join("imgs","bird1.png"))),pygame.transform.scale2x(pygame.image.load(os.path.join("imgs","bird2.png"))),pygame.transform.scale2x(pygame.image.load(os.path.join("imgs","bird3.png")))]
PIPE_IMG=pygame.transform.scale2x(pygame.image.load(os.path.join("imgs","pipe.png")))
BASE_IMG=pygame.transform.scale2x(pygame.image.load(os.path.join("imgs","base.png")))
BG_IMG=pygame.transform.scale(pygame.image.load(os.path.join("imgs","bg.png")),(600,800))
Flap_sound=os.path.join("music","flap.wav")
Background_sound=os.path.join("music","background.wav")
	



class Bird:
	IMGS=BIRD_IMGS
	MAX_ROTATION=25
	ROT_VEL=20
	ANIMATION_TIME=5

	def __init__(self,x,y):
		self.x = x
		self.y=y
		self.tilt=0
		self.tick_count=0
		self.vel=0
		self.height=self.y
		self.img_count=0
		self.img=self.IMGS[0]

	def jump(self):
		self.vel= -10.5
		self.tick_count= 0
		self.height= self.y
		jump_music=mixer.Sound(Flap_sound)
		jump_music.set_volume(0.3)
		jump_music.play()

	def move(self):
		
		self.tick_count+=1

		d=self.vel*self.tick_count + 1.5*(self.tick_count)**2

		#failsafe for max for going down 
		if d >= 16:
			d = 16

		#to move a little up when moving up	
		if d < 0:
			d -= 2

		#update position
		self.y= self.y+d

		#when moving upwards or are above the initial jump point at that time 
		if d < 0 or self.y < self.height +50:
			if self.tilt < self.MAX_ROTATION:
				self.tilt = self.MAX_ROTATION

		else:
			if self.tilt > -90:
				self.tilt -= self.ROT_VEL

	def draw(self,win):
		self.img_count+=1

		# for change in the animation picture of bird 
		if self.img_count < self.ANIMATION_TIME:
			self.img=self.IMGS[0]
		elif self.img_count < self.ANIMATION_TIME*2:
			self.img=self.IMGS[1]
		elif self.img_count < self.ANIMATION_TIME*3:
			self.img=self.IMGS[2]
		elif self.img_count < self.ANIMATION_TIME*4:
			self.img=self.IMGS[1]
		elif self.img_count == self.ANIMATION_TIME*4 +1:
			self.img=self.IMGS[0]
			self.img_count=0

		if self.tilt <= -80:
			self.img=self.IMGS[1]
			self.img_count=self.ANIMATION_TIME*2

		rotated_image = pygame.transform.rotate(self.img,self.tilt)
		#draw on the rectangle
		new_rect=rotated_image.get_rect(center=self.img.get_rect(topleft=(self.x,self.y)).center)
		win.blit(rotated_image, new_rect.topleft)


	def get_mask(self):
		return pygame.mask.from_surface(self.img)

class Pipe:
	VEL=5
	GAP=200

	def __init__(self,x):
		self.x=x
		self.height=0
		self.gap=100
		self.top=0
		self.bottom=0

		self.PIPE_TOP=pygame.transform.flip(PIPE_IMG,False,True)
		self.PIPE_BOTTOM=PIPE_IMG
		# check if bird passed the pipe
		self.passed=False
		self.set_height()


	def set_height(self):
		self.height=random.randrange(50,450)
		self.top=self.height - self.PIPE_TOP.get_height()
		self.bottom=self.height+self.GAP

	def move(self):
		self.x -= self.VEL

	def draw(self,win):
		win.blit(self.PIPE_TOP,(self.x,self.top))
		win.blit(self.PIPE_BOTTOM,(self.x,self.bottom))

	def collide(self,bird):
		bird_mask=bird.get_mask()
		top_mask=pygame.mask.from_surface(self.PIPE_TOP)
		bottom_mask=pygame.mask.from_surface(self.PIPE_BOTTOM)

		#it tells difference between the image masks
		top_offset=(self.x-bird.x,self.top-round(bird.y))
		bottom_offset=(self.x-bird.x,self.bottom-round(bird.y))
		
		#it tells the first overlapping point
		b_point=bird_mask.overlap(bottom_mask,bottom_offset)
		t_point=bird_mask.overlap(top_mask,top_offset)

		if b_point or t_point:
			return True

		return False

class Base:
	VEL=5
	IMG=BASE_IMG
	WIDTH=BASE_IMG.get_width()

	def __init__(self,y):
		self.y=y
		self.x1=0
		self.x2=self.WIDTH

	def move(self):
		self.x1 -=self.VEL
		self.x2 -=self.VEL

		if self.x1 + self.WIDTH <0:
			self.x1=self.x2 + self.WIDTH

		if self.x2 + self.WIDTH <0:
			self.x2=self.x1 + self.WIDTH


	def draw(self,win):
		win.blit(self.IMG,(self.x1,self.y))
		win.blit(self.IMG,(self.x2,self.y))

def draw_window(win,birds,pipes,base,score,gen):
	win.blit(BG_IMG,(0,0))

	text=STAT_FONT.render("Score:"+str(score),1,(255,255,255))
	win.blit(text,(WIN_WIDTH - 10 - text.get_width(),10))

	text=STAT_FONT.render("GEN:"+str(gen),1,(255,255,255))
	win.blit(text,(10,10))

	for pipe in pipes:
		pipe.draw(win)

	base.draw(win)
	for bird in birds:
		bird.draw(win)
	pygame.display.update()

def main(genome,config):
	global GEN
	GEN+=1
	nets=[]
	ge=[]
	birds=[]

	mixer.music.load(Background_sound)
	mixer.music.play(-1)

	for _,g in genome:
		net=neat.nn.FeedForwardNetwork.create(g,config)
		nets.append(net)
		birds.append(Bird(230,350))
		g.fitness=0
		ge.append(g)

	base=Base(630)
	pipes = [Pipe(500)]
	win=pygame.display.set_mode((WIN_WIDTH,WIN_HEIGHT))
	clock=pygame.time.Clock()
	run=True

	score=0

	while run:
		clock.tick(30)
		for event in pygame.event.get():
			if event.type==pygame.QUIT:
				run=False
				pygame.quit()
				quit() 
		
		pipe_ind=0
		if len(birds)>0:
			if len(pipes)>1 and birds[0].x > pipes[0].x + pipes[0].PIPE_TOP.get_width():
				pipe_ind=1
		else:
			run=False
			break

		for x,bird in enumerate(birds):
			bird.move()
			ge[x].fitness+=0.1


			output=nets[x].activate((bird.y,abs(bird.y-pipes[pipe_ind].height),abs(bird.y-pipes[pipe_ind].bottom)))

			if output[0]>0.5:
				bird.jump()
		

		add_pipe=False
		rem=[]

		for pipe in pipes:
			for x,bird in enumerate(birds):	
				if pipe.collide(bird):
					ge[x].fitness-=1
					birds.pop(x)
					nets.pop(x)
					ge.pop(x)

				if not pipe.passed and pipe.x<bird.x:
					pipe.passed=True
					add_pipe=True

			if pipe.x + pipe.PIPE_TOP.get_width()<0:
				rem.append(pipe)

			pipe.move()

		if add_pipe:
			score+=1
			for g in ge:
				g.fitness+=5
			pipes.append(Pipe(500))

		for r in rem:
			pipes.remove(r)

		for x,bird in enumerate(birds):
			if bird.y+ bird.img.get_height() >= 630 or bird.y<0:
				birds.pop(x)
				ge.pop(x)
				nets.pop(x)
		if score>50:
			break

		base.move()
		draw_window(win,birds,pipes,base,score,GEN)
	
def run(config_file):
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    p = neat.Population(config)

    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    winner = p.run(main, 50)

    #out=open("best_bird.pickle","wb")
    #pickle.dump(winner,out)
    #out.close()

    print('\nBest genome:\n{!s}'.format(winner))


if __name__ == '__main__':
    

    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config.txt')
    run(config_path)












