import pygame
import neat
import pickle
# Code is written for training an AI using NEAT fo the Trex Runner game
# Using the code we have save the model in std1.pkl file

# Use the model to play the game and remove the training code

# Load configuration file
config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,neat.DefaultSpeciesSet, neat.DefaultStagnation,'config-feedforward.txt')  

# Open the model and load the winner genome
with open('std1.pkl', 'rb') as f:
    genome = pickle.load(f)

# Create a neural network named net
net = neat.nn.FeedForwardNetwork.create(genome, config)

pygame.init()

screen = pygame.display.set_mode((1200, 400))

gameState = "play"
dinoState = "run"

score = 0
scoreFont = pygame.font.Font("freesansbold.ttf", 16)

dino = pygame.image.load("sprites/trex1.png")
cacti = pygame.image.load("sprites/obstacle1.png")
ground = pygame.image.load("sprites/ground.png")

dinoRect = pygame.Rect(100, 250, 64, 64)
cactusRect = pygame.Rect(1100, 275, 32, 32)
groundRect = pygame.Rect(0, 330, 1200, 2)

dinoYChange = 0

def save(winner):
    file_name = 'std1.pkl'
    with open(file_name, 'wb') as file:
        pickle.dump(winner, file)
        print(f'Object successfully saved to "{file_name}"')

def jump():
    global dinoYChange , dinoState
    if dinoState =="run":
       dinoYChange = -2

# Remove these lines and only keep while loop
# def eval_fitness(generation, config):
#     global gameState, dinoState, dinoYChange, score, gc

#     for gid, genome in generation:
#         net = neat.nn.FeedForwardNetwork.create(genome, config)

while True:
            screen.fill((255, 255, 255))
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                
                if gameState == "play":
                    if dinoState == "run":
                        if event.type == pygame.KEYDOWN:
                            if event.key == pygame.K_SPACE:
                                jump()
                                
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_UP:
                            gameState = "over"
                                
            if gameState == "over":
                    gameState = "play"
                    cactusRect.x = 1200
                    score = 0
                    dinoYChange = 0
                    break
                
            if gameState == "play":
                dinoRect.y += dinoYChange
                
                if dinoRect.y > 250:
                    dinoState = "run"
                    dinoRect.y = 250
                else:
                    dinoState = "jump"
                    dinoYChange = dinoYChange + 0.01
                    if dinoYChange == 0:
                        dinoYChange = 0.1
                
                cactusRect.x = cactusRect.x - 1
                if cactusRect.x <= -30:
                    cactusRect.x = 1200
                
                score += 1
                showScore = round(score/100)
                scoreShow = scoreFont.render("Score: " + str(showScore), True, (0, 0, 0))
                screen.blit(scoreShow, (10, 10))   
                
                screen.blit(dino, dinoRect)
                screen.blit(cacti, cactusRect)
                
                imageWidth = ground.get_width()
                screen.blit(ground, groundRect)
                screen.blit(ground, (imageWidth + groundRect.x, groundRect.y))
                if groundRect.x <= -imageWidth:
                    screen.blit(ground, (imageWidth + groundRect.x, groundRect.y))
                    groundRect.x = 0
                groundRect.x -= 1
                
                if dinoRect.colliderect(cactusRect):
                    pygame.time.delay(500)
                    gameState = "over"
                    
            if gameState == "over":
                scoreShow = scoreFont.render("Score: " + str(showScore), True, (0, 0, 0))
                screen.blit(scoreShow, (550, 190))
            
            output = net.activate((dinoRect.y, cactusRect.x))
            
            if output[0]>0.5:
                  jump()

            # Remove fitness calculation 
            # genome.fitness = score 
            # if(genome.fitness>50000):
            #   save(genome)
            #   gameState = "over"

            pygame.display.update()

# Remove these lines
# config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,neat.DefaultSpeciesSet, neat.DefaultStagnation,'config-feedforward.txt')  
# p = neat.Population(config)
#winner = p.run(eval_fitness,10) 
