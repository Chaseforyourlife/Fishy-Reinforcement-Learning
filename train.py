from game import *
from game_ai import *

FPS = 10000

def main(MODE = 'play'):
  fishy_background = pygame.image.load('static/images/fishy-background.png')
  clock = pygame.time.Clock()
  running = True
  while running:
    #initialize stats
    record = 0
    main_agent = Agent()
    #draw background
    screen.blit(fishy_background,(0,0))
    clock.tick(FPS)
    while True:
        ##Initialize train start
        main_fishy = Fishy()
        main_school = School()
        
        win = False
        done = False
        ###MAIN GAME LOOP AFTER START
        while done == False:
            #get original_stae
            state_old = main_agent.get_state(main_fishy,main_school)
            #start clock
            clock.tick(FPS)
            #draw background
            screen.blit(fishy_background,(0,0))
            #update fish_list
            main_school.update()
            #get ai model inference
            move = main_agent.get_action(state_old)
            #handle move
            main_fishy.handle_move(move)
            #handle main_fishy movement
            main_fishy.move()
            #check if fishy collided with any fish in the main_school
            fish_eaten = main_fishy.check_collide(main_school)
            #draw every fish in the main_school
            main_school.draw(screen)
            #draw fishy on the screen
            main_fishy.draw(screen)
            ##End game if fishy reaches 150, break and later check if fishy is alive
            if main_fishy.fish_eaten >= 150:
                win = True
                done = False
            if main_fishy.alive == False:
                done = True
            ##Ai Events
            #calculate reward based on if fishy is alive and if he ate anything
            reward = calculate_reward(main_fishy,fish_eaten,win)
            #get new game state
            state_new = main_agent.get_state(main_fishy,main_school)
            #train short memory
            main_agent.train_short_memory(state_old,move,reward,state_new,done)
            #remember
            main_agent.remember(state_old,move,reward,state_new,done)
            #train long memory if done
            if done:
                main_agent.n_games += 1
                record = max(record,main_fishy.fish_eaten)
                main_agent.train_long_memory()

                print('Game:',main_agent.n_games,'Fish Eaten:',main_fishy.fish_eaten,'Record:',record)
                #TODO plot


            #update screen
            pygame.display.update()
            ##check if window gets closed
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    pygame.quit()
        if main_fishy.alive:
            print('Game Won')
        elif not main_fishy.alive:
            print('Game Lost')
        #TODO
        ###MAKE PLAY AGAIN METHOD
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                pygame.quit()




main()