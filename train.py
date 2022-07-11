from game import *
from game_ai import *
from graph import plot,plot_time

FPS = 500#1600 #30

def main():
  fishy_background = pygame.image.load('static/images/fishy-background.png')
  clock = pygame.time.Clock()
  running = True
  while running:
    ###GRAPH_STATS
    plot_fish_eatens = []
    plot_mean_fish_eatens = []
    plot_records = []
    total_fish_eaten = 0
    record = 0
    plot_time_alives = []
    plot_mean_time_alives = []
    plot_time_records = []
    total_time_alive = 0 
    time_record = 0
    
    main_agent = Agent()
    #draw background
    screen.blit(fishy_background,(0,0))
    clock.tick(FPS)
    while True:
        ##Initialize train start
        main_fishy = Fishy()
        main_school = School()
        time_alive = 0
        win = False
        done = False
        ###MAIN GAME LOOP AFTER START
        while done == False:
            time_alive += 1/FPS
            #start clock
            #####clock.tick(FPS)
            #update fish_list
            main_school.update()
            #get original_state
            state_old = main_agent.get_state(main_fishy,main_school)
            #draw background
            screen.blit(fishy_background,(0,0))
            
            #move fish_list
            main_school.move()
            ##Get AI Model Inference
            move = main_agent.get_action(state_old)
            #handle move
            main_fishy.handle_move(move)
            #handle main_fishy movement
            flipped,stopped = main_fishy.move()
            #check if fishy collided with any fish in the main_school
            fish_eaten = main_fishy.check_collide(main_school)
            #####TEMPORARY
            if fish_eaten > 0:
                done = True
            #draw every fish in the main_school
            main_school.draw(screen)
            #draw fishy on the screen
            main_fishy.draw(screen)
            ##End game if fishy reaches 150, break and later check if fishy is alive
            if main_fishy.fish_eaten >= MAX_FISH_SIZE+4:
                win = True
                done = True
            if main_fishy.alive == False:
                done = True
            ##Ai Events
            #calculate reward based on if fishy is alive and if he ate anything
            reward = calculate_reward(main_fishy,fish_eaten,win,flipped,stopped)
            #get new game state
            state_new = main_agent.get_state(main_fishy,main_school)
            #train short memory
            ###main_agent.train_short_memory(state_old,move,reward,state_new,done)
            #remember
            main_agent.remember(state_old,move,reward,state_new,done)
            #train long memory if done
            if done:
                main_agent.n_games += 1
                if main_fishy.fish_eaten >= record:
                    record = main_fishy.fish_eaten
                    main_agent.model.save()
                    print('MODEL SAVED')
                    #print(main_agent.model.parameters())
                main_agent.train_long_memory()
                plot_fish_eatens.append(main_fishy.fish_eaten)
                total_fish_eaten += main_fishy.fish_eaten
                plot_mean_fish_eaten = total_fish_eaten/main_agent.n_games
                plot_mean_fish_eatens.append(plot_mean_fish_eaten)
                plot_records.append(record)
                if time_alive > time_record:
                    time_record = time_alive
                plot_time_alives.append(time_alive)
                total_time_alive += time_alive
                plot_mean_time_alives.append(total_time_alive/main_agent.n_games)
                plot_time_records.append(time_record)
                if main_agent.n_games%1+1 == 1:
                    plot(plot_fish_eatens,plot_mean_fish_eatens,plot_records)
                    plot_time(plot_time_alives,plot_mean_time_alives,plot_time_records)

                
                print('Game:',main_agent.n_games,'Fish Eaten:',main_fishy.fish_eaten,'Record:',record)
                


            #update screen
            pygame.display.update()
            ##check if window gets closed
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    pygame.quit()
        if main_fishy.alive:
            #print('Game Won')
            pass
        elif not main_fishy.alive:
            #print('Game Lost')
            pass
        #TODO
        ###MAKE PLAY AGAIN METHOD
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                pygame.quit()




main()