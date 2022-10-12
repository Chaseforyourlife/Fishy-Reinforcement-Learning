from pickle import FRAME
from game import *
from game_ai import *
from graph import plot,plot_time
from variables import *
from collections import deque
import os
import optuna
from optuna.trial import TrialState

os.environ['KMP_DUPLICATE_LIB_OK']='True'


def main(trial=None,max_game_limit=MAX_GAME_LIMIT):
  fishy_background = pygame.image.load('../static/images/fishy-background.png')
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
    recent_fish_eaten_deque = deque(maxlen=50)
    plot_recent_fish_eaten_means = []
    main_agent = Agent(trial)
    print(main_agent.critic.critic)
    if LOAD_MODEL:
        try:
            main_agent.load_models()
        except:
            print('Model Load Failed')
    #draw background
    screen.blit(fishy_background,(0,0))
    if SHOW_GAME and FPS!=0:
        clock.tick(FPS)
    n_frames = 0
    max_game_count = 1
    while max_game_count != max_game_limit+1 or max_game_limit==0:
        max_game_count+=1
        ##Initialize train start
        main_fishy = Fishy()
        main_school = School()
        time_alive = 0
        frame_number = 0
        win = False
        done = False
        ###MAIN GAME LOOP AFTER START
        while not done:
            n_frames+=1
            time_alive += 1
            if SHOW_GAME:
                #start clock
                clock.tick(FPS)
                #draw background
                screen.blit(fishy_background,(0,0))
            frame_number +=1 
            #update fish_list
            main_school.update()
            #get original_state
            state_old = main_agent.get_state(main_fishy,main_school)
        
            
            #move fish_list
            main_school.move()
            ##Get AI Model Inference
            move,prob,val = main_agent.get_action(state_old)
            #handle move
            main_fishy.handle_move(move)
            #handle main_fishy movement
            flipped,stopped = main_fishy.move()
            #check if fishy collided with any fish in the main_school
            fish_eaten = main_fishy.check_collide(main_school)
            #####TEMPORARY
            if main_fishy.fish_eaten > MAX_FISH_CONSUMED:
                done = True
            if SHOW_GAME:
                #draw every fish in the main_school
                main_school.draw(screen)
                #draw fishy on the screen
                main_fishy.draw(screen)
            ##End game if fishy reaches 150, break and later check if fishy is alive
            if main_fishy.fish_eaten >= MAX_FISH_SIZE+MAX_FISH_CONSUMED:
                win = True
                done = True
            if main_fishy.alive == False:
                done = True
            if time_alive >= FRAME_MAX and FRAME_MAX!=0:
                done = True
            ##Ai Events
            #calculate reward based on if fishy is alive and if he ate anything
            reward = calculate_reward(main_fishy,main_school,fish_eaten,win,flipped,stopped,state_old)
            #get new game state
            state_new = main_agent.get_state(main_fishy,main_school)

            #train short memory
            #main_agent.train_short_memory(state_old,move,reward,state_new,done)
            #remember
            '''
            if frame_number%FRAME_FREQUENCY == 0:
                main_agent.remember(state_old,move,reward,state_new,done)
            '''
            #######main_agent.remember(state_old,move,reward,state_new,done)
            main_agent.remember(state_old,move,prob,val,reward,done)
            #train long memory if done

            if done:




                printt(len(main_agent.memory.states),'MEMORY')
                main_agent.n_games += 1
                if main_fishy.fish_eaten >= record:
                    record = main_fishy.fish_eaten
                    if not TEST:
                        main_agent.save_models()
                        printt('MODEL SAVED')
                    #print(main_agent.model.parameters())
                
                #^replace with 
                main_agent.learn()

                plot_fish_eatens.append(main_fishy.fish_eaten)
                total_fish_eaten += main_fishy.fish_eaten
                recent_fish_eaten_deque.append(main_fishy.fish_eaten)
                plot_mean_fish_eaten = total_fish_eaten/main_agent.n_games
                plot_mean_fish_eatens.append(plot_mean_fish_eaten)
                plot_records.append(record)
                if time_alive > time_record:
                    time_record = time_alive
                plot_time_alives.append(time_alive)
                total_time_alive += time_alive
                recent_fish_eaten_mean = sum(recent_fish_eaten_deque)/len(recent_fish_eaten_deque)
                plot_mean_time_alives.append(total_time_alive/main_agent.n_games)
                plot_time_records.append(time_record)
                plot_recent_fish_eaten_means.append(recent_fish_eaten_mean)
                if main_agent.n_games%10 == 0:
                    plot(plot_fish_eatens,plot_mean_fish_eatens,plot_records,plot_recent_fish_eaten_means)
                    plot_time(plot_time_alives,plot_mean_time_alives,plot_time_records)
                    print('GAMES:',main_agent.n_games)
                 

                
                printt('Game:',main_agent.n_games,'Fish Eaten:',main_fishy.fish_eaten,'Record:',record)
                


            #update screen
            pygame.display.update()
            ##check if window gets closed
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    print(main_fishy.moves_counter)
                    
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
    running=False
    print(f'Average Fish Eaten: {plot_mean_fish_eatens[-1]}')
    print(f'Recent Average Fish Eaten: {plot_recent_fish_eaten_means[-1]}')
    #return plot_recent_fish_eaten_means[-1]
    if OPTUNA_MIN_MAX=='maximum':
        return plot_mean_fish_eatens[-1]
    elif OPTUNA_MIN_MAX=='minimum':
        return plot_mean_time_alives[-1]

if __name__ == '__main__':
    if OPTUNA:
        study = optuna.create_study(direction=OPTUNA_MIN_MAX) # 'maximize' or 'maximize'
        objective = main
        study.optimize(objective,n_trials=OPTUNA_NUM_TRIALS)
        pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
        complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

        print("Study statistics: ")
        print("  Number of finished trials: ", len(study.trials))
        print("  Number of pruned trials: ", len(pruned_trials))
        print("  Number of complete trials: ", len(complete_trials))

        print("Best trial:")
        trial = study.best_trial

        print("  Value: ", trial.value)

        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))
    else:
        main()