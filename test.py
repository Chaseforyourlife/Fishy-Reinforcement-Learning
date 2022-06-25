from game import *
import game_ai as GAI

def main(MODE = 'play'):
  fishy_background = pygame.image.load('static/images/fishy-background.png')
  clock = pygame.time.Clock()
  running = True
  while running:
    #draw background
    screen.blit(fishy_background,(0,0))
    clock.tick(FPS)
    ##Initialize game start
    main_fishy = Fishy()
    main_school = School()
    print('Game Start')
    ###MAIN GAME LOOP AFTER START
    while main_fishy.alive == True:
        clock.tick(FPS)
        #draw background
        screen.blit(fishy_background,(0,0))
        #update fish_list
        main_school.update()
        #check if fishy collided with any fish in the main_school
        main_fishy.check_collide(main_school)
        #get ai model inference and handle move
        main_fishy.handle_move(GAI.make_inference())
        #handle main_fishy movement
        main_fishy.move()
        #draw every fish in the main_school
        main_school.draw(screen)
        #draw fishy on the screen
        main_fishy.draw(screen)
        ##End game if fishy reaches 150, break and later check if fishy is alive
        if main_fishy.fish_eaten >= 150:
            break
        ##Ai Events
        game_state = GAI.get_game_state(main_fishy,main_school)
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