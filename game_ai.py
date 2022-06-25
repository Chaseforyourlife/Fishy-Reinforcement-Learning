def get_game_state(fishy,school):
    game_state = []
    #Input Layer Data
    game_state.append(fishy.x)
    game_state.append(fishy.y)
    game_state.append(fishy.width)
    game_state.append(fishy.height)
    game_state.append(fishy.x_speed)
    game_state.append(fishy.y_speed)
    #Add data for 8 fish
    for fish in school.fish_list:
        game_state.append(fish.x)
        game_state.append(fish.y)
        game_state.append(fish.width)
        game_state.append(fish.height)
        game_state.append(fish.x_speed)

    return game_state