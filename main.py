import pygame
import math
from tools import *
from car import Car
import os

"""
TODO:
1. Choose Car Colors
2. Make the screen size scalable
"""
def draw_idle(win):
    win.fill((0, 0, 0))
    pygame.display.flip()

def main():
    DEFAULT_WINDOW_SIZE = (1920, 1080)
    if screensize == -1:
        WINDOW_SIZE = DEFAULT_WINDOW_SIZE
        windowScale = 1
    else:
        windowScale = screensize/DEFAULT_WINDOW_SIZE[0]
        WINDOW_SIZE = [i * windowScale for i in DEFAULT_WINDOW_SIZE]
    WIN = pygame.display.set_mode(WINDOW_SIZE)
    pygame.display.set_caption("AUTOCAR")


    tracks = {
        "default": {
            "img": r"images/track_defualt.png",
            "pos": (850, 835),
            "border": (255, 255, 255, 255),
            "finish": (255, 255, 0, 255)
        },
        "hard": {
            "img": r"images/track_hard.png",
            "pos": (850, 835),
            "border": (255, 255, 255, 255),
            "finish": (255, 255, 0, 255)
        },
        "training": {
            "img": r"images/track_training.png",
            "pos": (850, 835),
            "border": (255, 255, 255, 255),
            "finish": (255, 255, 0, 255)
        },
    }

    userTrackChoice = ""
    tracks_keys = tracks.keys()
    while userTrackChoice not in tracks_keys:
        userTrackChoice = input(f"Choose a track: \n{tracks.keys()}\n>")
    chosen_track = userTrackChoice

    TRACK = pygame.image.load(tracks[chosen_track]["img"]).convert()
    TRACK = pygame.transform.scale(TRACK, WINDOW_SIZE)
    WIN.blit(TRACK, (0, 0))

    FPS = 60

    show_radar = False

    # the colors needed to determine whether the car is on track or not
    BORDER_COLOR = tracks[chosen_track]["border"]
    FINISH_COLOR = tracks[chosen_track]["finish"]

    # font setup
    pygame.font.init()
    font1 = pygame.font.Font('fonts\BebasNeue-Regular.ttf', int(70*windowScale))
    font2 = pygame.font.Font('fonts\BebasNeue-Regular.ttf', int(50*windowScale))

    # car setup
    car_size = (int(60*windowScale), int(30*windowScale))

    spawn_point = tracks[chosen_track]["pos"]
    
    car_num = 100

    gen = 1
    init_mr = 5
    post_qual_mr = 0.5
    num_survive = 5

    car_nn_size = (5, 5, 3)
    nn_len = range(len(car_nn_size))

    bias = 1
    nn_layer_output_size = tuple([bias + i for i in car_nn_size[:-1]])
    nn_layer_output_size += (car_nn_size[-1],)
    nn_layer_iter = tuple([range(i) for i in nn_layer_output_size])

    cars = ()

    start_pos = tuple([int(i * windowScale) for i in spawn_point])

    size = car_size
    start_vel = 4*windowScale
    max_vel = 10 * windowScale
    min_vel = 2 * windowScale
    rot_vel = 0.04 * windowScale
    acc = 0.1 * windowScale
    scan_length = int(300*windowScale)

    init_dist = np.random.uniform
    init_params = (-1, 1)
    pre_qual_dist = np.random.uniform
    pre_qual_params = (-1, 1)
    post_qual_dist = np.random.normal
    post_qual_params = (0, 1)
    
    pygame.display.flip()

    # start_pos:tuple, size:tuple, mr:float, nn_size:tuple, start_vel:float, max_vel:float, min_vel:float, rot_vel:float, acc:float, bias_term:int, num_sensors:int=5, sensor_angle=math.pi

    for i in range(car_num):
        new_car = Car(start_pos, car_size, init_mr, car_nn_size, start_vel, max_vel, min_vel, rot_vel, acc, bias, scan_length, 5, math.pi, dist_type=init_dist, dist_params=init_params)
        cars += (new_car, )

    clock = pygame.time.Clock()

    run = True

    def draw_screen(win):
        win.blit(TRACK, (0, 0))

    # NN DISPLAY
    nn_disp_rect_pos = tuple([int(i/4) for i in WINDOW_SIZE])
    nn_disp_rect_size = tuple([int(i/2) for i in WINDOW_SIZE])
    nn_display_size = tuple([int(i*5/12) for i in WINDOW_SIZE])
    nn_display_pos = tuple([int(nn_disp_rect_pos[i]+nn_display_size[i]/12) for i in range(len(WINDOW_SIZE))])
    nn_display_spacing_y = ()
    nn_display_spacing_x = int(nn_display_size[0]/(len(car_nn_size)-1))
    for layer in nn_layer_output_size:
        nn_display_spacing_y += (int(nn_display_size[1]/(layer-1)),)

    frame_num = 0
    survivors = []
    show_radar = False
    zoom = False
    display_nn = False
    cars_dist = ()
    cars_time = ()
    track_finish = False
    first_qual = False

    # data collection
    gen_dist = []
    best_distances = []
    gen_time = []
    best_times = []

    draw = True

    save = input("Save (type y)? ") == 'y'
    if save:
        gen_name = input("Please type in the name: ")
        dir_name = "saves/"+gen_name
        if not os.path.isdir(dir_name):
            os.mkdir(dir_name)
            print("New save created.")
        else:
            print("Loading " + gen_name + "...")
            cars[0].init_weight(np.load(f"{dir_name}/nn.npy"), distribution_type=init_dist, params=init_params, equal=True)
            gen_dist = list(np.load(f"{dir_name}/gen_dist.npy"))
            best_distances = list(np.load(f"{dir_name}/dist.npy"))
            gen_time = list(np.load(f"{dir_name}/gen_time.npy"))
            best_times = list(np.load(f"{dir_name}/time.npy"))
            if len(gen_time) == 0:
                gen=max(gen_dist) + 1
            else:
                gen=max(gen_time) + 1
            print("Loading complete!")
            

    while run:
        fps = FPS
        cars_status = ()
        car_data = ()
        frame_num += 1
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                break
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_BACKSPACE:
                    run = False
                if event.key == pygame.K_1:
                    show_radar = not show_radar
                if event.key == pygame.K_2:
                    zoom = not zoom
                if event.key == pygame.K_3 and gen>1:
                    display_nn = not display_nn
                    if display_nn:
                        nn_disp_start_frame = frame_num
                if event.key == pygame.K_g:
                    graph_data(gen_dist, best_distances, gen_time, best_times)
                if event.key == pygame.K_d:
                    draw = not draw
                    if not draw:
                        draw_idle(WIN)
        
        if zoom: 
            fps = 0
        for car in cars:
            cars_status += (car.life_status(),)
        
        for car in cars:
            car.update(WIN, TRACK, BORDER_COLOR, FINISH_COLOR)

        if not any(cars_status):
            for car in cars:
                cars_dist += (car.return_distance(),)
                cars_time += (car.return_time(), )
            cars_dist = np.array(cars_dist)
            cars_time = np.array(cars_time)
            
            if not np.count_nonzero(cars_time != float('inf')):
                sorted_dist = np.argsort(cars_dist)
                survivors = sorted_dist[-num_survive:]
                gen_dist.append(gen)
                best_distances.append(max(cars_dist))
            else:
                if not track_finish:
                    first_qual = True
                    track_finish = True
                else:
                    first_qual = False
                
                num_qualified = np.count_nonzero(cars_time != float('inf'))
                sorted_time = np.argsort(cars_time)
                survivors = sorted_time[:min(num_qualified, num_survive)]
                gen_time.append(gen)
                best_times.append(min(cars_time))
            
            car_weights = np.array([cars[car].return_weights() for car in survivors])

            if track_finish:
                print("Finish!")
                if first_qual:
                    for car in cars:
                        car.set_mutation_rate(post_qual_mr)
                
                for carIndex in range(len(cars)):
                    if carIndex not in survivors:
                        cars[carIndex].reset(True, car_weights[np.random.randint(len(survivors))], distribution_type=post_qual_dist, dist_params=post_qual_params)
                    else:
                        cars[carIndex].reset(False)
                
            else:
                for carIndex in range(len(cars)):
                    if carIndex not in survivors:
                        cars[carIndex].reset(True, car_weights[np.random.randint(len(survivors))], distribution_type=pre_qual_dist, dist_params=pre_qual_params)
                    else:
                        cars[carIndex].reset(False)
            
            
            cars_dist = ()
            cars_time = ()

            gen += 1
        
        # DRAW SCREEN
        if draw:
            if not zoom or frame_num > 10:
                draw_screen(WIN)
                for car in cars:
                    car.draw(WIN)
                    if show_radar:
                        car.draw_radar(WIN)
            
            if display_nn and (len(survivors) != 0):
                if frame_num%nn_disp_start_frame == 0:
                    toDisplay = cars[survivors[0]].get_nn()
                    out = ()
                    for layer in toDisplay[:]:
                        out+=(layer>0.5,)
                pygame.draw.rect(WIN, (128, 128, 128), (*nn_disp_rect_pos, *nn_disp_rect_size))
                for layer in nn_len:
                    for neuron in nn_layer_iter[layer]:
                        if out[layer][0][neuron]:
                            neuron_color = (255, 255, 255)
                        else:
                            neuron_color = (0, 0, 0)
                        pygame.draw.circle(WIN, neuron_color, (nn_display_pos[0]+nn_display_spacing_x*layer, nn_display_pos[1]+nn_display_spacing_y[layer]*neuron), 10)
            
            clock.tick(fps)
            pygame.display.flip()
                        
            if frame_num>10:
                frame_num = 0

    pygame.quit()
    if save:
        # save best neural network
        np.save(f"saves/{gen_name}/nn.npy", car_weights[0])
        # save graph data
        np.save(f"saves/{gen_name}/gen_dist.npy", gen_dist)
        np.save(f"saves/{gen_name}/dist.npy", best_distances)
        np.save(f"saves/{gen_name}/gen_time.npy", gen_time)
        np.save(f"saves/{gen_name}/time.npy", best_times)
        print("Save complete!")

if __name__ == "__main__":
    screensize = int(input("Enter screen size (-1 for default): "))
    main()
