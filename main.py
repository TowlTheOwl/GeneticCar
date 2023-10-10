import pygame
import math
from tools import *
from car import Car

"""
TODO:
1. Choose Car Colors
2. Make the screen size scalable
"""
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
            "pos": (850, 820),
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
    car_num = 5

    gen = 1
    mutation_rate = 0.2
    num_survive = 1

    car_nn_size = (5, 5, 4)

    cars = ()

    start_pos = tuple([i * windowScale for i in spawn_point])

    size = car_size
    start_vel = 4*windowScale
    max_vel = 10 * windowScale
    min_vel = 2 * windowScale
    rot_vel = 0.04 * windowScale
    acc = 0.1 * windowScale
    scan_length = int(300*windowScale)

    # start_pos:tuple, size:tuple, mr:float, nn_size:tuple, start_vel:float, max_vel:float, min_vel:float, rot_vel:float, acc:float, bias_term:int, num_sensors:int=5, sensor_angle=math.pi

    for i in range(car_num):
        new_car = Car(spawn_point, car_size, mutation_rate, car_nn_size, start_vel, max_vel, min_vel, rot_vel, acc, 1, scan_length, 5, math.pi)
        cars += (new_car, )

    clock = pygame.time.Clock()

    run = True

    def draw_screen(win):
        win.blit(TRACK, (0, 0))

    while run:
        fps = FPS
        show_radar = False
        zoom = False
        cars_status = ()
        cars_dist = ()
        cars_time = ()
        car_data = ()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                break
        
        keys = pygame.key.get_pressed()

        if keys[pygame.K_BACKSPACE]:
            run = False
        if keys[pygame.K_1]:
            show_radar = True
        if keys[pygame.K_2]:
            zoom = True

        if zoom: 
            fps = 0
        for car in cars:
            cars_status += (car.life_status(),)
            cars_dist += (car.return_distance(),)
            cars_time += (car.return_time(), )
        
        draw_screen(WIN)
        
        for car in cars:
            car.update(WIN, TRACK, BORDER_COLOR, FINISH_COLOR, show_radar)

        if not any(cars_status):
            gen += 1
            bestCar = cars_dist.index(max(cars_dist))
            bestCarWeight = cars[bestCar].return_weights()
            for carIndex in range(len(cars)):
                if carIndex != bestCar:
                    cars[carIndex].reset(True, bestCarWeight)
                else:
                    cars[carIndex].reset(False)
        clock.tick(fps)
        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    screensize = int(input("Enter screen size (-1 for default): "))
    main()
