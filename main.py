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
    tracks = {
        "default": {
            "img": r"images/track_defualt.png",
            "pos": [850, 820],
            "border": (255, 255, 255, 255),
            "finish": (255, 255, 0, 255)
        },
    }

    WINDOW_SIZE = (1920, 1080)
    WIN = pygame.display.set_mode(WINDOW_SIZE)
    pygame.display.set_caption("AUTOCAR")

    chosen_track = "default"

    TRACK = pygame.image.load(tracks[chosen_track]["img"]).convert()

    FPS = 60

    show_radar = False

    # the colors needed to determine whether the car is on track or not
    BORDER_COLOR = tracks[chosen_track]["border"]
    FINISH_COLOR = tracks[chosen_track]["finish"]

    # font setup
    pygame.font.init()
    font1 = pygame.font.Font('fonts\BebasNeue-Regular.ttf', 70)
    font2 = pygame.font.Font('fonts\BebasNeue-Regular.ttf', 50)

    # car setup
    car_size = (60, 30)

    spawn_point = tracks[chosen_track]["pos"]
    car_num = 5

    gen = 1
    mutation_rate = 1.5
    num_survive = 1

    car_nn_size = (5, 5, 4)

    cars = ()

    for i in range(car_num):
        new_car = Car(spawn_point, car_size, mutation_rate, car_nn_size, 4, 10, 1, 0.04, 0.1, 1, 5, math.pi)
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
            for car in cars:
                car.reset(False)
        clock.tick(fps)
        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    main()
