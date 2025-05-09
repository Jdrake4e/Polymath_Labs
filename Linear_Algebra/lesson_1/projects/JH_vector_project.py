# Note: I used the following enviroment command "set PYGAME_DETECT_AVX2=1"
# Author: John Hohman

import pygame

pygame.init()

resolution = [500,500]
circle_color = (0,0,255)
circle_position = [100,370]
vector_modifier = [2,5]
circle_radius = 15

screen = pygame.display.set_mode(resolution)

x_dir = True
y_dir = True

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen_color = (0,0,0)
    screen.fill(screen_color)

    # Draws new circle
    pygame.draw.circle(screen, 
                       circle_color, 
                       (circle_position[0], circle_position[1]), 
                       circle_radius)

    # Check for out of bounds condition and reflect the vector, if that is the case
    if circle_position[0] >= resolution[0]-circle_radius:
        x_dir = False
        circle_color = (0,0,255)
    elif circle_position[0] <= 0+circle_radius:
        x_dir = True
        circle_color = (0,255,0)

    if circle_position[1] >= resolution[1]-circle_radius:
        y_dir = False
        circle_color = (255,0,0)
    elif circle_position[1] <= 0+-circle_radius:
        y_dir = True
        circle_color = (255,255,255)

    # Update new position of circle according to vector_modifier's velocity
    if x_dir:
        circle_position[0] += vector_modifier[0]
    else:
        circle_position[0] -= vector_modifier[0]
    
    if y_dir:
        circle_position[1] += vector_modifier[1]
    else:
        circle_position[1] -= vector_modifier[1]

    # Make system wait to display smoother animation
    pygame.time.wait(20)
    # Update canvas to display changes to user
    pygame.display.flip()

pygame.quit()
