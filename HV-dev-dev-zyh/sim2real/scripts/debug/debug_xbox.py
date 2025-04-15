import pygame
import sys

# Initialize pygame
pygame.init()

# Set up the joystick
if pygame.joystick.get_count() > 0:
    joystick = pygame.joystick.Joystick(0)
    joystick.init()
    print(f"Joystick detected: {joystick.get_name()}")
else:
    print("No joystick detected")
    pygame.quit()
    sys.exit()

# Set up the windows
screen1 = pygame.display.set_mode((600, 400))
screen2 = pygame.display.set_mode((1200, 800))
pygame.display.set_caption("Xbox Joystick Test")

# Font setup
font = pygame.font.Font(None, 36)

# Main loop
running = True
while running:
    screen1.fill((30, 30, 30))  # Set background color

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Get the number of axes and buttons
    num_axes = joystick.get_numaxes()
    num_buttons = joystick.get_numbuttons()
    
    # Display the values of all joystick axes on the screen
    for i in range(num_axes):
        axis_value = joystick.get_axis(i)
        axis_text = font.render(f"Axis {i}: {axis_value:.2f}", True, (255, 255, 255))
        screen1.blit(axis_text, (50, 50 + i * 40))  # Position the text on the screen

    # Display the state of all joystick buttons on the screen
    for i in range(num_buttons):
        button_value = joystick.get_button(i)
        button_text = font.render(f"Button {i}: {button_value}", True, (255, 255, 255))
        screen2.blit(button_text, (300, 50 + i * 40))

    # Refresh the screen
    pygame.display.flip()

    # Add a delay to prevent the screen from refreshing too quickly
    pygame.time.wait(100)

pygame.quit()
