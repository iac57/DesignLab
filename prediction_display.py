import pygame
import numpy as np

class PredictionDisplay:
    def __init__(self):
        # Initialize pygame
        pygame.init()
        
        # Screen setup
        self.WIDTH, self.HEIGHT = 800, 600
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        pygame.display.set_caption("Prediction Display")
        
        # Colors
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.GREEN = (0, 255, 0)
        self.RED = (255, 0, 0)
        self.BLUE = (0, 0, 255)
        
        # Font setup
        self.font = pygame.font.SysFont('Arial', 24)
        self.title_font = pygame.font.SysFont('Arial', 36)
        
        # Initial state
        self.mocap_prediction = None
        self.behavioral_prediction = None
        self.actual_machine = None
        self.payouts = [0.25, 0.25, 0.25, 0.25]  # Initial equal payouts
        self.background_color = self.WHITE
        
    def update(self, mocap_pred, behavioral_pred, actual_machine, payouts):
        self.mocap_prediction = mocap_pred
        self.behavioral_prediction = behavioral_pred
        self.actual_machine = actual_machine
        self.payouts = payouts
        
        # Update background color based on prediction accuracy
        if actual_machine is not None:
            if actual_machine == mocap_pred or actual_machine == behavioral_pred:
                self.background_color = self.GREEN
            else:
                self.background_color = self.RED
        else:
            self.background_color = self.WHITE
        
        self.draw()
        
    def draw(self):
        # Clear screen
        self.screen.fill(self.background_color)
        
        # Draw title
        title = self.title_font.render("Prediction Display", True, self.BLACK)
        self.screen.blit(title, (self.WIDTH//2 - title.get_width()//2, 20))
        
        # Draw predictions
        y_offset = 100
        if self.mocap_prediction is not None:
            mocap_text = self.font.render(f"MoCap Prediction: Machine {self.mocap_prediction}", True, self.BLUE)
            self.screen.blit(mocap_text, (50, y_offset))
            
        if self.behavioral_prediction is not None:
            behavioral_text = self.font.render(f"Behavioral Prediction: Machine {self.behavioral_prediction}", True, self.BLUE)
            self.screen.blit(behavioral_text, (50, y_offset + 40))
            
        if self.actual_machine is not None:
            actual_text = self.font.render(f"Actual Machine Played: Machine {self.actual_machine}", True, self.BLACK)
            self.screen.blit(actual_text, (50, y_offset + 80))
        
        # Draw payouts
        y_offset += 150
        payout_title = self.font.render("Machine Payouts:", True, self.BLACK)
        self.screen.blit(payout_title, (50, y_offset))
        
        for i, payout in enumerate(self.payouts):
            payout_text = self.font.render(f"Machine {i+1}: {payout:.2f}", True, self.BLACK)
            self.screen.blit(payout_text, (50, y_offset + 40 + i*30))
        
        # Update display
        pygame.display.flip()
        
    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
        return True 