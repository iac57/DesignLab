import pygame
import numpy as np

class PredictionDisplay:
    def __init__(self, B):
        self.B = B  # Total payout for the machines
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
        self.GOLD = (255, 215, 0)  # Gold color for win
        self.LIGHT_GRAY = (220, 220, 220)
        
        # Font setup
        self.font = pygame.font.SysFont('Arial', 24)
        self.title_font = pygame.font.SysFont('Arial', 36)
        self.result_font = pygame.font.SysFont('Arial', 48, bold=True)  # Bold font for win/loss
        
        # Initial state
        self.mocap_prediction = None
        self.behavioral_prediction = None
        self.actual_machine = None
        self.payouts = [self.B/4, self.B/4, self.B/4, self.B/4]  # Initial equal payouts
        self.won = None  # Store win/loss status
        
        # Box dimensions
        self.box_width = 300
        self.box_height = 100
        self.mocap_box_color = self.LIGHT_GRAY
        self.behavioral_box_color = self.LIGHT_GRAY
        
    def check(self, mocap_pred, behavioral_pred, actual_machine, won):
        self.actual_machine = actual_machine
        self.won = won  # Store win/loss status
        
        # Update box colors based on prediction accuracy
        if actual_machine is not None:
            # Update MoCap box color
            if mocap_pred == actual_machine:
                self.mocap_box_color = self.GREEN
            else:
                self.mocap_box_color = self.RED
                
            # Update Behavioral box color
            if behavioral_pred == actual_machine:
                self.behavioral_box_color = self.GREEN
            elif behavioral_pred is None:
                self.behavioral_box_color = self.LIGHT_GRAY
            else:
                self.behavioral_box_color = self.RED
        
        self.draw()
        
    def display_prediction(self, mocap_pred, behavioral_pred, actual_machine, payouts):
        self.mocap_prediction = mocap_pred
        self.behavioral_prediction = behavioral_pred
        self.actual_machine = actual_machine
        self.payouts = payouts
        self.won = None  # Reset win/loss status for new prediction
        self.behavioral_box_color = self.LIGHT_GRAY  # Reset color for new prediction
        self.mocap_box_color = self.LIGHT_GRAY  # Reset color for new prediction
        self.draw()

    def draw(self):
        # Clear screen
        self.screen.fill(self.WHITE)
        
        # Draw title
        title = self.title_font.render("Prediction Display", True, self.BLACK)
        self.screen.blit(title, (self.WIDTH//2 - title.get_width()//2, 20))
        
        # Draw prediction boxes
        y_offset = 100
        
        # MoCap prediction box
        pygame.draw.rect(self.screen, self.mocap_box_color, 
                         (50, y_offset, self.box_width, self.box_height))
        pygame.draw.rect(self.screen, self.BLACK, 
                         (50, y_offset, self.box_width, self.box_height), 2)
        
        if self.mocap_prediction is not None:
            mocap_text = self.font.render(f"MoCap Prediction:", True, self.BLACK)
            self.screen.blit(mocap_text, (60, y_offset + 15))
            
            pred_text = self.font.render(f"Machine {self.mocap_prediction}", True, self.BLUE)
            self.screen.blit(pred_text, (60, y_offset + 50))
        
        # Behavioral prediction box
        pygame.draw.rect(self.screen, self.behavioral_box_color, 
                         (400, y_offset, self.box_width, self.box_height))
        pygame.draw.rect(self.screen, self.BLACK, 
                         (400, y_offset, self.box_width, self.box_height), 2)
        
        if self.behavioral_prediction is not None:
            behavioral_text = self.font.render(f"Behavioral Prediction:", True, self.BLACK)
            self.screen.blit(behavioral_text, (410, y_offset + 15))
            
            pred_text = self.font.render(f"Machine {self.behavioral_prediction}", True, self.BLUE)
            self.screen.blit(pred_text, (410, y_offset + 50))
        
        # Draw actual machine played
        if self.actual_machine is not None:
            if self.actual_machine == 0:
                actual_text = self.font.render("Waiting for play...", True, self.BLACK)
            else:
                actual_text = self.font.render(f"Actual Machine Played: Machine {self.actual_machine}", True, self.BLACK)
            self.screen.blit(actual_text, (self.WIDTH//2 - actual_text.get_width()//2, y_offset + 120))
        
        # Display WON or LOST based on won status
        if self.won is not None and self.actual_machine != 0:
            if self.won:
                # Create animated gold text for WIN
                # Use sine wave to oscillate color brightness
                t = pygame.time.get_ticks() / 500  # Time-based animation
                color_intensity = int(200 + 55 * abs(np.sin(t)))
                win_color = (color_intensity, color_intensity, 0)  # Golden yellow
                win_text = self.result_font.render("WON!", True, win_color)
                # Add glow effect (draw the text multiple times with slight offsets)
                for offset in range(1, 4):
                    glow = self.result_font.render("WON!", True, (255, 255, 100, 50//offset))
                    self.screen.blit(glow, (self.WIDTH//2 - win_text.get_width()//2 + offset, y_offset + 160 + offset))
                    self.screen.blit(glow, (self.WIDTH//2 - win_text.get_width()//2 - offset, y_offset + 160 - offset))
                self.screen.blit(win_text, (self.WIDTH//2 - win_text.get_width()//2, y_offset + 160))
            else:
                # Create red, slightly animated text for LOST
                t = pygame.time.get_ticks() / 800  # Slower animation
                color_intensity = int(200 + 55 * abs(np.sin(t)))
                lose_color = (color_intensity, 0, 0)  # Red
                lose_text = self.result_font.render("LOST", True, lose_color)
                self.screen.blit(lose_text, (self.WIDTH//2 - lose_text.get_width()//2, y_offset + 160))
        
        # Draw payouts
        y_offset += 220  # Increased offset to make room for WON/LOST text
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
                pygame.quit()
                return False
        return True
