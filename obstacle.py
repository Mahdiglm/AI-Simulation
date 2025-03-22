"""
Obstacle class for the AI Dot Collector simulation.
"""
import pygame
from typing import Tuple
from constants import GRAY

class Obstacle:
    """A rectangular obstacle in the simulation."""
    
    def __init__(self, x: float, y: float, width: float, height: float):
        """Initialize an obstacle with position and size."""
        self.rect = (x, y, width, height)
        
    def draw(self, surface: pygame.Surface) -> None:
        """Draw the obstacle on the given surface."""
        pygame.draw.rect(surface, GRAY, self.rect)
        
    def contains_point(self, x: float, y: float) -> bool:
        """Check if a point is inside the obstacle."""
        ox, oy, width, height = self.rect
        return (ox <= x <= ox + width and oy <= y <= oy + height)
        
    def get_sides(self) -> Tuple[Tuple[Tuple[float, float], Tuple[float, float]], ...]:
        """Get the four sides of the obstacle as line segments."""
        x, y, width, height = self.rect
        return (
            ((x, y), (x + width, y)),         # Top side
            ((x + width, y), (x + width, y + height)),  # Right side
            ((x, y + height), (x + width, y + height)),  # Bottom side
            ((x, y), (x, y + height))         # Left side
        ) 