from typing import Tuple

def calculate_distance(flash_time: float, boom_time: float, temperature: float, frame_rate: float = 25.0) -> Tuple[float, float]:
    """Calculate the distance and error based on flash time, boom time, and temperature."""
    # Speed of sound in air depending on temperature (meters/second)
    speed_of_sound = 331 + 0.6 * temperature
    
    # Compute distance and error estimate
    distance = (boom_time - flash_time) * speed_of_sound
    error = (0.5 / frame_rate) * speed_of_sound
    
    return distance, error

def print_distance(flash_time: float, boom_time: float, temperature: float, distance: float, error: float) -> None:
    """Output distance calculation to terminal."""
    speed_of_sound = 331 + 0.6 * temperature
    print(f"( {boom_time:.2f} - {flash_time:.2f} ) [s] * {speed_of_sound:.0f} [m/s] @ {temperature:.0f} [deg C]")
    print(f"  = {distance:.0f} +/- {error:.0f} meters")

if __name__ == "__main__":
    # Optional parameters
    frame_rate_val = 25.0 # Video frame rate (frames/second)
    
    # Get input parameters
    try:
        flash_val = float(input('Enter time of flash: '))
        boom_val = float(input('Enter time of boom: '))
        temp_val = float(input('Enter local temperature in degrees C: '))
        
        dist, err = calculate_distance(flash_val, boom_val, temp_val, frame_rate_val)
        print_distance(flash_val, boom_val, temp_val, dist, err)
    except ValueError:
        print("Invalid input. Please enter valid numbers.")
