
class Sensor():
    def __init__(
        self,
        freq,
        pyb_freq,
        client_id
    ):
        self.freq = float(freq)
        self.pyb_freq = pyb_freq
        self.client_id = client_id
        self._update_counter = 0
        self._first_update_step = 0
        self.timestamp = 0.0 
    
        
    def should_update(self, step_counter):
        # first update
        if step_counter == 0:
            self._update_counter = 0
            self._first_update_step = 0

        # frequency control
        if not round(1.0 * self._update_counter / (step_counter - self._first_update_step + 1e-6) * self.pyb_freq) <= self.freq:
            return False
        
        if abs(1.0 * self._update_counter / (step_counter - self._first_update_step + 1e-6) * self.pyb_freq - self.freq) <= 0.01 * self.freq:
            self._first_update_step = step_counter
            self._update_counter = 0
        self.timestamp = step_counter / self.pyb_freq
        self._update_counter += 1
        return True