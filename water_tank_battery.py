class Battery:
    """(Way too) simple battery."""
    def __init__(self, capacity_in_Ah, battery_voltage = 10, charge_level_in_Ah=0):
        self.capacity_in_Ah = capacity_in_Ah
        self.battery_voltage = battery_voltage
        self.charge_level_in_Ah = charge_level_in_Ah

    def update(self, energy):
        """Can be called during simulation to feed or draw energy.
        
        If `energy` is positive the battery is charged.
        If `energy` is negative the battery is discharged.
        
        Returns the excess energy after the update:
        - Positive if your battery is fully charged
        - Negative if your battery is empty
        - else 0
        """
        energy_in_Ah = energy / self.battery_voltage
        
        self.charge_level_in_Ah += energy_in_Ah
        excess_energy = 0
        
        if self.charge_level_in_Ah < 0:
            excess_energy = self.charge_level_in_Ah * self.battery_voltage
            self.charge_level_in_Ah = 0
        elif self.charge_level_in_Ah > self.capacity_in_Ah:
            excess_energy = (self.charge_level_in_Ah - self.capacity_in_Ah) * self.battery_voltage
            self.charge_level_in_Ah = self.capacity_in_Ah

        return excess_energy