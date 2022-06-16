from abc import ABC, abstractmethod


class Battery(ABC):

    def __init__(self, capacity: float, soc: float = 0, **kwargs):
        """Base class for batteries

        Args:
            capacity: Capacity in Ah
            soc: Initial state of charge from 0 to 1 (0% to 100%)
            **kwargs: Subclasses can implement additional parameters
        """
        self.capacity = capacity
        self.voltage = 10
        self.soc_in_ah = soc * self.capacity
        self._last_update_time = 0

    @property
    def soc(self):
        """Returns the state of charge from 0 to 1 (0% to 100%)"""
        return self.soc_in_ah / self.capacity

    def update(self, power: float, simulation_time: float):
        """Called during the simulation to (dis)charge the battery.

        Args:
            power: Power in Watts. If positive the battery is being charged, otherwise discharged.
            simulation_time: Current time in the simulation in seconds. This is used to compute the amount of time the
                battery is being (dis)charged based on when it was last updated.

        Returns:
            The excess energy in Watt-seconds.
            - 0 if the battery was successfully (dis)charged
            - negative if the battery is now empty and not all requested energy could be discharged
            - positive if the battery is now full and not all provided energy could be charged
        """
        time_since_last_update = simulation_time - self._last_update_time
        self._last_update_time = simulation_time
        if power == 0 or time_since_last_update == 0:
            return 0
        else:
            return self._update_internal(power, time_since_last_update)

    @abstractmethod
    def _update_internal(self, power: float, time_since_last_update: float):
        """To be implemented by subclasses."""


class SimpleBattery(Battery):
    """Simplified "water tank" battery without any losses."""

    def _update_internal(self, power: float, time_since_last_update: float):
        charge_in_ah = power * time_since_last_update / self.voltage

        new_soc_in_ah = self.soc_in_ah + charge_in_ah
        excess_energy = 0
        
        if new_soc_in_ah < 0:
            excess_energy = new_soc_in_ah * self.voltage
            new_soc_in_ah = 0
        elif new_soc_in_ah > self.capacity:
            excess_energy = (new_soc_in_ah - self.capacity) * self.voltage
            new_soc_in_ah = self.capacity

        self.soc_in_ah = new_soc_in_ah
        return excess_energy
