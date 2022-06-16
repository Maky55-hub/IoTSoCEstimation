import pandas as pd
import simpy


def charging_process(env, battery, power_production, power_consumption, frequency, measurements):
    for production, consumption in zip(power_production, power_consumption):
        delta = production - consumption
        excess_energy = battery.update(delta, simulation_time=env.now)
        measurements.append({
            "production_power": production,
            "consumption_power": consumption,
            "delta_power": delta,
            "excess_energy": excess_energy,
            "soc": battery.soc,
        })
        yield env.timeout(frequency)


def simulate(battery, power_production, power_consumption, frequency):
    measurements = []  # this will contain the resulting measurements
    env = simpy.Environment()
    env.process(charging_process(env, battery, power_production, power_consumption, frequency, measurements))
    env.run()  # env.run(until=100) runs for 100 timesteps only
    measurements_df = pd.DataFrame(measurements)  # , index=solar.index
    return measurements_df
