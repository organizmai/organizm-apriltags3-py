# Generator that goes through a full cycle
def cycle(seed: int, sample_size: int, increment: int):
    nb = seed
    for i in range(sample_size):
        nb = (nb + increment) % sample_size
        yield nb

# Example values
seed = 17
sample_size = 100
increment = 13

# Print all the numbers
print(list(cycle(seed, sample_size, increment)))

# Verify that all numbers were generated correctly
assert set(cycle(seed, sample_size, increment)) == set(range(sample_size))

